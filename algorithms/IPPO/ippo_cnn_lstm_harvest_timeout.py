"""
Based on PureJaxRL & jaxmarl Implementation of PPO
CNN + LSTM actor-critic version of ippo_cnn_harvest_timeout.py
"""
import sys
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
# from flax.training import checkpoints
import distrax
import functools
from gymnax.wrappers.purerl import FlattenObservationWrapper
import socialjax
from socialjax.wrappers.baselines import LogWrapper, SVOLogWrapper
import hydra
from omegaconf import OmegaConf
import wandb
import copy
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import json
from datetime import datetime

import imageio

# Absolute path of this file, resolved at import time (before hydra can change the cwd).
# Used to launch the evaluation subprocess (see eval_callback / _eval_worker_main).
SCRIPT_PATH = os.path.abspath(__file__)
# sys.path as seen by this process (cwd entry made absolute), handed to the eval subprocess so it
# can import socialjax the same way even when it is only importable through the cwd (python -m ...)
LAUNCH_SYS_PATH = [os.path.abspath(p) if p else os.getcwd() for p in sys.path]


def get_algo_name(config):
    base = "ippo"
    im = config.get("ENV_KWARGS", {}).get("intrinsic_motivation")
    if not im or not im.get("im_type"):
        return base
    im_type = im["im_type"]
    if im_type == "svo":
        w = im.get("im_w", "")
        angle = int(im.get("im_ideal_angle_degrees", 0))
        return f"{base}_svo_w{w}_a{angle}"
    if im_type == "ia":
        alpha = im.get("im_alpha", "")
        beta = im.get("im_beta", "")
        return f"{base}_ia_a{alpha}_b{beta}"
    if im_type == "social_conformity":
        w = im.get("im_w", 1.0)
        return f"{base}_conformity_w{w}"
    return f"{base}_{im_type}"

class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class ScannedLSTM(nn.Module):
    """
    LSTM unrolled over the leading (time) axis with jax.lax.scan (via nn.scan).

    carry : (h, c), each (B, hidden_size)
    x     : (embedding (T, B, D), reset (T, B))  -- reset[t] is True when the observation at
            step t is the first of a new episode, so the state is zeroed BEFORE the cell runs.
    """
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        h, c = carry
        embedding, reset = x
        keep = (1.0 - reset.astype(jnp.float32))[:, None]
        h = h * keep
        c = c * keep
        # NOTE: Flax's LSTM carry is ordered (c, h)
        (new_c, new_h), y = nn.OptimizedLSTMCell(features=h.shape[-1])((c, h), embedding)
        return (new_h, new_c), y


class ActorCriticLSTM(nn.Module):
    action_dim: Sequence[int]
    activation: str = "relu"
    hidden_size: int = 128  # LSTM hidden size

    @nn.compact
    def __call__(self, x, hidden_state):
        """
        Args:
            x: (obs, reset)
               obs   (T, B, H, W, C)
               reset (T, B) bool -- True if obs[t] starts a new episode (previous step was done)
            hidden_state: (h, c), each (B, hidden_size), the state BEFORE processing x[0]
        Returns:
            pi: action distribution, batch shape (T, B)
            value: (T, B)
            new_hidden_state: (h, c) after the last step
        For acting use T = 1 (add obs[None], reset[None], squeeze(0) the outputs).
        """
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs, reset = x
        T, B = obs.shape[0], obs.shape[1]

        # CNN embedding, applied to all T*B frames at once
        embedding = CNN(self.activation)(obs.reshape((T * B,) + obs.shape[2:]))
        embedding = embedding.reshape((T, B, -1))

        # LSTM over time, hidden state reset at episode boundaries
        new_hidden_state, lstm_out = ScannedLSTM()(
            hidden_state, (embedding, reset)
        )

        # Actor head
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(lstm_out)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic head
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(lstm_out)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1), new_hidden_state


class Transition(NamedTuple):
    done: jnp.ndarray       # episode ended AFTER this step (used by GAE)
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    prev_done: jnp.ndarray  # episode ended BEFORE this step, i.e. obs is a first obs (LSTM reset flag)


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_dict(x: dict, agent_list, num_actors):
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def get_single_agent_reward(agent_obs, agent_action, agent_idx, all_params_stacked, network, num_agents, weight, is_critical):
    """
    TEST VERSION: punishes the agent for taking action 2 (move_forward) while
    in a critical state (is_critical=True, i.e. standing in front of an
    endangered apple with too few neighbors), regardless of group consensus.

    Used to sanity-check that is_critical flows correctly env -> info ->
    here, and that the penalty actually reaches PPO's training reward,
    before re-enabling the full cross-policy consensus logic below.

    agent_obs, all_params_stacked, network, num_agents are unused here but
    kept in the signature so the vmap call sites (training _env_step and
    the eval loop) don't need to change.
    """
    def perform_penalty(_):
        return -1.0 * weight * (agent_action == 2).astype(jnp.float32)

    return jax.lax.cond(is_critical, perform_penalty, lambda _: 0.0, operand=None)

"""
# NOTE (LSTM version): this disabled block was written for the feed-forward network. Before
# re-enabling it, network.apply must be called with ((obs[None, None], reset), hidden) and its
# 3 outputs (pi, value, hidden) unpacked, and a hidden state has to be chosen for each of the
# num_agents policies (the cross-policy check has no natural hidden state for "what would agent j
# do on agent i's view"). Left unchanged until the test version above is validated.
def get_single_agent_reward(agent_obs, agent_action, agent_idx, all_params_stacked, network, num_agents, weight, is_critical):
    # Calculates reward for ONE agent in ONE environment.

    # is_critical is supplied by the caller — read from info["is_critical"]
    # returned by the env, so criticality logic lives only in the env.
    
    def perform_cross_check(_):
        # FIX: Ensure agent_obs has a batch dimension for the CNN's reshape layer
        obs_batched = jnp.expand_dims(agent_obs, axis=0)

        # Cross-policy check: What do all agents think of this view?
        # all_pi.probs will have shape [num_agents, 1, action_dim]
        all_pi, _ = jax.vmap(network.apply, in_axes=(0, None))(all_params_stacked, obs_batched)

        # Remove the dummy batch dimension to get [num_agents, action_dim]
        probs = all_pi.probs.squeeze(axis=1)

        # Calculate group norm (excluding the acting agent)
        mask = jnp.arange(num_agents) != agent_idx
        group_avg_probs = jnp.mean(probs, axis=0, where=mask[:, None])

        norm_action = jnp.argmax(group_avg_probs)
        consensus = jnp.max(group_avg_probs)

        is_deviating = (agent_action != norm_action)
        
        return -1.0 * weight * (is_deviating.astype(jnp.float32)) * consensus

    return jax.lax.cond(is_critical, perform_cross_check, lambda _: 0.0, operand=None)
"""

def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if config["PARAMETER_SHARING"]:
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    else:
        config["NUM_ACTORS"] = config["NUM_ENVS"]
    # int(): the yaml value 1e8 is parsed as a float, and scan lengths must be ints
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    config["EVAL_INTERVAL"] = config.get("EVAL_INTERVAL", 50000)
    config["EVAL_EPISODES"] = config.get("EVAL_EPISODES", 10)

    # With an LSTM, minibatches are made of whole trajectories (NUM_STEPS long), split along the
    # actor axis, so the actors must divide evenly into the minibatches.
    assert config["NUM_ACTORS"] % config["NUM_MINIBATCHES"] == 0, (
        f"NUM_ACTORS ({config['NUM_ACTORS']}) must be divisible by NUM_MINIBATCHES "
        f"({config['NUM_MINIBATCHES']}): the LSTM minibatches split whole trajectories by actor"
    )

    env = LogWrapper(env, replace_info=False)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        hidden_size = config.get("LSTM_HIDDEN_SIZE", 128)

        # INIT NETWORK
        if config["PARAMETER_SHARING"]:
            network = ActorCriticLSTM(
                env.action_space().n,
                activation=config["ACTIVATION"],
                hidden_size=hidden_size,
            )
        else:
            network = [
                ActorCriticLSTM(
                    env.action_space().n,
                    activation=config["ACTIVATION"],
                    hidden_size=hidden_size,
                ) for _ in range(env.num_agents)
            ]

        rng, _rng = jax.random.split(rng)
        # dummy input for init: (T=1, B=1) sequence + zero hidden state
        init_x = (
            jnp.zeros((1, 1, *(env.observation_space()[0]).shape)),
            jnp.zeros((1, 1), dtype=bool),
        )
        init_hidden_1 = (jnp.zeros((1, hidden_size)), jnp.zeros((1, hidden_size)))

        if config["PARAMETER_SHARING"]:
            network_params = network.init(_rng, init_x, init_hidden_1)
        else:
            network_params = [
                network[i].init(_rng, init_x, init_hidden_1)
                for i in range(env.num_agents)
            ]
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        if config["PARAMETER_SHARING"]:
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
        else:
            train_state = [TrainState.create(
                apply_fn=network[i].apply,
                params=network_params[i],
                tx=tx,
            ) for i in range(env.num_agents)]

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # INIT LSTM HIDDEN STATES + "previous step was done" FLAGS
        if config["PARAMETER_SHARING"]:
            init_hidden = (
                jnp.zeros((config["NUM_ACTORS"], hidden_size)),
                jnp.zeros((config["NUM_ACTORS"], hidden_size)),
            )
            init_done = jnp.zeros((config["NUM_ACTORS"],), dtype=bool)
        else:
            init_hidden = [
                (
                    jnp.zeros((config["NUM_ENVS"], hidden_size)),
                    jnp.zeros((config["NUM_ENVS"], hidden_size)),
                ) for _ in range(env.num_agents)
            ]
            init_done = jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=bool)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # hidden state at the START of this rollout: the update re-runs the LSTM from it
            rollout_init_hidden = runner_state[4]

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, last_hidden, update_step, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                # obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(-1, *env.observation_space().shape)

                if config["PARAMETER_SHARING"]:
                    obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                    print("input_obs_shape", obs_batch.shape)
                    # single step: add time axis (T=1), drop it from the outputs
                    pi, value, new_hidden = network.apply(
                        train_state.params, (obs_batch[None], last_done[None]), last_hidden
                    )
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)
                    action, log_prob, value = action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)
                    env_act = unbatchify(
                        action, env.agents, config["NUM_ENVS"], env.num_agents
                    )
                else:
                    obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                    env_act = {}
                    log_prob = []
                    value = []
                    new_hidden = []
                    for i in range(env.num_agents):
                        print("input_obs_shape", obs_batch[i].shape)
                        rng, _rng_agent = jax.random.split(rng)
                        pi, value_i, new_hidden_i = network[i].apply(
                            train_state[i].params,
                            (obs_batch[i][None], last_done[i][None]),
                            last_hidden[i],
                        )
                        action = pi.sample(seed=_rng_agent)
                        log_prob.append(pi.log_prob(action).squeeze(0))
                        env_act[env.agents[i]] = action.squeeze(0)
                        value.append(value_i.squeeze(0))
                        new_hidden.append(new_hidden_i)

                # env_act = {k: v.flatten() for k, v in env_act.items()}
                env_act = [v for v in env_act.values()]

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                # Define a function to conditionally reshape based on dimensionality
                def reshape_if_scalar(x):
                    # Check if the array is multi-dimensional (like agent_locs, which is (N, 3))
                    # Multi-dimensional arrays are returned as-is.
                    if x.ndim > 1:
                        return x
                    # If it's a 1D vector (like freeze status, apple count), reshape it to (N, 1)
                    # This is safe and mathematically valid.
                    return x.reshape((config["NUM_ACTORS"]), 1)

                im_cfg = config.get("ENV_KWARGS", {}).get("intrinsic_motivation", {})
                if im_cfg and im_cfg.get("im_type") == "social_conformity" and not config["PARAMETER_SHARING"]:
                    w = im_cfg.get("im_w", 1.0)
                    num_agents = env.num_agents
                    num_envs = config["NUM_ENVS"]

                    obs_t = last_obs
                    act_t = jnp.array(env_act).T
                    # index map [num_envs, num_agents]
                    idx_t = jnp.broadcast_to(jnp.arange(num_agents), (num_envs, num_agents))

                    # 2. Flatten everything to a single batch of num_envs * num_agents
                    obs_flat = obs_t.reshape((-1,) + tuple(env.observation_space()[0].shape))
                    act_flat = act_t.reshape(-1)
                    idx_flat = idx_t.reshape(-1)

                    is_critical_flat = info["is_critical"].reshape(-1)

                    # 3. Stack parameters for the vmap
                    params_list = [train_state[i].params for i in range(num_agents)]
                    all_params_stacked = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *params_list)

                    # 4. Map over the flattened instances
                    intrinsic_rewards_flat = jax.vmap(
                        get_single_agent_reward,
                        in_axes=(0, 0, 0, None, None, None, None, 0)
                    )(
                        obs_flat,
                        act_flat,
                        idx_flat,
                        all_params_stacked,
                        network[0],
                        num_agents,
                        w,
                        is_critical_flat,
                    )

                    # 5. Reshape back to [num_envs, num_agents] to add to environment reward
                    intrinsic_rewards = intrinsic_rewards_flat.reshape(num_envs, num_agents)
                    reward = reward + intrinsic_rewards

                if config["PARAMETER_SHARING"]:
                    done_batch = batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze()
                    new_last_done = done_batch.astype(bool)
                    info = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                    transition = Transition(
                        done_batch,
                        action,
                        value,
                        batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                        log_prob,
                        obs_batch,
                        info,
                        last_done,
                        )
                else:
                    transition = []
                    done = [v for v in done.values()]
                    new_last_done = jnp.stack(done[:env.num_agents], axis=0).astype(bool)
                    for i in range(env.num_agents):
                        info_i = {
                            key: jax.tree_util.tree_map(
                                reshape_if_scalar,
                                value[:, i]
                            ) for key, value in info.items()
                        }
                        transition.append(Transition(
                            done[i],
                            env_act[i],
                            value[i],
                            reward[:,i],
                            log_prob[i],
                            obs_batch[i],
                            info_i,
                            last_done[i],
                        ))
                # The LSTM state is carried as-is; the reset at an episode boundary is applied
                # inside the network at the next step through the `last_done` flag.
                runner_state = (train_state, env_state, obsv, new_last_done, new_hidden, update_step, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, last_hidden, update_step, rng = runner_state
            if config["PARAMETER_SHARING"]:
                last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                _, last_val, _ = network.apply(
                    train_state.params, (last_obs_batch[None], last_done[None]), last_hidden
                )
                last_val = last_val.squeeze(0)
            else:
                last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                last_val = []
                for i in range(env.num_agents):
                    _, last_val_i, _ = network[i].apply(
                        train_state[i].params,
                        (last_obs_batch[i][None], last_done[i][None]),
                        last_hidden[i],
                    )
                    last_val.append(last_val_i.squeeze(0))
                last_val = jnp.stack(last_val, axis=0)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    # reward_mean = jnp.mean(reward, axis=0)
                    # # reward_std = jnp.std(reward, axis=0) + 1e-8
                    # reward = (reward - reward_mean)# / reward_std
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value
            if config["PARAMETER_SHARING"]:
                advantages, targets = _calculate_gae(traj_batch, last_val)
            else:
                advantages = []
                targets = []
                for i in range(env.num_agents):
                    advantages_i, targets_i = _calculate_gae(traj_batch[i], last_val[i])
                    advantages.append(advantages_i)
                    targets.append(targets_i)
                advantages = jnp.stack(advantages, axis=0)
                targets = jnp.stack(targets, axis=0)
            # UPDATE NETWORK
            def _update_epoch(update_state, unused, i):
                def _update_minbatch(train_state, batch_info, network_used):
                    init_hidden_mb, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hidden_mb, traj_batch, gae, targets, network_used):
                        # RERUN NETWORK over whole trajectories (T, B_mb), starting from the stored
                        # rollout-start hidden state and resetting at episode boundaries (BPTT)
                        pi, value, _ = network_used.apply(
                            params, (traj_batch.obs, traj_batch.prev_done), init_hidden_mb
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)


                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                            train_state.params, init_hidden_mb, traj_batch, advantages, targets, network_used
                        )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hidden_ep, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                # Shuffle and split along the ACTOR axis (never along time): every minibatch is
                # a set of whole NUM_STEPS-long trajectories with its own initial hidden state.
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                shuffled_hidden = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), init_hidden_ep
                )  # (B, hidden)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), (traj_batch, advantages, targets)
                )  # (T, B, ...)

                def _split_actors(x):
                    # (T, B, ...) -> (NUM_MINIBATCHES, T, B // NUM_MINIBATCHES, ...)
                    x = x.reshape((x.shape[0], config["NUM_MINIBATCHES"], -1) + x.shape[2:])
                    return jnp.swapaxes(x, 0, 1)

                minibatch_hidden = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["NUM_MINIBATCHES"], -1) + x.shape[1:]),
                    shuffled_hidden,
                )  # (NUM_MINIBATCHES, B_mb, hidden)
                minibatch_traj, minibatch_adv, minibatch_tgt = jax.tree_util.tree_map(
                    _split_actors, shuffled_batch
                )
                minibatches = (minibatch_hidden, minibatch_traj, minibatch_adv, minibatch_tgt)

                if config["PARAMETER_SHARING"]:
                    train_state, total_loss = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network), train_state, minibatches
                    )
                else:
                    train_state, total_loss = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network[i]), train_state, minibatches
                    )

                update_state = (train_state, init_hidden_ep, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            if config["PARAMETER_SHARING"]:
                update_state = (train_state, rollout_init_hidden, traj_batch, advantages, targets, rng)
                update_state, loss_info = jax.lax.scan(
                    lambda state, unused: _update_epoch(state, unused, 0), update_state, None, config["UPDATE_EPOCHS"]
                )
                train_state = update_state[0]
                metric = traj_batch.info
                rng = update_state[-1]
            else:
                update_state_dict = []
                metric = []
                for i in range(env.num_agents):
                    update_state = (train_state[i], rollout_init_hidden[i], traj_batch[i], advantages[i], targets[i], rng)
                    update_state, loss_info = jax.lax.scan(
                        lambda state, unused: _update_epoch(state, unused, i), update_state, None, config["UPDATE_EPOCHS"]
                    )
                    update_state_dict.append(update_state)
                    train_state[i] = update_state[0]
                    metric_i = traj_batch[i].info
                    metric_i['loss'] = loss_info[0]
                    metric.append(metric_i)
                    rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)

            def eval_callback(step, params):
                import subprocess, pickle, tempfile, json
                import numpy as np
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n{'='*60}\nRunning evaluation at step {step}...\n{'='*60} at {timestamp}")

                # Write params and config to temp files
                params = jax.tree_util.tree_map(np.asarray, params)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
                    pickle.dump(params, f)
                    params_path = f.name

                output_dir = f"{config['RUN_OUTPUT_DIR']}/evaluation"
                os.makedirs(output_dir, exist_ok=True)
                final_step = config["NUM_UPDATES"] * config["NUM_STEPS"] * config["NUM_ENVS"]
                is_final_step = (int(step) >= final_step)

                # The video (final step only) is rendered by the worker process too: running
                # jitted JAX code inside this host callback can deadlock the training process.
                eval_config = {
                    **config,
                    "_eval_seed": config["SEED"] + int(step) + 999999,
                    "_make_video": bool(is_final_step),
                    "_video_dir": output_dir,
                    "_step": int(step),
                }
                with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as f:
                    json.dump(eval_config, f)
                    config_path = f.name

                result_path = params_path + '_result.json'

                # Run eval in completely separate process with its own CUDA context.
                # The worker is this same file started with --eval-worker (see __main__).
                proc_env = {
                    **os.environ,
                    'LD_PRELOAD': os.environ.get('LD_PRELOAD', ''),
                    # the training process already holds the GPU memory pool
                    'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
                    'PYTHONPATH': os.pathsep.join(
                        LAUNCH_SYS_PATH + [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]
                    ),
                }
                subprocess.run([
                    sys.executable,
                    SCRIPT_PATH,
                    '--eval-worker',
                    params_path, config_path, result_path
                ], check=True, env=proc_env)

                with open(result_path, 'r') as f:
                    eval_results = json.load(f)

                os.unlink(params_path)
                os.unlink(config_path)
                os.unlink(result_path)

                summary_data = eval_results["summary"]
                raw_data = {k: np.array(v) for k, v in eval_results["raw"].items()}

                # Print summary to console
                for key, values in summary_data.items():
                    print(f"  {key}: {np.mean(values):.2f} ± {np.std(values):.2f}")

                # --- LOGGING TO WANDB ---
                for key, values in summary_data.items():
                    wandb.log({f"eval/{key}_mean": float(np.mean(values)), "eval/step": int(step)})

                env_name = config["ENV_NAME"]
                algo_name = get_algo_name(config)
                seed_name = f"seed_{config['SEED']}"

                # --- PART A: UPDATE GENERAL JSON (Standard Structure) ---
                json_path = os.path.join(output_dir, f"{env_name}_ippo_seed_{config['SEED']}.json")

                if os.path.exists(json_path):
                    with open(json_path, 'r') as f: main_data = json.load(f)
                else:
                    main_data = {"socialjax": {env_name: {algo_name: {seed_name: {}}}}}

                seed_results = main_data["socialjax"][env_name][algo_name][seed_name]
                step_key = f"step_{len([k for k in seed_results.keys() if k.startswith('step_')])}"
                seed_results[step_key] = {"step_count": int(step), **summary_data}
                seed_results["absolute_metrics"] = {k: [float(np.mean(v))] for k, v in summary_data.items()}

                with open(json_path, 'w') as f: json.dump(main_data, f, indent=2)

                # --- PART B: UPDATE AGENT JSONs (Expanded Action Metrics) ---
                for agent_idx in range(config["ENV_KWARGS"]["num_agents"]):
                    agent_algo_name = f"ippo (agent {agent_idx})"
                    agent_path = os.path.join(output_dir, f"{env_name}_ippo_agent_{agent_idx}_seed_{config['SEED']}.json")

                    if os.path.exists(agent_path):
                        with open(agent_path, 'r') as f: agent_data = json.load(f)
                    else:
                        agent_data = {"socialjax": {env_name: {agent_algo_name: {seed_name: {}}}}}

                    # Extract basic metrics
                    agent_metrics = {
                        "return": raw_data["returns"][:, agent_idx].tolist(),
                        "apples_collected": raw_data["apples"][:, agent_idx].tolist(),
                        "endangered_apples_collected": raw_data["endangered_apples_collected"][:, agent_idx].tolist(), 
                        "zap_hits": raw_data["zap_hits"][:, agent_idx].tolist(),
                        "endangered_opportunities": raw_data["endangered_opportunities"][:, agent_idx].tolist(),
                        "zap_opportunities": raw_data["zap_opportunities"][:, agent_idx].tolist(),
                        "zap_successes": raw_data["zap_successes"][:, agent_idx].tolist(),
                        "frozen_steps": raw_data["freeze_steps"][:, agent_idx].tolist(),
                        "sustainability": raw_data["pos_reward_timesteps"][:, agent_idx].tolist(),
                        # Intrinsic motivation tracing (zero when social_conformity not active)
                        "original_rewards": raw_data["original_rewards"][:, agent_idx].tolist(),
                        "shaped_rewards": raw_data["shaped_rewards"][:, agent_idx].tolist(),
                    }

                    # raw_data["actions"] shape: (num_episodes, num_agents, num_actions)
                    # Extract actions for this specific agent: (num_episodes, num_actions)
                    agent_action_counts = raw_data["actions"][:, agent_idx, :]
                    num_actions = agent_action_counts.shape[-1]

                    # Create a separate list of counts for each individual action
                    for action_id in range(num_actions):
                        metric_name = f"action_{action_id}_count"
                        # Get the count of this specific action across all episodes
                        agent_metrics[metric_name] = agent_action_counts[:, action_id].tolist()

                    a_seed_results = agent_data["socialjax"][env_name][agent_algo_name][seed_name]
                    a_step_key = f"step_{len([k for k in a_seed_results.keys() if k.startswith('step_')])}"

                    # Store in identical format
                    a_seed_results[a_step_key] = {"step_count": int(step), **agent_metrics}
                    a_seed_results["absolute_metrics"] = {k: [float(np.mean(v))] for k, v in agent_metrics.items()}

                    with open(agent_path, 'w') as f: json.dump(agent_data, f, indent=2)

                if is_final_step:
                    # mp4 was written by the worker; only the (non-JAX) wandb upload happens here
                    n_agents = config["ENV_KWARGS"]["num_agents"]
                    mp4_path = f"{output_dir}/{n_agents}-agents_seed-{config['SEED']}_step_{int(step)}.mp4"
                    print(f"\nLogging FINAL video for step {step}...")
                    wandb.log({"Episode Video": wandb.Video(mp4_path, caption="Evaluation Episode", format="mp4")})
                else:
                    print(f"\nSkipping video generation for step {step} (Only saved at final step).")

            update_step = update_step + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            if config["PARAMETER_SHARING"]:
                metric["update_step"] = update_step
                metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                jax.debug.callback(callback, metric)
            else:
                for i in range(env.num_agents):
                    metric[i]["update_step"] = update_step
                    metric[i]["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                metric = metric[0]
                jax.debug.callback(callback, metric)

            current_step = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]

            should_eval = (current_step > 0) & ((update_step % config["EVAL_INTERVAL"] == 0) | (update_step == config["NUM_UPDATES"]))
            should_eval = jnp.asarray(should_eval)  # ensure scalar if needed

            def eval_branch(_):
                if config["PARAMETER_SHARING"]:
                    jax.debug.callback(eval_callback, current_step, train_state.params)
                else:
                    params_list = [train_state[i].params for i in range(env.num_agents)]
                    jax.debug.callback(eval_callback, current_step, params_list)
                return runner_state  # must return something (even if it's unchanged)

            def no_eval_branch(_):
                return runner_state

            runner_state = jax.lax.cond(
                    should_eval,
                    eval_branch,
                    no_eval_branch,
                    operand=None,
                )

            runner_state = (train_state, env_state, last_obs, last_done, last_hidden, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, init_done, init_hidden, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

def build_marl_eval_json(config, all_eval_results):
    """
    Build MARL-eval format JSON from evaluation results.

    Args:
        config: experiment configuration
        all_eval_results: dict mapping step_count -> evaluation results

    Returns:
        dict: MARL-eval formatted results
    """
    env_name = config["ENV_NAME"]
    algo_name = get_algo_name(config)
    seed_name = f"seed_{config['SEED']}"

    # Initialize structure
    result = {
        env_name: {
            env_name: {
                algo_name: {
                    seed_name: {}
                }
            }
        }
    }

    seed_results = result[env_name][f"{env_name}"][algo_name][seed_name]

    # Add evaluation checkpoints
    for step_idx, (step_count, eval_data) in enumerate(sorted(all_eval_results.items())):
        seed_results[f"step_{step_idx}"] = {
            "step_count": int(step_count),
            **eval_data
        }

    # Add absolute metrics (final evaluation)
    if all_eval_results:
        final_step = max(all_eval_results.keys())
        final_data = all_eval_results[final_step]

        # Compute mean of each metric across episodes
        seed_results["absolute_metrics"] = {
            key: [float(np.mean(values))]
            for key, values in final_data.items()
        }

    return result


def single_run(config):
    config = OmegaConf.to_container(config)
    algo_name = get_algo_name(config)
    run = wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[algo_name, "LSTM"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'ippo_cnn_lstm_harvest_common_seed{config["SEED"]}'
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config["RUN_OUTPUT_DIR"] = run.dir
    print(f"\n[{timestamp}] Started training...")
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    jax.clear_caches()

    print("\n** Saving Results **")
    filename = f'{config["ENV_NAME"]}_seed{config["SEED"]}'
    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])

    # Save model checkpoints
    if config["PARAMETER_SHARING"]:
        save_path = f"{config['RUN_OUTPUT_DIR']}/checkpoints/individual/{filename}.pkl"
        save_params(train_state, save_path)
        params = load_params(save_path)
    else:
        params = []
        for i in range(config['ENV_KWARGS']['num_agents']):
            save_path = f"{config['RUN_OUTPUT_DIR']}/checkpoints/individual/{filename}_{i}.pkl"
            save_params(train_state[i], save_path)
            params.append(load_params(save_path))

    json_filename = f"{config['ENV_NAME']}_ippo_seed_{config['SEED']}.json"
    json_path = os.path.join(f"{config['RUN_OUTPUT_DIR']}/evaluation", json_filename)

    if os.path.exists(json_path):
        print(f"\n✓ MARL-eval results saved at: {json_path}")

        # Load and print summary
        with open(json_path, 'r') as f:
            data = json.load(f)

        env_name = config["ENV_NAME"]
        algo_name = get_algo_name(config)
        seed_name = f"seed_{config['SEED']}"
        seed_results = data["socialjax"][env_name][algo_name][seed_name]

        num_checkpoints = len([k for k in seed_results.keys() if k.startswith("step_")])

        print(f"\nEvaluation Summary:")
        print(f"  Environment: {config['ENV_NAME']}")
        print(f"  Algorithm: ippo")
        print(f"  Seed: {config['SEED']}")
        print(f"  Checkpoints: {num_checkpoints}")

        wandb.save(json_path)
    else:
        print("⚠ No evaluation results found")

    wandb.finish()

def save_params(train_state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)

    with open(save_path, 'wb') as f:
        pickle.dump(params, f)

def load_params(load_path):
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)


def run_evaluation_episodes(params, config, eval_seed, num_episodes=10):
    """
    Run multiple evaluation episodes in parallel using JAX vmap.
 
    Returns:
        dict: {"summary": {...per-episode group metrics...}, "raw": {...per-agent arrays...}}
    """
    # Create eval environment
    eval_env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    hidden_size = config.get("LSTM_HIDDEN_SIZE", 128)
    if config["PARAMETER_SHARING"]:
        network = ActorCriticLSTM(
            eval_env.action_space().n,
            activation=config["ACTIVATION"],
            hidden_size=hidden_size,
        )
    else:
        network = [
            ActorCriticLSTM(
                eval_env.action_space().n,
                activation=config["ACTIVATION"],
                hidden_size=hidden_size,
            ) for _ in range(eval_env.num_agents)
        ]
 
    # --- General original / shaped reward tracing for eval ---
    # The env always returns info["original_rewards"] (pre-shaping, raw env
    # reward) and either info["shaped_rewards"] (when an im_type is set:
    # "ia", "svo", or "social_conformity") or info["group_shared_rewards"]
    # (when no im_type is set). For "ia"/"svo"/None, the env has ALREADY
    # applied the shaping, so info[shaped_key] is the final answer. Only
    # "social_conformity" is shaped OUTSIDE the env (at the algo level via
    # get_single_agent_reward), so we add that on top in that case only.
    im_cfg = config.get("ENV_KWARGS", {}).get("intrinsic_motivation", {})
    im_type = im_cfg.get("im_type")  # None, "ia", "svo", or "social_conformity"
    shaped_key = "shaped_rewards" if im_type is not None else "group_shared_rewards"
    greedy = config.get("EVAL_GREEDY", False)   # evaluation only; training rollouts stay stochastic
 
    track_conformity = (
        im_type == "social_conformity"
        and not config["PARAMETER_SHARING"]
    )
    
    if track_conformity:
        _conf_w = im_cfg.get("im_w", 1.0)
        all_params_stacked = jax.tree_util.tree_map(
            lambda *args: jnp.stack(args), *params
        )
 
    def run_single_episode(rng):
        """Run a single episode and return cumulative rewards per agent"""
        rng, _rng = jax.random.split(rng)
        obs, state = eval_env.reset(_rng)

        # Initialize LSTM hidden states (zeros at the start of every episode)
        if config["PARAMETER_SHARING"]:
            hidden = (
                jnp.zeros((eval_env.num_agents, hidden_size)),
                jnp.zeros((eval_env.num_agents, hidden_size)),
            )
        else:
            hidden = [
                (jnp.zeros((1, hidden_size)), jnp.zeros((1, hidden_size)))
                for _ in range(eval_env.num_agents)
            ]
 
        # Initialize episode rewards
        episode_rewards = jnp.zeros(eval_env.num_agents)
 
        # For sustainability: track timesteps where each agent got positive reward
        positive_reward_timesteps_sum = jnp.zeros(eval_env.num_agents)
        positive_reward_counts = jnp.zeros(eval_env.num_agents)
 
        total_freeze_steps = jnp.zeros(eval_env.num_agents)
 
        num_actions = eval_env.action_space().n # Get dynamically
 
        # Trackers
        action_counts = jnp.zeros((eval_env.num_agents, num_actions))
        apple_counts = jnp.zeros(eval_env.num_agents)
        event_counts = jnp.zeros((eval_env.num_agents, 5))
        # General reward-shaping tracers — always populated regardless of im_type
        original_rewards_sum = jnp.zeros(eval_env.num_agents)
        shaped_rewards_sum   = jnp.zeros(eval_env.num_agents)
 
        def episode_step(carry, t):
            """Single step in the episode"""
            (obs, state, episode_rewards, pos_reward_sum, pos_reward_counts,
             current_total_freeze_steps, steps_taken, current_actions, current_apples,
             current_events,
             current_original_rewards_sum, current_shaped_rewards_sum, rng, done, hidden) = carry
 
            # Select actions with hidden states
            # `done` is the done flag of the previous step -> reset flag for the LSTM
            obs_batch = jnp.stack([obs[a] for a in eval_env.agents])
            if config["PARAMETER_SHARING"]:
                reset = jnp.broadcast_to(jnp.asarray(done, dtype=bool), (eval_env.num_agents,))
                pi, _, new_hidden = network.apply(params, (obs_batch[None], reset[None]), hidden)
                rng, _rng = jax.random.split(rng)
                action = pi.mode() if greedy else pi.sample(seed=_rng)
                env_act = unbatchify(action.squeeze(0), eval_env.agents, 1, eval_env.num_agents)
                env_act = [v.squeeze() for v in env_act.values()]
            else:
                env_act = []
                new_hidden = []
                reset = jnp.broadcast_to(jnp.asarray(done, dtype=bool), (1,))
                for i in range(eval_env.num_agents):
                    # (H, W, C) -> (T=1, B=1, H, W, C)
                    obs_single = obs_batch[i][None, None]
                    pi, _, new_hidden_i = network[i].apply(
                        params[i], (obs_single, reset[None]), hidden[i]
                    )
                    rng, _rng = jax.random.split(rng)
                    action = pi.mode() if greedy else pi.sample(seed=_rng)
                    env_act.append(action.squeeze())
                    new_hidden.append(new_hidden_i)
 
            # Step environment
            rng, _rng = jax.random.split(rng)
            obs, state, reward, done_dict, info = eval_env.step(_rng, state, env_act)
 
            # Accumulate rewards (only if not done)
            done_flag = done_dict["__all__"]
            # group metrics (return, equality, sustainability) use the EXTERNAL reward;
            # under intrinsic motivation `reward` is the shaped signal
            env_reward = info["original_rewards"] if im_type is not None else reward
            episode_rewards = episode_rewards + env_reward * (1 - done_flag)
 
            # Track positive rewards for sustainability
            # For each agent, if reward > 0, accumulate the timestep
            has_positive_reward = (env_reward > 0).astype(jnp.float32)
            pos_reward_sum = pos_reward_sum + t * has_positive_reward * (1 - done_flag)
            pos_reward_counts = pos_reward_counts + has_positive_reward * (1 - done_flag)
 
            # info["agent_freeze"] is the counter for each agent
            freeze_counter = info["agent_freeze"].squeeze()
            is_frozen = (freeze_counter > 0).astype(jnp.float32)
 
            # Accumulate the total steps spent frozen (only if episode is not done)
            new_total_freeze_steps = current_total_freeze_steps + is_frozen * (1 - done_flag)
 
            action_updates = jax.nn.one_hot(info["actions_taken"], num_actions)
            new_actions = current_actions + action_updates * (1 - done_flag) * (1 - is_frozen[:, None]) # count actions only when the agent is alive
 
            new_apples = current_apples + info["apples_collected"] * (1 - done_flag)

            new_events = current_events + jnp.stack(
                [info["endangered_apples_collected"], info["endangered_opportunities"], info["zap_hits"],
                info["zap_opportunities"], info["zap_successes"]], axis=-1) * (1 - done_flag)
 
            steps_taken = steps_taken + (1 - done_flag)
 
            # --- Original / shaped reward tracing (general across im_type) ---
            # info["original_rewards"] and info[shaped_key] are always populated
            # by the env. For "ia"/"svo"/None they already reflect the final,
            # fully-shaped reward. For "social_conformity" the env leaves
            # rewards unshaped, so the algo-level penalty is added here.
            original_step = info["original_rewards"]
            shaped_step   = info[shaped_key]
 
            if track_conformity:
                # info["is_critical"] is set by the env from the pre-transition
                # obs (i.e. the obs agents used when choosing env_act this step).
                agent_indices = jnp.arange(eval_env.num_agents, dtype=jnp.int32)
                intrinsic_step = jax.vmap(
                    get_single_agent_reward,
                    in_axes=(0, 0, 0, None, None, None, None, 0)
                )(
                    obs_batch,
                    jnp.array(env_act, dtype=jnp.int32),
                    agent_indices,
                    all_params_stacked,
                    network[0],
                    eval_env.num_agents,
                    _conf_w,
                    info["is_critical"],
                )
                shaped_step = shaped_step + intrinsic_step
 
            new_original_rewards_sum = current_original_rewards_sum + original_step * (1 - done_flag)
            new_shaped_rewards_sum   = current_shaped_rewards_sum   + shaped_step   * (1 - done_flag)
 
            return (obs, state, episode_rewards, pos_reward_sum, pos_reward_counts,
                    new_total_freeze_steps, steps_taken, new_actions, new_apples,
                    new_events,
                    new_original_rewards_sum, new_shaped_rewards_sum, rng, done_flag, new_hidden), None
 
        # Run episode for max steps
        max_steps = config["ENV_KWARGS"].get("num_inner_steps", 1000)
        (_, _, final_rewards, final_pos_sum, final_pos_counts,
         final_freeze_steps, final_steps, final_actions, final_apples,
         final_events,
         final_original_rewards_sum, final_shaped_rewards_sum, _, _, _), _ = jax.lax.scan(
            episode_step,
            (obs, state, episode_rewards, positive_reward_timesteps_sum,
             positive_reward_counts, total_freeze_steps, 0.0, action_counts, apple_counts,
             event_counts,
             original_rewards_sum, shaped_rewards_sum, rng, jnp.array(False), hidden),
            jnp.arange(max_steps),
            length=max_steps
        )
 
        N = eval_env.num_agents
        T = jnp.maximum(final_steps, 1.0)
 
        # === SUSTAINABILITY METRIC ===
        # S = (1/N) * sum_i E[t | r_i^t > 0]
        # E[t | r_i^t > 0] = sum of timesteps with positive reward / count of positive rewards
        avg_positive_reward_timesteps = jnp.where(
            final_pos_counts > 0,
            final_pos_sum / final_pos_counts,
            0.0  # If agent never got positive reward, contribute 0
        )
        sustainability = jnp.mean(avg_positive_reward_timesteps)
 
        # === EQUALITY METRIC ===
        # E = 1 - sum_i sum_j |G_i - G_j| / (2N * sum_i G_i)
        # G_i = total reward for agent i (which is final_rewards[i])
        G = final_rewards  # Shape: (num_agents,)
 
        # Compute pairwise differences: |G_i - G_j| for all i,j
        # Broadcasting: G[:, None] - G[None, :] gives matrix of differences
        pairwise_diffs = jnp.abs(G[:, None] - G[None, :])  # Shape: (N, N)
        sum_abs_diffs = jnp.sum(pairwise_diffs)
 
        total_rewards = jnp.sum(G)
 
        # Avoid division by zero
        equality = jnp.where(
            total_rewards > 0,
            1.0 - (sum_abs_diffs / (2.0 * N * total_rewards)),
            1.0  # If no one got any reward, consider it perfectly equal
        )
 
        # === PEACE METRIC CALCULATION ===
        # P = N - (1/T) * sum_i (total_freeze_steps_i)
        sum_freeze_steps = jnp.sum(final_freeze_steps)
 
        # T is the effective number of steps taken (max(final_steps, 1.0))
        peace_metric = N - (sum_freeze_steps / T)
 
        # Normalize action counts by (Total Steps - Frozen Steps)
        alive_steps = max_steps - final_freeze_steps
        final_actions_norm = final_actions / (alive_steps[:, None] + 1e-8)
 
        return final_rewards, sustainability, equality, peace_metric, final_actions_norm, final_apples, final_events, avg_positive_reward_timesteps, final_freeze_steps, final_original_rewards_sum, final_shaped_rewards_sum
 
    # Generate random seeds for each episode
    rng = jax.random.PRNGKey(eval_seed)
    episode_rngs = jax.random.split(rng, num_episodes)
 
    # Run all episodes in parallel using vmap
    (all_episode_rewards, all_sustainability, all_equality, all_peace,
     all_actions, all_apples, all_events, all_positive_reward_timesteps, all_freeze_steps,
     all_original_rewards, all_shaped_rewards) = jax.vmap(run_single_episode)(episode_rngs)
 
    # 1. Summary dictionary for the ORIGINAL general JSON (No player_i_returns)
    summary_metrics = {
        "return": jnp.mean(all_episode_rewards, axis=1).tolist(),
        "sustainability": all_sustainability.tolist(),
        "equality": all_equality.tolist(),
        "peace": all_peace.tolist(),
    }
 
    # 2. Raw dictionary for the AGENT-SPECIFIC JSONs
    raw_data = {
        "returns": all_episode_rewards,                  # Shape: (episodes, agents)
        "actions": all_actions,                          # Shape: (episodes, agents, action_dim)
        "apples": all_apples,                            # Shape: (episodes, agents)
        "endangered_apples_collected": all_events[..., 0],
        "endangered_opportunities":    all_events[..., 1],
        "zap_hits":                    all_events[..., 2],
        "zap_opportunities":           all_events[..., 3],
        "zap_successes":               all_events[..., 4],
        "freeze_steps": all_freeze_steps,                # Shape: (episodes, agents)
        "pos_reward_timesteps": all_positive_reward_timesteps,  # Shape: (episodes, agents)
        "original_rewards": all_original_rewards,        # Shape: (episodes, agents)
        "shaped_rewards": all_shaped_rewards,              # Shape: (episodes, agents)
    }
 
    return {"summary": summary_metrics, "raw": raw_data}


def _eval_worker_main(params_path, config_path, result_path):
    """
    Entry point of the evaluation subprocess (python <this file> --eval-worker params config result).
    Runs run_evaluation_episodes in its own process / CUDA context and writes
    {"summary": ..., "raw": ...} as JSON for eval_callback to read.
    """
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, params)
    with open(config_path, 'r') as f:
        config = json.load(f)

    results = run_evaluation_episodes(
        params, config, config["_eval_seed"], num_episodes=config["EVAL_EPISODES"]
    )

    def to_lists(d):
        return {k: np.asarray(v).tolist() for k, v in d.items()}

    with open(result_path, 'w') as f:
        json.dump({"summary": to_lists(results["summary"]), "raw": to_lists(results["raw"])}, f)

    if config.get("_make_video", False):
        print(f"\nGenerating FINAL video for step {config['_step']}...")
        video_env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        evaluate(params, video_env, config["_video_dir"], config, save_video=True, step=config["_step"])


def evaluate(params, env, save_path, config, save_video=True, step=0):
    """
    Evaluate policy and optionally save GIF.

    Returns:
        dict: {"player_0": return, ...}
              Returns empty dict if save_video=True (called at end of training)
    """
    rng = jax.random.PRNGKey(0)

    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False

    greedy = config.get("EVAL_GREEDY", False)

    # Initialize return tracking
    episode_rewards = {f"player_{i}": 0.0 for i in range(env.num_agents)}

    # Initialize hidden states
    hidden_size = config.get("LSTM_HIDDEN_SIZE", 128)
    if config["PARAMETER_SHARING"]:
        hidden = (
            jnp.zeros((env.num_agents, hidden_size)),
            jnp.zeros((env.num_agents, hidden_size)),
        )
        # jitted: this loop runs un-jitted in Python for GIF_NUM_FRAMES steps and an eager
        # (nn.scan) apply costs ~0.25 s per call vs <1 ms compiled
        network_apply = jax.jit(ActorCriticLSTM(
            action_dim=env.action_space().n,
            activation=config["ACTIVATION"],
            hidden_size=hidden_size,
        ).apply)
    else:
        hidden = [
            (jnp.zeros((1, hidden_size)), jnp.zeros((1, hidden_size)))
            for _ in range(env.num_agents)
        ]
        network_apply = jax.jit(ActorCriticLSTM(
            action_dim=env.action_space().n,
            activation=config["ACTIVATION"],
            hidden_size=hidden_size,
        ).apply)  # one compiled fn shared by all agents (same architecture, different params)

    pics = []
    if save_video:
        img = env.render(state)
        pics.append(img)
        root_dir = save_path

    for o_t in range(config["GIF_NUM_FRAMES"]):
        # Get actions using the policy
        # `done` (previous step) is the LSTM reset flag
        if config["PARAMETER_SHARING"]:
            obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)
            reset = jnp.full((env.num_agents,), bool(done))
            pi, _, hidden = network_apply(params, (obs_batch[None], reset[None]), hidden)
            rng, _rng = jax.random.split(rng)
            actions = pi.mode() if greedy else pi.sample(seed=_rng)
            env_act = {k: v.squeeze() for k, v in unbatchify(
                actions.squeeze(0), env.agents, 1, env.num_agents
            ).items()}
        else:
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            env_act = {}
            reset = jnp.full((1,), bool(done))
            new_hidden = []
            for i in range(env.num_agents):
                obs_single = obs_batch[i][None, None]
                pi, _, new_hidden_i = network_apply(params[i], (obs_single, reset[None]), hidden[i])
                rng, _rng = jax.random.split(rng)
                single_action = pi.mode() if greedy else pi.sample(seed=_rng)
                env_act[env.agents[i]] = single_action.squeeze()
                new_hidden.append(new_hidden_i)
            hidden = new_hidden

        # Step environment
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])

        # Accumulate rewards
        for i in range(env.num_agents):
            episode_rewards[f"player_{i}"] += float(reward[i])

        done = done["__all__"]

        # Render if saving GIF
        if save_video:
            img = env.render(state)
            pics.append(img)

    # Save GIF if requested
    if save_video:
        print(f"Saving Episode MP4")
        n_agents = len(env.agents)
        mp4_path = f"{root_dir}/{n_agents}-agents_seed-{config['SEED']}_step_{step}.mp4"

        # Convert to numpy arrays
        frames = [np.array(img) for img in pics]

        # Save as MP4
        imageio.mimsave(
            mp4_path,
            frames,
            fps=8,
            codec='libx264',
            quality=8,
            pixelformat='yuv420p'
        )

        if wandb.run is not None:  # the eval worker process has no wandb run; the parent uploads it
            print("Logging MP4 to WandB")
            wandb.log({"Episode Video": wandb.Video(mp4_path, caption="Evaluation Episode", format="mp4")})
        return {}
    else:
        return episode_rewards


def tune(default_config):
    """
    Hyperparameter sweep with wandb, including logic to:
    - Initialize wandb
    - Train for each hyperparameter set
    - Save checkpoint
    - Evaluate and log GIF
    """
    import copy

    default_config = OmegaConf.to_container(default_config)

    sweep_config = {
        "name": "harvest_timeout",
        "method": "grid",
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            # "LR": {"values": [0.001, 0.0005, 0.0001, 0.00005]},
            # "ACTIVATION": {"values": ["relu", "tanh"]},
            # "UPDATE_EPOCHS": {"values": [2, 4, 8]},
            # "NUM_MINIBATCHES": {"values": [4, 8, 16, 32]},
            # "CLIP_EPS": {"values": [0.1, 0.2, 0.3]},
            # "ENT_COEF": {"values": [0.001, 0.01, 0.1]},
            # "NUM_STEPS": {"values": [64, 128, 256]},
            # "ENV_KWARGS.svo_w": {"values": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
            # "ENV_KWARGS.svo_ideal_angle_degrees": {"values": [0, 45, 90]},
            "SEED": {"values": [42, 52, 62]},

        },
    }

    def wrapped_make_train():


        wandb.init(project=default_config["PROJECT"])
        config = copy.deepcopy(default_config)
        # only overwrite the single nested key we're sweeping
        for k, v in dict(wandb.config).items():
            keys = k.split(".")
            d = config
            for key in keys[:-1]:
                d = d[key]
            d[keys[-1]] = v


        # Rename the run for clarity
        run_name = f"sweep_{config['ENV_NAME']}_seed{config['SEED']}"
        wandb.run.name = run_name
        print("Running experiment:", run_name)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))
        train_state = jax.tree_util.tree_map(lambda x: x[0], outs["runner_state"][0])

        # Evaluate and log
        # params = load_params(train_state.params)
        # test_env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        # evaluate(params, test_env, config)

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="config", config_name="ippo_cnn_lstm_harvest_timeout")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)
if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--eval-worker":
        # evaluation subprocess launched by eval_callback (bypasses hydra)
        _eval_worker_main(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()