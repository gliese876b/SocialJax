"""
Based on PureJaxRL & jaxmarl Implementation of PPO
"""
import sys
sys.path.append('/home/shuqing/SocialJax')
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
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
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


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(x)

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def get_rollout(params, config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    if config["PARAMETER_SHARING"]:
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
    else:
        network = [ActorCritic(env.action_space().n, activation=config["ACTIVATION"]) for _ in range(env.num_agents)]
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    for o in range(config["GIF_NUM_FRAMES"]):
        print(o)
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)
        if config["PARAMETER_SHARING"]:
            pi, value = network.apply(params, obs_batch)
            action = pi.sample(seed=key_a0)
            env_act = unbatchify(
                action, env.agents, 1, env.num_agents
            )
        else:
            env_act = {}
            for i in range(env.num_agents):
                pi, value = network[i].apply(params[i], obs_batch)
                action = pi.sample(seed=key_a0)
                env_act[env.agents[i]] = action




        env_act = {k: v.squeeze() for k, v in env_act.items()}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, env_act)
        done = done["__all__"]

        state_seq.append(state)

    return state_seq


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_dict(x: dict, agent_list, num_actors):
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    if config["PARAMETER_SHARING"]:
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    else:
        config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    config["EVAL_INTERVAL"] = config.get("EVAL_INTERVAL", 50000)
    config["EVAL_EPISODES"] = config.get("EVAL_EPISODES", 10)

    env = LogWrapper(env, replace_info=False)

    rew_shaping_anneal = optax.linear_schedule(
        init_value=0.,
        end_value=1.,
        transition_steps=config["REW_SHAPING_HORIZON"],
        transition_begin=config["SHAPING_BEGIN"]
    )

    rew_shaping_anneal_org = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config["REW_SHAPING_HORIZON"],
        transition_begin=config["SHAPING_BEGIN"]
    )
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        if config["PARAMETER_SHARING"]:
            network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        else:
            network = [ActorCritic(env.action_space().n, activation=config["ACTIVATION"]) for _ in range(env.num_agents)]

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))

        if config["PARAMETER_SHARING"]:
            network_params = network.init(_rng, init_x)
        else:
            network_params = [network[i].init(_rng, init_x) for i in range(env.num_agents)]
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

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                # obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(-1, *env.observation_space().shape)

                if config["PARAMETER_SHARING"]:
                    obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                    print("input_obs_shape", obs_batch.shape)
                    pi, value = network.apply(train_state.params, obs_batch)
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)
                    env_act = unbatchify(
                        action, env.agents, config["NUM_ENVS"], env.num_agents
                    )
                else:
                    obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                    env_act = {}
                    log_prob = []
                    value = []
                    for i in range(env.num_agents):
                        print("input_obs_shape", obs_batch[i].shape)
                        pi, value_i = network[i].apply(train_state[i].params, obs_batch[i])
                        action = pi.sample(seed=_rng)
                        log_prob.append(pi.log_prob(action))
                        env_act[env.agents[i]] = action
                        value.append(value_i)



                # env_act = {k: v.flatten() for k, v in env_act.items()}
                env_act = [v for v in env_act.values()]

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                # current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
                # shaped_reward = compute_grouped_rewards(reward)
                # reward = jax.tree_util.tree_map(lambda x,y: x*rew_shaping_anneal_org(current_timestep)+y*rew_shaping_anneal(current_timestep), reward, shaped_reward)

                # Define a function to conditionally reshape based on dimensionality
                def reshape_if_scalar(x):
                    # Check if the array is multi-dimensional (like agent_locs, which is (N, 3))
                    # Multi-dimensional arrays are returned as-is.
                    if x.ndim > 1:
                        return x
                    # If it's a 1D vector (like freeze status, apple count), reshape it to (N, 1)
                    # This is safe and mathematically valid.
                    return x.reshape((config["NUM_ACTORS"]), 1)

                if config["PARAMETER_SHARING"]:
                    info = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                    transition = Transition(
                        batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                        action,
                        value,
                        batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                        log_prob,
                        obs_batch,
                        info,
                        )
                else:
                    transition = []
                    done = [v for v in done.values()]
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
                        ))
                runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng = runner_state
            if config["PARAMETER_SHARING"]:
                last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                _, last_val = network.apply(train_state.params, last_obs_batch)
            else:
                last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                last_val = []
                for i in range(env.num_agents):
                    _, last_val_i = network[i].apply(train_state[i].params, last_obs_batch[i])
                    last_val.append(last_val_i)
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
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets, network_used):
                        # RERUN NETWORK
                        pi, value = network_used.apply(params, traj_batch.obs)
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
                            train_state.params, traj_batch, advantages, targets, network_used
                        )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                    )
                # if config["PARAMETER_SHARING"]:

                # else:
                #     batch = jax.tree_util.tree_map(
                #         lambda x: x.reshape((batch_size,) + x.shape[2:]),  # 保持第一个维度为batch_size，自动计算第二个维度
                #         batch
                #     )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                if config["PARAMETER_SHARING"]:
                    train_state, total_loss = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network), train_state, minibatches
                    )
                else:
                    train_state, total_loss = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network[i]), train_state, minibatches
                    )

                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            if config["PARAMETER_SHARING"]:
                update_state = (train_state, traj_batch, advantages, targets, rng)
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
                    update_state = (train_state[i], traj_batch[i], advantages[i], targets[i], rng)
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

                jax.clear_caches()

                """Run evaluation and append results to JSON file"""
                print(f"\n{'='*60}")
                print(f"Running evaluation at step {step}...")
                print(f"{'='*60}")

                # Use different seed for evaluation
                eval_seed = config["SEED"] + int(step) + 999999

                # Set up video directory
                video_dir = os.path.join(config["RUN_OUTPUT_DIR"], "eval_videos")

                eval_data = run_evaluation_episodes(
                    params,
                    config,
                    eval_seed,
                    num_episodes=config["EVAL_EPISODES"],
                )

                # Print summary
                print(f"\nEvaluation Results (step {step}):")
                print(f"  Episodes: {config['EVAL_EPISODES']}")
                for key, values in eval_data.items():
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f"  {key}: {mean_val:.2f} ± {std_val:.2f}")
                print(f"{'='*60}\n")

                # Log to wandb
                for key, values in eval_data.items():
                    wandb.log({
                        f"eval/{key}_mean": float(np.mean(values)),
                        f"eval/{key}_std": float(np.std(values)),
                        "eval/step": int(step)
                    })

                # UPDATE JSON FILE INCREMENTALLY
                output_dir = f"{config['RUN_OUTPUT_DIR'] }/evaluation"
                os.makedirs(output_dir, exist_ok=True)
                
                json_filename = f"{config['ENV_NAME']}_ippo_seed_{config['SEED']}.json"
                json_path = os.path.join(output_dir, json_filename)
                
                # Read existing data or create new structure
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        marl_eval_data = json.load(f)
                else:
                    # Initialize MARL-eval structure
                    env_name = config["ENV_NAME"]
                    algo_name = "ippo"
                    seed_name = f"seed_{config['SEED']}"
                    
                    marl_eval_data = {
                        "socialjax": {
                            f"{env_name}": {
                                algo_name: {
                                    seed_name: {}
                                }
                            }
                        }
                    }
                
                # Navigate to the seed results dict
                env_name = config["ENV_NAME"]
                algo_name = "ippo"
                seed_name = f"seed_{config['SEED']}"
                seed_results = marl_eval_data["socialjax"][f"{env_name}"][algo_name][seed_name]
                
                # Count existing steps to get the next step index
                existing_steps = [k for k in seed_results.keys() if k.startswith("step_")]
                step_idx = len(existing_steps)
                
                # Add this evaluation checkpoint
                seed_results[f"step_{step_idx}"] = {
                    "step_count": int(step),
                    **eval_data
                }
                
                # Update absolute metrics (always use the latest evaluation as final)
                seed_results["absolute_metrics"] = {
                    key: [float(np.mean(values))]
                    for key, values in eval_data.items()
                }
                
                # Write back to file
                with open(json_path, 'w') as f:
                    json.dump(marl_eval_data, f, indent=2)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
                print(f"[{timestamp}] ✓ Updated {json_path} with step {step_idx} (step_count={int(step)})")


                # gif_env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
                
                # print("\n" + "="*60)
                # print(f"Generating GIF for step {step}...")
                # print("="*60)
                # evaluate(
                #     params,
                #     gif_env,
                #     output_dir,
                #     config,
                #     save_gif=True, 
                #     step=step
                # )
                # print(f"✓ GIF generated and logged to WandB for step {step}")  

                jax.clear_caches()

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

            should_eval = (current_step > 0) & (current_step % config["EVAL_INTERVAL"] == 0)
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

            runner_state = (train_state, env_state, last_obs, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
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
    algo_name = "ippo"  # or config.get("ALGO_NAME", "ippo")
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

    run = wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'ippo_cnn_harvest_common_seed{config["SEED"]}'
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    config["RUN_OUTPUT_DIR"] = run.dir
    print(f"\n[{timestamp}] Started training...")
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

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

    evaluate(params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), f"{config['RUN_OUTPUT_DIR']}/evaluation", config)

    # MARL-eval JSON was already created and updated during training!
    json_filename = f"{config['ENV_NAME']}_ippo_seed{config['SEED']}.json"
    json_path = os.path.join(f"{config['RUN_OUTPUT_DIR']}/evaluation", json_filename)
    
    if os.path.exists(json_path):
        print(f"\n✓ MARL-eval results saved at: {json_path}")
        
        # Load and print summary
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        env_name = config["ENV_NAME"]
        seed_name = f"seed_{config['SEED']}"
        seed_results = data[env_name][f"{env_name}"]["ippo"][seed_name]
        
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
        dict: {"player_0_return": [...], "player_1_return": [...], "return": [...]}
    """
    # Create eval environment
    eval_env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if config["PARAMETER_SHARING"]:
        network = ActorCritic(eval_env.action_space().n, activation=config["ACTIVATION"])
    else:
        network = [ActorCritic(eval_env.action_space().n, activation=config["ACTIVATION"])
                   for _ in range(eval_env.num_agents)]

    def run_single_episode(rng):
        """Run a single episode and return cumulative rewards per agent"""
        rng, _rng = jax.random.split(rng)
        obs, state = eval_env.reset(_rng)

        # Initialize episode rewards
        episode_rewards = jnp.zeros(eval_env.num_agents)

        # For sustainability: track timesteps where each agent got positive reward
        positive_reward_timesteps_sum = jnp.zeros(eval_env.num_agents)
        positive_reward_counts = jnp.zeros(eval_env.num_agents)

        total_freeze_steps = jnp.zeros(eval_env.num_agents)

        def episode_step(carry, t):
            """Single step in the episode"""
            (obs, state, episode_rewards, pos_reward_sum, pos_reward_counts, current_total_freeze_steps, steps_taken, rng, done) = carry

            # Select actions
            if config["PARAMETER_SHARING"]:
                # obs is dict with shape per agent: (H, W, C)
                # Stack to: (num_agents, H, W, C)
                obs_batch = jnp.stack([obs[a] for a in eval_env.agents])
                # Reshape for network: (num_agents, H, W, C) -> already correct shape
                pi, _ = network.apply(params, obs_batch)
                rng, _rng = jax.random.split(rng)
                action = pi.sample(seed=_rng)
                env_act = unbatchify(action, eval_env.agents, 1, eval_env.num_agents)
                env_act = [v.squeeze() for v in env_act.values()]
            else:
                # obs is dict, stack to (num_agents, H, W, C)
                obs_batch = jnp.stack([obs[a] for a in eval_env.agents])
                env_act = []
                for i in range(eval_env.num_agents):
                    # Add batch dimension: (H, W, C) -> (1, H, W, C)
                    obs_single = jnp.expand_dims(obs_batch[i], axis=0)
                    pi, _ = network[i].apply(params[i], obs_single)
                    rng, _rng = jax.random.split(rng)
                    action = pi.sample(seed=_rng)
                    env_act.append(action.squeeze())

            # Step environment
            rng, _rng = jax.random.split(rng)
            obs, state, reward, done_dict, info = eval_env.step(_rng, state, env_act)

            # Accumulate rewards (only if not done)
            done_flag = done_dict["__all__"]
            episode_rewards = episode_rewards + reward * (1 - done_flag)

            # Track positive rewards for sustainability
            # For each agent, if reward > 0, accumulate the timestep
            has_positive_reward = (reward > 0).astype(jnp.float32)
            pos_reward_sum = pos_reward_sum + t * has_positive_reward * (1 - done_flag)
            pos_reward_counts = pos_reward_counts + has_positive_reward * (1 - done_flag)

            # info["agent_freeze"] is the counter for each agent
            freeze_counter = info["agent_freeze"].squeeze() 
            is_frozen = (freeze_counter > 0).astype(jnp.float32)
            
            # Accumulate the total steps spent frozen (only if episode is not done)
            new_total_freeze_steps = current_total_freeze_steps + is_frozen * (1 - done_flag)            
            
            steps_taken = steps_taken + (1 - done_flag)

            return (obs, state, episode_rewards, pos_reward_sum, pos_reward_counts, new_total_freeze_steps, steps_taken, rng, done_flag), None

        # Run episode for max steps
        max_steps = config["ENV_KWARGS"].get("num_inner_steps", 1000)
        (_, _, final_rewards, final_pos_sum, final_pos_counts, 
         final_freeze_steps, final_steps, _, _), _ = jax.lax.scan(
            episode_step,
            (obs, state, episode_rewards, positive_reward_timesteps_sum, 
             positive_reward_counts, total_freeze_steps, 0.0, rng, False),
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

        return final_rewards, sustainability, equality, peace_metric

    # Generate random seeds for each episode
    rng = jax.random.PRNGKey(eval_seed)
    episode_rngs = jax.random.split(rng, num_episodes)

    # Run all episodes in parallel using vmap
    all_episode_rewards, all_sustainability, all_equality, all_peace = jax.vmap(run_single_episode)(episode_rngs)

    # Convert to MARL-eval format
    all_returns = {}
    for i in range(eval_env.num_agents):
        all_returns[f"player_{i}_return"] = all_episode_rewards[:, i].tolist()

    # Calculate mean return across agents for each episode
    mean_returns = jnp.mean(all_episode_rewards, axis=1)
    all_returns["return"] = mean_returns.tolist()

    # Add all metrics
    all_returns["sustainability"] = all_sustainability.tolist()
    all_returns["equality"] = all_equality.tolist()
    all_returns["peace"] = all_peace.tolist()

    return all_returns


def evaluate(params, env, save_path, config, save_gif=True, step=0):
    """
    Evaluate policy and optionally save GIF.

    Returns:
        dict: {"player_0_return": [...], "player_1_return": [...], "return": [...]}
              Returns empty dict if save_gif=True (called at end of training)
    """
    rng = jax.random.PRNGKey(0)

    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False

    # Initialize return tracking
    episode_rewards = {f"player_{i}": 0.0 for i in range(env.num_agents)}

    pics = []
    if save_gif:
        img = env.render(state)
        pics.append(img)
        root_dir = save_path

    for o_t in range(config["GIF_NUM_FRAMES"]):
        # Get actions using the policy
        if config["PARAMETER_SHARING"]:
            obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)
            network = ActorCritic(action_dim=env.action_space().n, activation="relu")
            pi, _ = network.apply(params, obs_batch)
            rng, _rng = jax.random.split(rng)
            actions = pi.sample(seed=_rng)
            env_act = {k: v.squeeze() for k, v in unbatchify(
                actions, env.agents, 1, env.num_agents
            ).items()}
        else:
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            env_act = {}
            network = [ActorCritic(action_dim=env.action_space().n, activation="relu") for _ in range(env.num_agents)]
            for i in range(env.num_agents):
                obs = jnp.expand_dims(obs_batch[i], axis=0)
                pi, _ = network[i].apply(params[i], obs)
                rng, _rng = jax.random.split(rng)
                single_action = pi.sample(seed=_rng)
                env_act[env.agents[i]] = single_action

        # Step environment
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])

        # Accumulate rewards
        for i in range(env.num_agents):
            episode_rewards[f"player_{i}"] += float(reward[i])

        done = done["__all__"]

        # Render if saving GIF
        if save_gif:
            img = env.render(state)
            pics.append(img)

    # Save GIF if requested
    if save_gif:
        print(f"Saving Episode GIF")
        pics = [Image.fromarray(np.array(img)) for img in pics]
        n_agents = len(env.agents)
        gif_path = f"{root_dir}/{n_agents}-agents_seed-{config['SEED']}_step_{step}.gif"
        pics[0].save(
            gif_path,
            format="GIF",
            save_all=True,
            optimize=False,
            append_images=pics[1:],
            duration=200,
            loop=0,
        )
        print("Logging GIF to WandB")
        wandb.log({"Episode GIF": wandb.Video(gif_path, caption="Evaluation Episode", format="gif")})

        return {}  # Don't return data when saving GIF
    else:
        # Return episode returns for MARL-eval
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
            if "." in k:
                parent, child = k.split(".", 1)
                config[parent][child] = v
            else:
                config[k] = v


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


@hydra.main(version_base=None, config_path="config", config_name="ippo_cnn_harvest_timeout")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)
if __name__ == "__main__":
    main()
