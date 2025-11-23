from enum import IntEnum
import math
from typing import Any, Optional, Tuple, Union, Dict
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as onp
from flax.struct import dataclass
import colorsys

from socialjax.environments.multi_agent_env import MultiAgentEnv
from socialjax.environments import spaces


from socialjax.environments.common_harvest.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)

NUM_TYPES = 4  # empty (0), red (1), blue, red coin, blue coin, wall, interact
INTERACT_THRESHOLD = 0


@dataclass
class State:
    agent_locs: jnp.ndarray
    agent_invs: jnp.ndarray
    inner_t: int
    outer_t: int
    grid: jnp.ndarray
    apples: jnp.ndarray

    freeze: jnp.ndarray
    reborn_locs: jnp.ndarray

@chex.dataclass
class EnvParams:
    payoff_matrix: chex.ArrayDevice
    freeze_penalty: int

class Actions(IntEnum):
    turn_left = 0
    turn_right = 1
    move_forward = 2
    stay = 3
    zap_forward = 4

class Items(IntEnum):
    empty = 0
    wall = 1
    interact = 2
    apple = 3
    spawn_point = 4
    
char_to_int = {
    'W': 1,
    ' ': 0,  # 空格字符映射为 0
    'A': 3,
    'P': 4
}

ROTATIONS = jnp.array(
    [
        [0, 0, 1],   # Action 0: Turn Left (+1 dir)
        [0, 0, -1],  # Action 1: Turn Right (-1 dir)
        [0, 0, 0],   # Action 2: Move Forward (No rotation)
        [0, 0, 0],   # Action 3: Stay (No rotation)
        [0, 0, 0],   # Action 4: Zap (No rotation)
    ],
    dtype=jnp.int8,
)
STEP = jnp.array(
    [
        [1, 0, 0],  # up
        [0, 1, 0],  # right
        [-1, 0, 0],  # down
        [0, -1, 0],  # left
    ],
    dtype=jnp.int8,
)

def ascii_map_to_matrix(map_ASCII, char_to_int):
    """
    Convert ASCII map to a JAX numpy matrix using the given character mapping.
    
    Args:
    map_ASCII (list): List of strings representing the ASCII map
    char_to_int (dict): Dictionary mapping characters to integer values
    
    Returns:
    jax.numpy.ndarray: 2D matrix representation of the ASCII map
    """
    # Determine matrix dimensions
    height = len(map_ASCII)
    width = max(len(row) for row in map_ASCII)
    
    # Create matrix filled with zeros
    matrix = jnp.zeros((height, width), dtype=jnp.int32)
    
    # Fill matrix with mapped values
    for i, row in enumerate(map_ASCII):
        for j, char in enumerate(row):
            matrix = matrix.at[i, j].set(char_to_int.get(char, 0))
    
    return matrix

def generate_agent_colors(num_agents):
    colors = []
    for i in range(num_agents):
        hue = i / num_agents
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)  # Saturation and Value set to 0.8
        colors.append(tuple(int(x * 255) for x in rgb))
    return colors

###################################################

class Harvest_timeout(MultiAgentEnv):
    """
    JAX Compatible n-agent version of *inTheMatix environment.
    """

    # used for caching
    tile_cache: Dict[Tuple[Any, ...], Any] = {}

    def __init__(
        self,
        num_inner_steps=1000,
        num_outer_steps=1,
        num_agents=10,
        shared_rewards=False,
        zap_beam_width=1,      # cells on each side (1 means 3 total width)
        zap_beam_length=5,     # cells forward (2 means checks 1 and 2 ahead)
        inequity_aversion=False,
        inequity_aversion_target_agents=None,
        inequity_aversion_alpha=5,
        inequity_aversion_beta=0.05,
        enable_smooth_rewards=False,
        svo=False,
        svo_target_agents=None,
        svo_w=0.5,
        svo_ideal_angle_degrees=45,
        grid_size=(11, 27),
        jit=True,
        obs_size=11,
        cnn=True,
        map_ASCII = [
                "P    A  AAA  A  AAA  A    P",
                "P   AAA  A  AAA  A  AAA   P",
                "P  AAAAA   AAAAA   AAAAA  P",
                "P   AAA  A  AAA  A  AAA   P",
                "P    A  AAA  A  AAA  A    P",
                "P      AAAAA   AAAAA      P",
                "P    A  AAA  A  AAA  A    P",
                "P   AAA  A  AAA  A  AAA   P",
                "P  AAAAA   AAAAA   AAAAA  P",
                "P   AAA  A  AAA  A  AAA   P",
                "P    A  AAA  A  AAA  A    P",
            ]


    ):

        super().__init__(num_agents=num_agents)
        self.agents = list(range(num_agents))#, dtype=jnp.int16)
        self._agents = jnp.array(self.agents, dtype=jnp.int16) + len(Items)
        self.shared_rewards = shared_rewards
        self.zap_beam_width = zap_beam_width
        self.zap_beam_length = zap_beam_length
        self.cnn = cnn
        self.inequity_aversion = inequity_aversion
        self.inequity_aversion_target_agents = inequity_aversion_target_agents
        self.inequity_aversion_alpha = inequity_aversion_alpha
        self.inequity_aversion_beta = inequity_aversion_beta
        self.enable_smooth_rewards = enable_smooth_rewards
        self.svo = svo
        self.svo_target_agents = svo_target_agents
        self.svo_w = svo_w
        self.svo_ideal_angle_degrees = svo_ideal_angle_degrees
        self.smooth_rewards = enable_smooth_rewards
        self.PLAYER_COLOURS = generate_agent_colors(num_agents)
        self.GRID_SIZE_ROW = grid_size[0]
        self.GRID_SIZE_COL = grid_size[1]
        self.OBS_SIZE = obs_size
        self.PADDING = self.OBS_SIZE - 1
        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps

        GRID = jnp.zeros(
            (self.GRID_SIZE_ROW + 2 * self.PADDING, self.GRID_SIZE_COL + 2 * self.PADDING),
            dtype=jnp.int16,
        )

        # First layer of padding is Wall
        GRID = GRID.at[self.PADDING - 1, :].set(5)
        GRID = GRID.at[self.GRID_SIZE_ROW + self.PADDING, :].set(5)
        GRID = GRID.at[:, self.PADDING - 1].set(5)
        self.GRID = GRID.at[:, self.GRID_SIZE_COL + self.PADDING].set(5)

        def find_positions(grid_array, letter):
            a_positions = jnp.array(jnp.where(grid_array == letter)).T
            return a_positions

        nums_map = ascii_map_to_matrix(map_ASCII, char_to_int)
        self.SPAWNS_APPLE = find_positions(nums_map, 3)
        self.SPAWNS_PLAYERS = find_positions(nums_map, 4)
        self.SPAWNS_WALL = find_positions(nums_map, 1)


        def rand_interaction(
                key: int,
                conflicts: jnp.ndarray,
                conflicts_matrix: jnp.ndarray,
                step_arr: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function for randomly choosing between conflicting interactions.
            
            Args:
                - key: jax PRNGKey for randomisation.
                - conflicts: jnp.ndarray of bools where True if agent is in a
                conflicting interaction, False otherwise.
                - conflicts_matrix: jnp.ndarray matrix of bools of agents in
                conflicting interactions.
                - step_arr: jnp.ndarray, where each index is the index of an
                agent, and the element at each index is the item found at that
                agent's respective target location in the grid.

                
            Returns:
                - jnp.ndarray array of final interactions, where each index is
                an agent, and each element is caught in its interaction beam.
            '''
            def scan_fn(
                    state,
                    idx
            ):

                key, conflicts, conflicts_matrix, step_arr = state

                return jax.lax.cond(
                    conflicts[idx] > 0,
                    lambda: _rand_interaction(
                        key,
                        conflicts,
                        conflicts_matrix,
                        step_arr
                    ),
                    lambda: (state, step_arr.astype(jnp.int16))
                )

            _, ys = jax.lax.scan(
                scan_fn,
                (key, conflicts, conflicts_matrix, step_arr.astype(jnp.int16)),
                jnp.arange(self.num_agents)
            )

            final_itxs = ys[-1]
            return final_itxs

        def _rand_interaction(
                key: int,
                conflicts: jnp.ndarray,
                conflicts_matrix: jnp.ndarray,
                step_arr: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function for randomly choosing between conflicting interactions.
            
            Args:
                - key: jax PRNGKey for randomisation.
                - conflicts: jnp.ndarray of bools where True if agent is in a
                conflicting interaction, False otherwise.
                - conflicts_matrix: jnp.ndarray matrix of bools of agents in
                conflicting interactions.
                - step_arr: jnp.ndarray, where each index is the index of an
                agent, and the element at each index is the item found at that
                agent's respective target location in the grid.

                
            Returns:
                - jnp.ndarray array of final interactions, where each index is
                an agent, and each element is caught in its interaction beam.
            '''
            conflict_idx = jnp.nonzero(
                conflicts,
                size=self.num_agents,
                fill_value=-1
            )[0][0]

            agent_conflicts = conflicts_matrix[conflict_idx]

            agent_conflicts_idx = jnp.nonzero(
                agent_conflicts,
                size=self.num_agents,
                fill_value=-1
            )[0]
            max_rand = jnp.sum(agent_conflicts_idx > -1)

            # preparing random agent selection
            k1, k2 = jax.random.split(key, 2)
            random_number = jax.random.randint(
                k1,
                (1,),
                0,
                max_rand
            )

            # index in main matrix of agent of successful interaction
            rand_agent_idx = agent_conflicts_idx[random_number]

            # set that agent's bool to False, for inversion later
            new_agent_conflict = agent_conflicts.at[rand_agent_idx].set(False)

            # set all remaining True agents' values as the "empty" item
            step_arr = jnp.where(
                new_agent_conflict,
                Items.empty,
                step_arr
            ).astype(jnp.int16)

            # update conflict bools to reflect the post-conflict state
            _conflicts = conflicts.at[agent_conflicts_idx].set(0)
            conflicts = jnp.where(
                agent_conflicts,
                0,
                conflicts
            )
            conflicts = conflicts.at[conflict_idx].set(0)
            conflicts_matrix = jax.vmap(
                lambda c, x: jnp.where(c, x, jnp.array([False]*conflicts.shape[0]))
            )(conflicts, conflicts_matrix)

            # deal with next conflict
            return ((
                k2,
                conflicts,
                conflicts_matrix,
                step_arr
            ), step_arr)
        
        # first attempt at func - needs improvement
        # inefficient due to double-checking collisions
        def check_collision(
                new_agent_locs: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function to check agent collisions.
            
            Args:
                - new_agent_locs: jnp.ndarray, the agent locations at the 
                current time step.
                
            Returns:
                - jnp.ndarray matrix of bool of agents in collision.
            '''
            matcher = jax.vmap(
                lambda x,y: jnp.all(x[:2] == y[:2]),
                in_axes=(0, None)
            )

            collisions = jax.vmap(
                matcher,
                in_axes=(None, 0)
            )(new_agent_locs, new_agent_locs)

            return collisions
        
        # first attempt at func - needs improvement
        # inefficient due to double-checking collisions
        def check_interaction_conflict(
                items: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function to check conflicting interaction targets.
            
            Args:
                - items: jnp.ndarray, the agent itemss at the interaction
                targets.
                
            Returns:
                - jnp.ndarray matrix of bool of agents in collision.
            '''
            matcher = jax.vmap(
                lambda x,y: jnp.logical_and(
                    jnp.all(x == y),
                    jnp.isin(x, self._agents)
                ),
                in_axes=(0, None)
            )

            collisions = jax.vmap(
                matcher,
                in_axes=(None, 0)
            )(items, items)

            return collisions

        def fix_collisions(
            key: jnp.ndarray,
            collided_moved: jnp.ndarray,
            collision_matrix: jnp.ndarray,
            agent_locs: jnp.ndarray,
            new_agent_locs: jnp.ndarray
        ) -> jnp.ndarray:
            """
            Function defining multi-collision logic.

            Args:
                - key: jax key for randomisation
                - collided_moved: jnp.ndarray, the agents which moved in the
                last time step and caused collisions.
                - collision_matrix: jnp.ndarray, the agents currently in
                collisions
                - agent_locs: jnp.ndarray, the agent locations at the previous
                time step.
                - new_agent_locs: jnp.ndarray, the agent locations at the
                current time step.

            Returns:
                - jnp.ndarray of the final positions after collisions are
                managed.
            """
            def scan_fn(
                    state,
                    idx
            ):
                key, collided_moved, collision_matrix, agent_locs, new_agent_locs = state

                return jax.lax.cond(
                    collided_moved[idx] > 0,
                    lambda: _fix_collisions(
                        key,
                        collided_moved,
                        collision_matrix,
                        agent_locs,
                        new_agent_locs
                    ),
                    lambda: (state, new_agent_locs)
                )

            _, ys = jax.lax.scan(
                scan_fn,
                (key, collided_moved, collision_matrix, agent_locs, new_agent_locs),
                jnp.arange(self.num_agents)
            )

            final_locs = ys[-1]

            return final_locs

        def _fix_collisions(
            key: jnp.ndarray,
            collided_moved: jnp.ndarray,
            collision_matrix: jnp.ndarray,
            agent_locs: jnp.ndarray,
            new_agent_locs: jnp.ndarray
        ) -> Tuple[Tuple, jnp.ndarray]:
            def select_random_true_index(key, array):
                # Calculate the cumulative sum of True values
                cumsum_array = jnp.cumsum(array)

                # Count the number of True values
                true_count = cumsum_array[-1]

                # Generate a random index in the range of the number of True
                # values
                rand_index = jax.random.randint(
                    key,
                    (1,),
                    0,
                    true_count
                )

                # Find the position of the random index within the cumulative
                # sum
                chosen_index = jnp.argmax(cumsum_array > rand_index)

                return chosen_index
            # Pick one from all who collided & moved
            colliders_idx = jnp.argmax(collided_moved)

            collisions = collision_matrix[colliders_idx]

            # Check whether any of collision participants didn't move
            collision_subjects = jnp.where(
                collisions,
                collided_moved,
                collisions
            )
            collision_mask = collisions == collision_subjects
            stayed = jnp.all(collision_mask)
            stayed_mask = jnp.logical_and(~stayed, ~collision_mask)
            stayed_idx = jnp.where(
                jnp.max(stayed_mask) > 0,
                jnp.argmax(stayed_mask),
                0
            )

            # Prepare random agent selection
            k1, k2 = jax.random.split(key, 2)
            rand_idx = select_random_true_index(k1, collisions)
            collisions_rand = collisions.at[rand_idx].set(False) # <<<< PROBLEM LINE        
            new_locs_rand = jax.vmap(
                lambda p, l, n: jnp.where(p, l, n)
            )(
                collisions_rand,
                agent_locs,
                new_agent_locs
            )

            collisions_stayed = jax.lax.select(
                jnp.max(stayed_mask) > 0,
                collisions.at[stayed_idx].set(False),
                collisions_rand
            )
            new_locs_stayed = jax.vmap(
                lambda p, l, n: jnp.where(p, l, n)
            )(
                collisions_stayed,
                agent_locs,
                new_agent_locs
            )

            # Choose between the two scenarios - revert positions if
            # non-mover exists, otherwise choose random agent if all moved
            new_agent_locs = jnp.where(
                stayed,
                new_locs_rand,
                new_locs_stayed
            )

            # Update move bools to reflect the post-collision positions
            collided_moved = jnp.clip(collided_moved - collisions, 0, 1)
            collision_matrix = collision_matrix.at[colliders_idx].set(
                [False] * collisions.shape[0]
            )
            return ((k2, collided_moved, collision_matrix, agent_locs, new_agent_locs), new_agent_locs)
        
        def combine_channels(
                grid: jnp.ndarray,
                agent: int,
                angles: jnp.ndarray,
                state: State,
            ):
            '''
            Function to enforce symmetry in observations & generate final
            feature representation; current agent is permuted to first 
            position in the feature dimension.
            
            Args:
                - grid: jax ndarray of current agent's obs grid
                - agent: int, an index indicating current agent number
                - angles: jnp.ndarray of current agents' relative orientation
                in current agent's obs grid
                - state: State, the env state obj
            Returns:
                - grid with current agent [x] permuted to 1st position (after
                the first 5 "Items" features) in the feature dimension, "other" 
                [x] agent 2nd, angle [x, x, x, x] 3rd, for a final obs grid of 
                shape (11, 11, 11) - 5 items + 1 self + 1 other + 4 orientation
            '''
            def move_and_collapse(
                    x: jnp.ndarray,
                    angle: jnp.ndarray,
                ) -> jnp.ndarray:

                # get agent's one-hot
                agent_element = jnp.array([jnp.int8(x[agent])])

                # mask to check if any other agent exists there
                mask = x[len(Items)-1:] > 0

                # does an agent exist which is not the subject?
                other_agent = jnp.int8(
                    jnp.logical_and(
                        jnp.any(mask),
                        jnp.logical_not(
                            agent_element
                        )
                    )
                )

                # build extension
                extension = jnp.concatenate(
                    [
                        agent_element,  # 1 channel
                        other_agent,    # 1 channel
                        angle,          # 4 channels
                    ],
                    axis=-1
                )

                # build final feature vector
                final_vec = jnp.concatenate(
                    [x[:len(Items)], extension],
                    axis=-1
                )

                return final_vec

            new_grid = jax.vmap(
                jax.vmap(
                    move_and_collapse
                )
            )(grid, angles)
            return new_grid
        
        def check_relative_orientation(
                agent: int,
                agent_locs: jnp.ndarray,
                grid: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Check's relative orientations of all other agents in view of
            current agent.
            
            Args:
                - agent: int, an index indicating current agent number
                - agent_locs: jax ndarray of agent locations (x, y, direction)
                - grid: jax ndarray of current agent's obs grid
                
            Returns:
                - grid with 1) int -1 in places where no agent exists, or
                where the agent is the current agent, and 2) int in range
                0-3 in cells of opposing agents indicating relative
                orientation to current agent.
            '''
            # we decrement by num of Items when indexing as we incremented by
            # 5 in constructor call (due to 5 non-agent Items enum & locations
            # are indexed from 0)
            idx = agent - len(Items)
            agents = jnp.delete(
                self._agents,
                idx,
                assume_unique_indices=True
            )
            curr_agent_dir = agent_locs[idx, 2]

            def calc_relative_direction(cell):
                cell_agent = cell - len(Items)
                cell_direction = agent_locs[cell_agent, 2]
                return (cell_direction - curr_agent_dir) % 4

            angle = jnp.where(
                jnp.isin(grid, agents),
                jax.vmap(calc_relative_direction)(grid),
                -1
            )

            return angle
        
        def rotate_grid(agent_loc: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
            '''
            Rotates agent's observation grid k * 90 degrees, depending on agent's
            orientation.

            Args:
                - agent_loc: jax ndarray of agent's x, y, direction
                - grid: jax ndarray of agent's obs grid

            Returns:
                - jnp.ndarray of new rotated grid.

            '''
            grid = jnp.where(
                agent_loc[2] == 1,
                jnp.rot90(grid, k=1, axes=(0, 1)),
                grid,
            )
            grid = jnp.where(
                agent_loc[2] == 2,
                jnp.rot90(grid, k=2, axes=(0, 1)),
                grid,
            )
            grid = jnp.where(
                agent_loc[2] == 3,
                jnp.rot90(grid, k=3, axes=(0, 1)),
                grid,
            )

            return grid

        def _get_obs_point(agent_loc: jnp.ndarray) -> jnp.ndarray:
            '''
            Obtain the position of top-left corner of obs map using
            agent's current location & orientation.
            Agent is centered in the observation window.

            Args: 
                - agent_loc: jnp.ndarray, agent x, y, direction.
            Returns:
                - x, y: ints of top-left corner of agent's obs map.
            '''
            
            x, y, direction = agent_loc

            # Add padding to get position in padded grid
            x, y = x + self.PADDING, y + self.PADDING

            # Center the agent: subtract half the observation size
            x = x - (self.OBS_SIZE // 2)
            y = y - (self.OBS_SIZE // 2)

            return x, y

        def _get_obs(state: State) -> jnp.ndarray:
            '''
            Obtain the agent's observation of the grid.

            Args: 
                - state: State object containing env state.
            Returns:
                - jnp.ndarray of grid observation.
            '''
            # create state
            grid = jnp.pad(
                state.grid,
                ((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)),
                constant_values=Items.wall,
            )

            # obtain all agent obs-points
            agent_start_idxs = jax.vmap(_get_obs_point)(state.agent_locs)

            dynamic_slice = partial(
                jax.lax.dynamic_slice,
                operand=grid,
                slice_sizes=(self.OBS_SIZE, self.OBS_SIZE)
            )

            # obtain agent obs grids
            grids = jax.vmap(dynamic_slice)(start_indices=agent_start_idxs)

            # rotate agent obs grids
            grids = jax.vmap(rotate_grid)(state.agent_locs, grids)

            angles = jax.vmap(
                check_relative_orientation,
                in_axes=(0, None, 0)
            )(
                self._agents,
                state.agent_locs,
                grids
            )

            angles = jax.nn.one_hot(angles, 4)

            # one-hot (now includes the Items.empty channel)
            grids = jax.nn.one_hot(
                grids,
                num_agents + len(Items) - 1,
                dtype=jnp.int8
            )
            
            # make index len(Item) always the current agent
            # and sum all others into an "other" agent
            grids = jax.vmap(
                combine_channels,
                in_axes=(0, 0, 0, None) 
            )(
                grids,
                self._agents,
                angles,
                state  
            )

            frozen_mask = (state.freeze > 0)[:, None, None, None]
            grids = jnp.where(frozen_mask, 0, grids)

            return grids


        def _interact(
            key: jnp.ndarray, state: State, actions: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray, State, jnp.ndarray]:
            
            # 1. Flatten inputs
            actions = actions.squeeze()
            
            # 2. Identify who wants to zap
            zaps = jnp.isin(actions, jnp.array([Actions.zap_forward]))
            
            # 3. Safety Lock: Frozen agents cannot zap
            is_unfrozen = state.freeze == 0
            zaps = jnp.logical_and(zaps, is_unfrozen)

            interact_idx = jnp.int16(Items.interact)
            
            # Define a safe purgatory location for blocked beams 
            # (Using your requested offset logic to be safe)
            purgatory_val = -self.OBS_SIZE * 2 
            purgatory_loc = jnp.array([purgatory_val, purgatory_val, 0])

            # remove old interacts
            state = state.replace(grid=jnp.where(
                state.grid == interact_idx, jnp.int16(Items.empty), state.grid))

            # --- HELPER: Safe Grid Reader ---
            def get_item_safe(r, c):
                # check bounds
                in_bounds_r = (r >= 0) & (r < self.GRID_SIZE_ROW)
                in_bounds_c = (c >= 0) & (c < self.GRID_SIZE_COL)
                is_valid = in_bounds_r & in_bounds_c
                
                # safe indices for reading (prevents crash, but result only used if valid)
                safe_r = jnp.clip(r, 0, self.GRID_SIZE_ROW - 1)
                safe_c = jnp.clip(c, 0, self.GRID_SIZE_COL - 1)
                
                # If out of bounds, return WALL. Else return actual grid item.
                return jnp.where(
                    is_valid,
                    state.grid[safe_r, safe_c],
                    jnp.int16(Items.wall)
                )
            # --------------------------------

            def generate_beam_targets(agent_loc: jnp.ndarray) -> jnp.ndarray:
                all_targets = []
                
                # Generate all potential targets
                for forward in range(1, self.zap_beam_length + 1):
                    if forward == self.zap_beam_length:
                        target = agent_loc + forward * STEP[agent_loc[2]]
                        all_targets.append((forward, 0, target))
                    else:
                        for side_offset in range(-self.zap_beam_width, self.zap_beam_width + 1):
                            target = agent_loc + forward * STEP[agent_loc[2]]
                            
                            if side_offset > 0:
                                perp_dir = (agent_loc[2] + 1) % 4
                                target = target + side_offset * STEP[perp_dir]
                            elif side_offset < 0:
                                perp_dir = (agent_loc[2] - 1) % 4
                                target = target + abs(side_offset) * STEP[perp_dir]
                            
                            all_targets.append((forward, side_offset, target))
                
                forwards = jnp.array([t[0] for t in all_targets])
                sides = jnp.array([t[1] for t in all_targets])
                targets = jnp.stack([t[2] for t in all_targets])
                
                # Check walls using SAFE reader
                # This treats map edges as walls correctly without wrapping/clipping to self
                grid_items = jax.vmap(get_item_safe)(targets[:, 0], targets[:, 1])
                is_wall = grid_items == Items.wall
                
                # For each lane, find first wall and block everything after
                def check_blocked(idx):
                    forward = forwards[idx]
                    side = sides[idx]
                    is_wall_here = is_wall[idx]
                    
                    same_lane_mask = sides == side
                    earlier_mask = forwards < forward
                    lane_and_earlier = same_lane_mask & earlier_mask
                    
                    wall_before = jnp.any(jnp.where(lane_and_earlier, is_wall, False))
                    
                    return jnp.logical_or(is_wall_here, wall_before)
                
                blocked = jax.vmap(check_blocked)(jnp.arange(len(forwards)))
                
                # Set blocked targets to Purgatory
                valid_targets = jnp.where(
                    blocked[:, None],
                    purgatory_loc,
                    targets
                )
                
                return valid_targets

            # Generate beam targets
            all_beam_targets = jax.vmap(generate_beam_targets)(state.agent_locs)
            
            # Flatten
            all_zaped_locs = all_beam_targets.reshape(-1, 3)
            
            # Valid mask (checks if row is >= 0, filtering out purgatory)
            valid_mask = all_zaped_locs[:, 0] >= 0

            # Expand zaps
            num_beam_cells = all_beam_targets.shape[1]
            zaps_expanded = jnp.repeat(zaps, num_beam_cells, axis=0)
            zaps_expanded = zaps_expanded & valid_mask

            # Identify what is hit
            def zaped_grid_check(loc, z):
                # We use get_item_safe again here just to be doubly sure 
                # we don't read garbage indices, though valid_mask should catch it.
                item = get_item_safe(loc[0], loc[1])
                return jnp.where(z, item, -1)

            all_zaped_grid = jax.vmap(zaped_grid_check)(all_zaped_locs, zaps_expanded)

            def check_hit_agent(a):
                return jnp.isin(a, all_zaped_grid)
            
            hit_agents = jax.vmap(check_hit_agent)(self._agents)
            
            FREEZE_DURATION = 25
            new_freeze = jnp.where(
                hit_agents,
                FREEZE_DURATION,
                state.freeze
            )
                        
            # Update grid visuals
            aux_grid = jnp.copy(state.grid)

            # Get items for visual update
            # If it's an empty space, we draw the interact beam.
            # If it's an item/agent, we leave it alone (or you can choose to overwrite).
            current_items = jax.vmap(get_item_safe)(all_zaped_locs[:, 0], all_zaped_locs[:, 1])
            
            beam_items = jnp.where(
                current_items != Items.empty, # If not empty
                current_items, # Keep existing item
                interact_idx   # Else draw beam
            )
            
            qualified_to_zap = zaps_expanded.squeeze()

            def update_grid_batch(locs, items, qualified, grid):
                # Only write if qualified AND valid index
                in_bounds_r = (locs[:, 0] >= 0) & (locs[:, 0] < self.GRID_SIZE_ROW)
                in_bounds_c = (locs[:, 1] >= 0) & (locs[:, 1] < self.GRID_SIZE_COL)
                should_write = qualified & in_bounds_r & in_bounds_c
                
                # Safe clip for the write operation index (only executes if should_write is True)
                safe_r = jnp.clip(locs[:, 0], 0, self.GRID_SIZE_ROW - 1)
                safe_c = jnp.clip(locs[:, 1], 0, self.GRID_SIZE_COL - 1)
                
                return grid.at[safe_r, safe_c].set(
                    jax.vmap(jnp.where)(
                        should_write,
                        items,
                        aux_grid[safe_r, safe_c]
                    )
                )

            aux_grid = update_grid_batch(all_zaped_locs, beam_items, qualified_to_zap, aux_grid)
            
            state = state.replace(
                grid=jnp.where(jnp.any(zaps), aux_grid, state.grid),
                freeze=new_freeze
            )

            return state

        def _step(
            key: chex.PRNGKey,
            state: State,
            actions: jnp.ndarray
        ):
            """Step the environment."""
            actions = jnp.array(actions)

            # regrow apple
            grid_apple = state.grid

            def count_apple(apple_locs):

                apple_nums = jnp.where((grid_apple[apple_locs[0]-1, apple_locs[1]] == 3) & (apple_locs[0]-1 >=0), 1, 0) + \
                                jnp.where((grid_apple[apple_locs[0]+1, apple_locs[1]] == 3) & (apple_locs[0]+1 < self.GRID_SIZE_ROW), 1, 0) + \
                                jnp.where((grid_apple[apple_locs[0], apple_locs[1]-1] == 3) & (apple_locs[1]-1 >=0), 1, 0) + \
                                jnp.where((grid_apple[apple_locs[0], apple_locs[1]+1] == 3) & (apple_locs[1]+1 < self.GRID_SIZE_COL), 1, 0)+ \
                                jnp.where((grid_apple[apple_locs[0]-2, apple_locs[1]] == 3) & (apple_locs[0]-2 >=0), 1, 0) + \
                                jnp.where((grid_apple[apple_locs[0]+2, apple_locs[1]] == 3) & (apple_locs[0]+2 < self.GRID_SIZE_ROW), 1 ,0) + \
                                jnp.where((grid_apple[apple_locs[0], apple_locs[1]-2] == 3) & (apple_locs[1]-2 >=0), 1, 0) + \
                                jnp.where((grid_apple[apple_locs[0], apple_locs[1]+2] == 3) & (apple_locs[1]+2 < self.GRID_SIZE_COL), 1 ,0) + \
                                jnp.where((grid_apple[apple_locs[0]-1, apple_locs[1]-1] == 3) & (apple_locs[0]-1 >=0) & (apple_locs[1]-1 >=0), 1, 0) + \
                                jnp.where((grid_apple[apple_locs[0]-1, apple_locs[1]+1] == 3) & (apple_locs[0]-1 >=0) & (apple_locs[1]+1 < self.GRID_SIZE_COL), 1, 0) + \
                                jnp.where((grid_apple[apple_locs[0]+1, apple_locs[1]-1] == 3) & (apple_locs[1]+1 < self.GRID_SIZE_COL) & (apple_locs[1]-1 >=0), 1, 0) + \
                                jnp.where((grid_apple[apple_locs[0]+1, apple_locs[1]+1] == 3) & (apple_locs[0]+1 < self.GRID_SIZE_ROW) & (apple_locs[1]+1 < self.GRID_SIZE_COL) , 1, 0)
                
                return apple_nums

            near_apple_nums = jax.vmap(count_apple)(self.SPAWNS_APPLE)
            
            def regrow_apple(apple_locs, near_apple, prob):
                new_apple = jnp.where((((grid_apple[apple_locs[0], apple_locs[1]] == Items.empty) & (near_apple == 0) & (prob > 1)) |
                                       (grid_apple[apple_locs[0], apple_locs[1]] == Items.apple) |
                                      ((grid_apple[apple_locs[0], apple_locs[1]] == Items.empty) & (near_apple >= 3) & (prob < 0.025)) |
                                      ((grid_apple[apple_locs[0], apple_locs[1]] == Items.empty) & (near_apple == 2) & (prob < 0.005)) |
                                      ((grid_apple[apple_locs[0], apple_locs[1]] == Items.empty) & (near_apple == 1) & (prob < 0.001)))
                                      ,  Items.apple, Items.empty)

                return new_apple
            
            prob = jax.random.uniform(key, shape=(len(self.SPAWNS_APPLE),))
            new_apple = jax.vmap(regrow_apple)(self.SPAWNS_APPLE, near_apple_nums, prob)

            new_apple_grid = grid_apple.at[self.SPAWNS_APPLE[:, 0], self.SPAWNS_APPLE[:, 1]].set(new_apple[:])
            state = state.replace(grid=new_apple_grid)
            
            # Remove agents from grid
            new_grid = state.grid.at[
                state.agent_locs[:, 0],
                state.agent_locs[:, 1]
            ].set(jnp.int16(Items.empty))
            
            # Only place unfrozen agents back on grid
            is_frozen = state.freeze > 0
            
            # Place only active (unfrozen) agents on grid
            def place_agent_if_active(grid, loc, agent_id, frozen):
                return jnp.where(
                    frozen,
                    grid,  # Don't place if frozen
                    grid.at[loc[0], loc[1]].set(agent_id)
                )
            
            for i in range(self.num_agents):
                new_grid = place_agent_if_active(
                    new_grid, 
                    state.agent_locs[i], 
                    self._agents[i], 
                    is_frozen[i]
                )
            
            state = state.replace(grid=new_grid)

            # Apply actions only to unfrozen agents
            actions = jnp.where(
                is_frozen,
                Actions.stay,  # Frozen agents can't act
                actions
            )

            # moving all agents
            key, subkey = jax.random.split(key)

            rotated_locs = jax.vmap(lambda p, a: jnp.int16(p + ROTATIONS[a]) % jnp.array([self.GRID_SIZE_ROW + 1, self.GRID_SIZE_COL + 1, 4], dtype=jnp.int16))(p=state.agent_locs, a=actions).squeeze()            
            
            # Check if the action is specifically "move_forward" (Index 2)
            is_move_forward = (actions == Actions.move_forward)
            agent_move = (actions == Actions.move_forward)
            
            # Helper: If moving forward, add STEP[current_direction] to position
            def apply_forward(p, move_bool):
                current_dir = p[2]
                forward_delta = STEP[current_dir] # Look up delta based on direction
                return jnp.where(move_bool, p + forward_delta, p)

            all_new_locs = jax.vmap(apply_forward)(rotated_locs, is_move_forward)

            all_new_locs = jax.vmap(
                jnp.clip,
                in_axes=(0, None, None)
            )(
                all_new_locs,
                jnp.array([0, 0, 0], dtype=jnp.int16),
                jnp.array(
                    [self.GRID_SIZE_ROW - 1, self.GRID_SIZE_COL - 1, 3],
                    dtype=jnp.int16
                ),
            ).squeeze()

            # Apply Unique Purgatory Locations (After clipping!)
            # We create a "Parking Lot" far away so agents don't collide with each other
            base_purgatory = -self.OBS_SIZE * 2
            
            # Create distinct column positions: [-22, -23, -24, ...]
            # This ensures check_collision returns False for frozen agents
            p_cols = jnp.int16(base_purgatory - jnp.arange(self.num_agents))
            p_rows = jnp.full((self.num_agents,), base_purgatory, dtype=jnp.int16)
            p_dirs = jnp.zeros((self.num_agents,), dtype=jnp.int16)
            
            purgatory_locs = jnp.stack([p_rows, p_cols, p_dirs], axis=-1)

            all_new_locs = jnp.where(
                is_frozen[:, None],
                purgatory_locs,
                all_new_locs
            )

            # if you bounced back to your original space,
            # change your move to stay (for collision logic)
            agents_move = jax.vmap(lambda n, p: jnp.any(n[:2] != p[:2]))(n=all_new_locs, p=state.agent_locs)

            # generate bool mask for agents colliding
            collision_matrix = check_collision(all_new_locs)

            # sum & subtract "self-collisions"
            collisions = jnp.sum(
                collision_matrix,
                axis=-1,
                dtype=jnp.int8
            ) - 1
            collisions = jnp.minimum(collisions, 1)

            # identify which of those agents made wrong moves
            collided_moved = jnp.maximum(
                collisions - ~agents_move,
                0
            )

            # fix collisions at the correct indices
            new_locs = jax.lax.cond(
                jnp.max(collided_moved) > 0,
                lambda: fix_collisions(
                    key,
                    collided_moved,
                    collision_matrix,
                    state.agent_locs,
                    all_new_locs
                ),
                lambda: all_new_locs
            )

            # fix collisions
            # TODO - fix this to be more efficient; agents moving would be less efficient.
            condition = jnp.where((state.grid[new_locs[:, 0], new_locs[:, 1]] != Items.empty) & 
                                  (state.grid[new_locs[:, 0], new_locs[:, 1]] != Items.apple), True, False)
            condition_3d = jnp.stack([condition, condition, condition], axis=-1)

            move_mask_3d = jnp.stack([agent_move, agent_move, agent_move], axis=-1)

            # Only revert when (movement action AND collision)
            new_locs = jnp.where(move_mask_3d & condition_3d,
                                state.agent_locs,
                                new_locs)

            # update inventories
            def apple_matcher(p: jnp.ndarray) -> jnp.ndarray:
                c_matches = jnp.array([
                    state.grid[p[0], p[1]] == Items.apple
                    ])
                return c_matches
            
            apple_matches = jax.vmap(apple_matcher)(p=new_locs)

            new_invs = state.agent_invs + apple_matches

            state = state.replace(
                agent_invs=new_invs
            )

            # update grid
            old_grid = state.grid

            new_grid = old_grid.at[
                state.agent_locs[:, 0],
                state.agent_locs[:, 1]
            ].set(
                jnp.int16(Items.empty)
            )
            
            # Only place ACTIVE agents back on the grid
            # We use a loop or vmap to safely handle the update without indexing errors 
            # from the (-1, -1) purgatory agents
            def place_active(grid, loc, agent_id, frozen):
                # If frozen, return grid as is (don't draw). 
                # If not frozen, draw agent at loc.
                return jax.lax.cond(
                    frozen > 0,
                    lambda g: g,
                    lambda g: g.at[loc[0], loc[1]].set(agent_id),
                    grid
                )

            # Sequentially update grid (scan is cleaner than loop in JAX)
            def update_body(carry_grid, inputs):
                loc, agent_id, frozen = inputs
                return place_active(carry_grid, loc, agent_id, frozen), None

            new_grid, _ = jax.lax.scan(
                update_body, 
                new_grid, 
                (new_locs, self._agents, state.freeze)
            )
            
            state = state.replace(grid=new_grid)

            # update agent locations
            state = state.replace(agent_locs=new_locs)

            state = _interact(key, state, actions)

            # Decrement freeze timers
            new_freeze = jnp.maximum(state.freeze - 1, 0)
            state = state.replace(freeze=new_freeze)        

            # Check which agents just became unfrozen (freeze went from 1 to 0)
            just_unfrozen = (state.freeze == 1) & (new_freeze == 0)

            # 1. Identify valid spawn points (Empty tiles only)
            # We look at the grid we just updated in Step 2 to ensure we don't spawn on existing agents
            valid_spawns_mask = (state.grid[self.SPAWNS_PLAYERS[:, 0], self.SPAWNS_PLAYERS[:, 1]] == Items.empty)
            
            # 2. Convert mask to probabilities
            # If a tile is empty, weight is 1.0. If occupied, weight is 0.0.
            respawn_weights = valid_spawns_mask.astype(jnp.float32)
            
            # Safety: Add a tiny epsilon to avoid division by zero if ALL spawns are blocked (rare but possible)
            # In that worst case, it defaults to uniform random selection over all spawns.
            respawn_weights = respawn_weights + 1e-6
            respawn_weights = respawn_weights / jnp.sum(respawn_weights)

            # 3. Select indices based on weights
            spawn_indices = jax.random.choice(
                subkey,
                a=len(self.SPAWNS_PLAYERS),
                shape=(self.num_agents,), 
                p=respawn_weights  # <--- This ensures we pick empty spots
            )
            
            selected_spawns = self.SPAWNS_PLAYERS[spawn_indices]
            
            # Generate random directions
            player_dir = jax.random.randint(subkey, shape=(self.num_agents,), minval=0, maxval=3, dtype=jnp.int8)

            re_agent_locs = jnp.array(
                [selected_spawns[:, 0], selected_spawns[:, 1], player_dir],
                dtype=jnp.int16
            ).T

            # 4. Apply new locations to just_unfrozen agents
            # (Others keep their current 'new_locs')
            new_locs = jnp.where(
                just_unfrozen[:, None],
                re_agent_locs,
                new_locs
            )

            # 5. Immediately draw the respawned agents onto the grid
            # If we don't do this, they will be invisible for 1 frame
            def place_respawned(grid, loc, agent_id, just_born):
                return jax.lax.cond(
                    just_born,
                    lambda g: g.at[loc[0], loc[1]].set(agent_id),
                    lambda g: g,
                    grid
                )

            def respawn_body(carry_grid, inputs):
                loc, agent_id, born = inputs
                return place_respawned(carry_grid, loc, agent_id, born), None

            final_grid, _ = jax.lax.scan(
                respawn_body, 
                state.grid, # This is the grid from Step 2
                (new_locs, self._agents, just_unfrozen)
            )

            state = state.replace(agent_locs=new_locs, freeze=new_freeze, grid=final_grid)

            if self.shared_rewards:
                rewards = jnp.zeros((self.num_agents, 1))
                original_rewards = jnp.where(apple_matches, 1, rewards)

                rewards_sum_all_agents = jnp.zeros((self.num_agents, 1))
                rewards_sum = jnp.sum(original_rewards)
                rewards_sum_all_agents += rewards_sum
                rewards = rewards_sum_all_agents
                info = {
                    "original_rewards": original_rewards.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
            elif self.inequity_aversion:
                rewards = jnp.zeros((self.num_agents, 1))
                original_rewards = jnp.where(apple_matches, 1, rewards) * self.num_agents
                if self.smooth_rewards:
                    should_smooth = (state.inner_t % 1) == 0
                    new_smooth_rewards = 0.99 * 0.01* state.smooth_rewards + original_rewards
                    rewards,disadvantageous,advantageous = self.get_inequity_aversion_rewards_immediate(new_smooth_rewards, self.inequity_aversion_target_agents, state.inner_t, self.inequity_aversion_alpha, self.inequity_aversion_beta)
                    state = state.replace(smooth_rewards=new_smooth_rewards)
                    info = {
                    "original_rewards": original_rewards.squeeze(),
                    "smooth_rewards": state.smooth_rewards.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
                else:
                    rewards,disadvantageous,advantageous = self.get_inequity_aversion_rewards_immediate(original_rewards, self.inequity_aversion_target_agents, state.inner_t, self.inequity_aversion_alpha, self.inequity_aversion_beta)
                    info = {
                    "original_rewards": original_rewards.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
            elif self.svo:
                rewards = jnp.zeros((self.num_agents, 1))
                original_rewards = jnp.where(apple_matches, 1, rewards) * self.num_agents
                rewards, theta = self.get_svo_rewards(original_rewards, self.svo_w, self.svo_ideal_angle_degrees, self.svo_target_agents)
                info = {
                    "original_rewards": original_rewards.squeeze(),
                    "svo_theta": theta.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
            else:
                rewards = jnp.zeros((self.num_agents, 1))
                rewards = jnp.where(apple_matches, 1, rewards)
                info = {}
            
            AppleCount = jnp.sum(state.grid == Items.apple)
            info["apple_count"] = jnp.zeros((self.num_agents, 1)).squeeze() + AppleCount
            info["agent_locs"] = state.agent_locs
            info["agent_freeze"] = state.freeze
            
            
            state_nxt = State(
                agent_locs=state.agent_locs,
                agent_invs=state.agent_invs,
                inner_t=state.inner_t + 1,
                outer_t=state.outer_t,
                grid=state.grid,
                apples=state.apples,
                freeze=state.freeze,
                reborn_locs=state.reborn_locs
            )

            # now calculate if done for inner or outer episode
            inner_t = state_nxt.inner_t
            outer_t = state_nxt.outer_t
            reset_inner = inner_t == num_inner_steps

            # if inner episode is done, return start state for next game
            state_re = _reset_state(key)

            state_re = state_re.replace(outer_t=outer_t + 1)
            state = jax.tree_util.tree_map(
                lambda x, y: jnp.where(reset_inner, x, y),
                state_re,
                state_nxt,
            )
            outer_t = state.outer_t
            reset_outer = outer_t == num_outer_steps
            done = {f'{a}': reset_outer for a in self.agents}
            done["__all__"] = reset_outer

            obs = _get_obs(state)
            rewards = jnp.where(
                reset_inner,
                jnp.zeros_like(rewards, dtype=jnp.int16),
                rewards
            )


            return (
                obs,
                state,
                rewards.squeeze(),
                done,
                info,
            )

        def _reset_state(
            key: jnp.ndarray
        ) -> State:
            key, subkey = jax.random.split(key)

            # Find the free spaces in the grid
            grid = jnp.zeros((self.GRID_SIZE_ROW, self.GRID_SIZE_COL), jnp.int16)

            agent_pos = jax.random.permutation(subkey, self.SPAWNS_PLAYERS)[:num_agents]
            wall_pos = self.SPAWNS_WALL

            apple_pos = self.SPAWNS_APPLE

            grid = grid.at[
                apple_pos[:, 0],
                apple_pos[:, 1]
            ].set(jnp.int16(Items.apple))

            grid = grid.at[
                wall_pos[:, 0],
                wall_pos[:, 1]
            ].set(jnp.int16(Items.wall))


            player_dir = jax.random.randint(
                subkey, shape=(
                    num_agents,
                    ), minval=0, maxval=3, dtype=jnp.int8
            )

            agent_locs = jnp.array(
                [agent_pos[:, 0], agent_pos[:, 1], player_dir],
                dtype=jnp.int16
            ).T

            grid = grid.at[
                agent_locs[:, 0],
                agent_locs[:, 1]
            ].set(jnp.int16(self._agents))

            freeze = jnp.zeros(num_agents, dtype=jnp.int16)  # All agents start unfrozen

            return State(
                agent_locs=agent_locs,
                agent_invs=jnp.array([(0,0)]*num_agents, dtype=jnp.int8),
                inner_t=0,
                outer_t=0,
                grid=grid,
                apples=apple_pos,

                freeze=freeze,
                reborn_locs = agent_locs
            )

        def reset(
            key: jnp.ndarray
        ) -> Tuple[jnp.ndarray, State]:
            state = _reset_state(key)
            obs = _get_obs(state)
            return obs, state
        ################################################################################
        # if you want to test whether it can run on gpu, activate following code
        # overwrite Gymnax as it makes single-agent assumptions
        if jit:
            self.step_env = jax.jit(_step)
            self.reset = jax.jit(reset)
            self.get_obs_point = jax.jit(_get_obs_point)
        else:
            # if you want to see values whilst debugging, don't jit
            self.step_env = _step
            self.reset = reset
            self.get_obs_point = _get_obs_point
        ################################################################################

    @property
    def name(self) -> str:
        """Environment name."""
        return "Harvest_timeout"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(Actions)

    def action_space(
        self, agent_id: Union[int, None] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(Actions))

    def observation_space(self) -> spaces.Dict:
        """Observation space of the environment."""
        _shape_obs = (
            (self.OBS_SIZE, self.OBS_SIZE, len(Items) + 6)  # CHANGED: 5 + 6 = 11 channels
            if self.cnn
            else (self.OBS_SIZE**2 * (len(Items) + 6),)
        )

        return spaces.Box(
                low=0, high=1E9, shape=_shape_obs, dtype=jnp.uint8
            ), _shape_obs
    
    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        _shape = (
            (self.GRID_SIZE_ROW, self.GRID_SIZE_COL, NUM_TYPES + 4)
            if self.cnn
            else (self.GRID_SIZE_ROW* self.GRID_SIZE_COL * (NUM_TYPES + 4),)
        )
        return spaces.Box(low=0, high=1, shape=_shape, dtype=jnp.uint8)
    
    def render_tile(
        self,
        obj: int,
        agent_dir: Union[int, None] = None,
        agent_hat: bool = False,
        highlight: bool = False,
        tile_size: int = 32,
        subdivs: int = 3,
    ) -> onp.ndarray:
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (agent_dir, agent_hat, highlight, tile_size)
        if obj:
            key = (obj, 0, 0, 0) + key if obj else key

        if key in self.tile_cache:
            return self.tile_cache[key]

        img = onp.full(
            shape=(tile_size * subdivs, tile_size * subdivs, 3),
            fill_value=(210, 190, 140),
            dtype=onp.uint8,
        )


    # class Items(IntEnum):

        if obj in self._agents:
            # Draw the agent
            agent_color = self.PLAYER_COLOURS[obj-len(Items)]
        elif obj == Items.apple:
            fill_coords(
                img, point_in_circle(0.5, 0.5, 0.31), (102, 0, 0)
            )
        elif obj == Items.wall:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (127.0, 127.0, 127.0))
        elif obj == Items.interact:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (188.0, 189.0, 34.0))
        elif obj == 99:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (44.0, 160.0, 44.0))
        elif obj == 100:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (214.0, 39.0, 40.0))
        elif obj == 101:
            # white square
            fill_coords(img, point_in_rect(0, 1, 0, 1), (255.0, 255.0, 255.0))

        # Overlay the agent on top
        if agent_dir is not None:
            if agent_hat:
                tri_fn = point_in_triangle(
                    (0.12, 0.19),
                    (0.87, 0.50),
                    (0.12, 0.81),
                    0.3,
                )

                # Rotate the agent based on its direction
                tri_fn = rotate_fn(
                    tri_fn,
                    cx=0.5,
                    cy=0.5,
                    theta=0.5 * math.pi * (1 - agent_dir),
                )
                fill_coords(img, tri_fn, (255.0, 255.0, 255.0))

            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
                0.0,
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(
                tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * (1 - agent_dir)
            )
            fill_coords(img, tri_fn, agent_color)

        # # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        self.tile_cache[key] = img
        return img

    def render(
        self,
        state: State,
    ) -> onp.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
        tile_size = 32
        highlight_mask = onp.zeros_like(onp.array(self.GRID))

        # Compute the total grid size
        width_px = self.GRID.shape[1] * tile_size
        height_px = self.GRID.shape[0] * tile_size

        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)
        grid = onp.array(state.grid)
        
        grid = onp.pad(
            grid, ((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)), constant_values=Items.wall
        )
        for a in range(self.num_agents):
            if state.freeze[a] > 0:
                continue

            startx, starty = self.get_obs_point(
                state.agent_locs[a]
            )
            highlight_mask[
                startx : startx + self.OBS_SIZE, starty : starty + self.OBS_SIZE
            ] = True

        # Render the grid
        for j in range(0, grid.shape[1]):
            for i in range(0, grid.shape[0]):
                cell = grid[i, j]
                if cell == 0:
                    cell = None
                agent_here = []
                for a in self._agents:
                    agent_here.append(cell == a)

                agent_dir = None
                for a in range(self.num_agents):
                    agent_dir = (
                        state.agent_locs[a,2].item()
                        if agent_here[a]
                        else agent_dir
                    )
                
                agent_hat = False

                tile_img = self.render_tile(
                    cell,
                    agent_dir=agent_dir,
                    agent_hat=agent_hat,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = i * tile_size
                ymax = (i + 1) * tile_size
                xmin = j * tile_size
                xmax = (j + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img
        img = onp.rot90(
            img[
                (self.PADDING - 1) * tile_size : -(self.PADDING - 1) * tile_size,
                (self.PADDING - 1) * tile_size : -(self.PADDING - 1) * tile_size,
                :,
            ],
            2,
        )
        return img

    def render_time(self, state, width_px) -> onp.array:
        inner_t = state.inner_t
        outer_t = state.outer_t
        tile_height = 32
        img = onp.zeros(shape=(2 * tile_height, width_px, 3), dtype=onp.uint8)
        tile_width = width_px // (self.num_inner_steps)
        j = 0
        for i in range(0, inner_t):
            ymin = j * tile_height
            ymax = (j + 1) * tile_height
            xmin = i * tile_width
            xmax = (i + 1) * tile_width
            img[ymin:ymax, xmin:xmax, :] = onp.int8(255)
        tile_width = width_px // (self.num_outer_steps)
        j = 1
        for i in range(0, outer_t):
            ymin = j * tile_height
            ymax = (j + 1) * tile_height
            xmin = i * tile_width
            xmax = (i + 1) * tile_width
            img[ymin:ymax, xmin:xmax, :] = onp.int8(255)
        return img
    
    def get_inequity_aversion_rewards_immediate(self, array, inner_t, target_agents=None, alpha=5, beta=0.05):
        """
        Calculate inequity aversion rewards using immediate rewards, based on equation (3) in the paper
        
        Args:
            array: shape: [num_agents, 1] immediate rewards r_i^t for each agent
            target_agents: list of agent indices to apply inequity aversion
            alpha: inequity aversion coefficient (when other agents' rewards are greater than self)
            beta: inequity aversion coefficient (when self's rewards are greater than others)
        Returns:
            subjective_rewards: adjusted subjective rewards u_i^t after inequity aversion
        """
        # Ensure correct input shape
        assert array.shape == (self.num_agents, 1), f"Expected shape ({self.num_agents}, 1), got {array.shape}"
        
        # Calculate inequality using immediate rewards
        r_i = array  # [num_agents, 1]
        r_j = jnp.transpose(array)  # [1, num_agents]
        
        # Calculate inequality
        disadvantageous = jnp.maximum(r_j - r_i, 0)  # when other agents' rewards are higher
        advantageous = jnp.maximum(r_i - r_j, 0)     # when self's rewards are higher
        
        # Create mask to exclude self-comparison
        mask = 1 - jnp.eye(self.num_agents)
        disadvantageous = disadvantageous * mask
        advantageous = advantageous * mask
        
        # Calculate inequality penalty
        n_others = self.num_agents - 1
        inequity_penalty = (alpha * jnp.sum(disadvantageous, axis=1, keepdims=True) +
                           beta * jnp.sum(advantageous, axis=1, keepdims=True)) / n_others

        # Calculate subjective rewards u_i^t = r_i^t - inequality penalty
        subjective_rewards = array - inequity_penalty

        subjective_rewards = jnp.where(jnp.all(array == 0), -(alpha + beta) * n_others, subjective_rewards)
        
        # Apply inequity aversion only to target agents if specified
        if target_agents is not None:
            target_agents_array = jnp.array(target_agents)
            agent_mask = jnp.zeros(self.num_agents, dtype=bool)
            agent_mask = agent_mask.at[target_agents_array].set(True)
            agent_mask = agent_mask.reshape(-1, 1)  # [num_agents, 1]
            return jnp.where(agent_mask, subjective_rewards, array),jnp.sum(disadvantageous, axis=1, keepdims=True),jnp.sum(advantageous, axis=1, keepdims=True)
        else:
            return subjective_rewards,jnp.sum(disadvantageous, axis=1, keepdims=True),jnp.sum(advantageous, axis=1, keepdims=True)

    def get_svo_rewards(self, array, w=0.5, ideal_angle_degrees=45, target_agents=None):
        """
        Reward shaping function based on Social Value Orientation (SVO)
        
        Args:
            array: shape: [num_agents, 1] immediate rewards r_i for each agent
            w: SVO weight to balance self-reward and social value (0 <= w <= 1)
               w=0 means completely selfish, w=1 means completely altruistic
            ideal_angle_degrees: ideal angle in degrees
               - 45 degrees means complete equality
               - 0 degrees means completely selfish
               - 90 degrees means completely altruistic
            target_agents: list of agent indices to apply SVO
        
        Returns:
            shaped_rewards: rewards adjusted by SVO
            theta: reward angle in radians
        """
        # Ensure correct input shape
        assert array.shape == (self.num_agents, 1), f"Expected shape ({self.num_agents}, 1), got {array.shape}"
        
        # Convert ideal angle from degrees to radians
        ideal_angle = (ideal_angle_degrees * jnp.pi) / 180.0
        
        # Calculate group average reward r_j (excluding self)
        mask = 1 - jnp.eye(self.num_agents)  # [num_agents, num_agents]
        # Modified: use matrix multiplication to calculate other agents' rewards
        others_rewards = jnp.matmul(mask, array)  # [num_agents, 1]
        mean_others = others_rewards / (self.num_agents - 1)  # divide by number of other agents
        
        # Calculate reward angle θ(R) = arctan(r_j / r_i)
        r_i = array  # [num_agents, 1]
        r_j = mean_others  # [num_agents, 1]
        theta = jnp.arctan2(r_j, r_i)
        
        # Calculate social value oriented utility
        # U(r_i, r_j) = r_i - w * |θ(R) - ideal_angle|
        angle_deviation = jnp.abs(theta - ideal_angle)
        svo_utility = r_i - self.num_agents * w * angle_deviation

        # Apply SVO only to target agents if specified
        if target_agents is not None:
            target_agents_array = jnp.array(target_agents)
            agent_mask = jnp.zeros(self.num_agents, dtype=bool)
            agent_mask = agent_mask.at[target_agents_array].set(True)
            agent_mask = agent_mask.reshape(-1, 1)  # [num_agents, 1]
            return jnp.where(agent_mask, svo_utility, array), theta
        else:
            return svo_utility, theta

    def get_standardized_svo_rewards(self, array, w=0.5, ideal_angle_degrees=45, target_agents=None):
        """
        Reward shaping function based on Social Value Orientation (SVO)
        """
        # Ensure correct input shape
        assert array.shape == (self.num_agents, 1), f"Expected shape ({self.num_agents}, 1), got {array.shape}"
        
        # Convert ideal angle from degrees to radians
        ideal_angle = (ideal_angle_degrees * jnp.pi) / 180.0
        
        # Calculate group average reward r_j (excluding self)
        mask = 1 - jnp.eye(self.num_agents)
        others_rewards = jnp.matmul(mask, array)
        mean_others = others_rewards / (self.num_agents - 1)
        
        # Calculate reward angle θ(R) = arctan(r_j / r_i)
        r_i = array
        r_j = mean_others
        theta = jnp.arctan2(r_j, r_i)
        
        # Convert angle to [0, 2π] range
        theta = (theta + 2 * jnp.pi) % (2 * jnp.pi)
        
        # Calculate angle deviation and normalize to [0, 1] range
        angle_deviation = jnp.abs(theta - ideal_angle)
        angle_deviation = jnp.minimum(angle_deviation, 2 * jnp.pi - angle_deviation)  # take minimum deviation
        normalized_deviation = angle_deviation / jnp.pi  # normalize to [0, 1]
        
        # Use multiplicative form of penalty instead of subtraction
        svo_utility = r_i * (1 - w * normalized_deviation)
        
        # Apply SVO only to target agents if specified
        if target_agents is not None:
            target_agents_array = jnp.array(target_agents)
            agent_mask = jnp.zeros(self.num_agents, dtype=bool)
            agent_mask = agent_mask.at[target_agents_array].set(True)
            agent_mask = agent_mask.reshape(-1, 1)
            return jnp.where(agent_mask, svo_utility, array), theta
        else:
            return svo_utility, theta