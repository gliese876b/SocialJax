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


class OptimizedRenderer:
    """
    Optimized rendering system with pre-rendering and caching.
    Add this to your Harvest_timeout class.
    Uses onp (numpy) for rendering operations since rendering is CPU-side.
    """
    
    def __init__(self, num_agents, player_colours, grid_size_row, grid_size_col, 
                 obs_size, padding, tile_size=16, subdivs=3):
        self.num_agents = num_agents
        self.PLAYER_COLOURS = player_colours
        self.GRID_SIZE_ROW = grid_size_row
        self.GRID_SIZE_COL = grid_size_col
        self.OBS_SIZE = obs_size
        self.PADDING = padding
        self.tile_size = tile_size
        self.subdivs = subdivs
        
        # Cache for all pre-rendered tiles (using regular numpy)
        self.tile_cache: Dict[Tuple[Any, ...], onp.ndarray] = {}
        
        # Pre-render all possible tiles at initialization
        self._prerender_all_tiles()
    
    def _prerender_all_tiles(self):
        """Pre-render all possible tile combinations"""
        print("Pre-rendering tiles...")
        
        # 1. Render empty/background tile
        self._render_and_cache_tile(None, agent_dir=None, agent_hat=False, 
                                     highlight=False)
        self._render_and_cache_tile(None, agent_dir=None, agent_hat=False, 
                                     highlight=True)
        
        # 2. Render all item types from your Items enum
        # Items: empty=0, wall=1, interact=2, apple=3, spawn_point=4
        # Plus special types: 99, 100, 101
        item_types = [0, 1, 2, 3, 4, 99, 100, 101]
        
        for item in item_types:
            for highlight in [False, True]:
                self._render_and_cache_tile(item, agent_dir=None, 
                                            agent_hat=False, highlight=highlight)
        
        # 3. Render all agent orientations (4 directions * num_agents * hat variants)
        # Agents are offset by len(Items) = 5 in your code
        for agent_idx in range(self.num_agents):
            agent_id = agent_idx + 5  # Offset by len(Items)
            for direction in range(4):
                for agent_hat in [False, True]:
                    for highlight in [False, True]:
                        self._render_and_cache_tile(agent_id, agent_dir=direction,
                                                    agent_hat=agent_hat, 
                                                    highlight=highlight)
        
        print(f"Pre-rendered {len(self.tile_cache)} tiles")
    
    def _render_and_cache_tile(self, obj, agent_dir=None, agent_hat=False, 
                                highlight=False):
        """Render a single tile and store it in cache"""
        # Create cache key
        key = (obj, agent_dir, agent_hat, highlight, self.tile_size)
        
        # Skip if already cached
        if key in self.tile_cache:
            return
        
        # Create base image (using onp for CPU rendering)
        img = onp.full(
            shape=(self.tile_size * self.subdivs, 
                   self.tile_size * self.subdivs, 3),
            fill_value=(210, 190, 140),
            dtype=onp.uint8,
        )
        
        # Render object/item based on your Items enum
        if obj is not None and obj >= 5:  # Agent
            agent_idx = obj - 5
            # Don't render agent body here, only in overlay section below
            pass
        elif obj == 3:  # Items.apple
            self._fill_circle(img, 0.5, 0.5, 0.31, (102, 0, 0))
        elif obj == 1:  # Items.wall
            self._fill_rect(img, 0, 1, 0, 1, (127, 127, 127))
        elif obj == 2:  # Items.interact
            self._fill_rect(img, 0, 1, 0, 1, (188, 189, 34))
        elif obj == 99:
            self._fill_rect(img, 0, 1, 0, 1, (44, 160, 44))
        elif obj == 100:
            self._fill_rect(img, 0, 1, 0, 1, (214, 39, 40))
        elif obj == 101:
            self._fill_rect(img, 0, 1, 0, 1, (255, 255, 255))
        
        # Overlay agent if needed (matches your original render_tile logic)
        if agent_dir is not None:
            if obj is not None and obj >= 5:
                agent_idx = obj - 5
                agent_color = self.PLAYER_COLOURS[agent_idx]
            else:
                # Should not happen, but fallback
                agent_color = (255, 255, 255)
            
            if agent_hat:
                self._draw_agent_triangle(img, agent_dir, (255, 255, 255), 
                                         border=0.3)
            
            self._draw_agent_triangle(img, agent_dir, agent_color, border=0.0)
        
        # Apply highlight
        if highlight:
            self._highlight_img(img)
        
        # Downsample for anti-aliasing
        img = self._downsample(img, self.subdivs)
        
        # Store in cache
        self.tile_cache[key] = img
    
    def _fill_circle(self, img, cx, cy, r, color):
        """Optimized circle drawing using vectorized operations"""
        h, w = img.shape[:2]
        
        # Create coordinate grids (using onp)
        y, x = onp.ogrid[:h, :w]
        
        # Normalize coordinates
        xf = (x + 0.5) / w
        yf = (y + 0.5) / h
        
        # Calculate distance from center
        mask = (xf - cx)**2 + (yf - cy)**2 <= r**2
        img[mask] = color
    
    def _fill_rect(self, img, xmin, xmax, ymin, ymax, color):
        """Optimized rectangle drawing - just fill the region"""
        h, w = img.shape[:2]
        
        # Convert normalized coords to pixels
        x0 = int(xmin * w)
        x1 = int(xmax * w)
        y0 = int(ymin * h)
        y1 = int(ymax * h)
        
        img[y0:y1, x0:x1] = color
    
    def _draw_agent_triangle(self, img, direction, color, border=0.0):
        """Draw agent triangle with rotation (matches your original logic)"""
        h, w = img.shape[:2]
        
        # Define triangle points (normalized coordinates) - same as your original
        p1 = onp.array([0.12, 0.19])
        p2 = onp.array([0.87, 0.50])
        p3 = onp.array([0.12, 0.81])
        
        # Rotate triangle based on direction (same as your original)
        theta = 0.5 * math.pi * (1 - direction)
        cx, cy = 0.5, 0.5
        
        def rotate_point(p):
            x, y = p[0] - cx, p[1] - cy
            x_rot = cx + x * math.cos(theta) - y * math.sin(theta)
            y_rot = cy + y * math.cos(theta) + x * math.sin(theta)
            return onp.array([x_rot, y_rot])
        
        p1_rot = rotate_point(p1)
        p2_rot = rotate_point(p2)
        p3_rot = rotate_point(p3)
        
        # Convert to pixel coordinates
        pts = onp.array([p1_rot, p2_rot, p3_rot]) * onp.array([w, h])
        
        # Fill triangle using barycentric coordinates
        self._fill_triangle(img, pts, color, border)
    
    def _fill_triangle(self, img, pts, color, border=0.0):
        """Vectorized triangle filling - MUCH FASTER than pixel-by-pixel"""
        h, w = img.shape[:2]
        
        # Get triangle vertices
        a = pts[0]
        b = pts[1]
        c = pts[2]
        
        # Compute bounding box
        xmin = max(0, int(min(a[0], b[0], c[0])))
        xmax = min(w, int(max(a[0], b[0], c[0])) + 1)
        ymin = max(0, int(min(a[1], b[1], c[1])))
        ymax = min(h, int(max(a[1], b[1], c[1])) + 1)
        
        # Create meshgrid for vectorized computation
        yy, xx = onp.meshgrid(onp.arange(ymin, ymax), onp.arange(xmin, xmax), indexing='ij')
        
        # Pixel centers
        points = onp.stack([xx + 0.5, yy + 0.5], axis=-1)
        
        # Vectorized barycentric coordinates
        v0 = c - a
        v1 = b - a
        v2 = points - a
        
        dot00 = onp.dot(v0, v0)
        dot01 = onp.dot(v0, v1)
        dot11 = onp.dot(v1, v1)
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-10)
        
        dot02 = onp.sum(v0 * v2, axis=-1)
        dot12 = onp.sum(v1 * v2, axis=-1)
        
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
        # Check if inside triangle
        mask = (u >= -border) & (v >= -border) & (u + v <= 1 + border)
        
        # Apply color to masked pixels
        y_idx, x_idx = onp.where(mask)
        img[ymin + y_idx, xmin + x_idx] = color
    
    def _highlight_img(self, img, color=(255, 255, 255), alpha=0.30):
        """Add highlighting to an image (matches your highlight_img function)"""
        blend = img + alpha * (onp.array(color, dtype=onp.uint8) - img)
        img[:, :, :] = onp.clip(blend, 0, 255).astype(onp.uint8)
    
    def _downsample(self, img, factor):
        """Downsample image for anti-aliasing (matches your downsample function)"""
        if factor == 1:
            return img
        
        h, w = img.shape[0] // factor, img.shape[1] // factor
        img_reshaped = img.reshape(h, factor, w, factor, 3)
        return img_reshaped.mean(axis=(1, 3)).astype(onp.uint8)
    
    def get_tile(self, obj, agent_dir=None, agent_hat=False, highlight=False):
        """Get a pre-rendered tile from cache"""
        key = (obj, agent_dir, agent_hat, highlight, self.tile_size)
        
        tile = self.tile_cache.get(key)
        if tile is None:
            # Fallback: render on-the-fly if not in cache
            print(f"Warning: Tile not in cache: {key}")
            self._render_and_cache_tile(obj, agent_dir, agent_hat, highlight)
            tile = self.tile_cache[key]
        
        return tile
    
    def render_grid(self, state, _agents, get_obs_point_fn):
        """Fast rendering using pre-cached tiles"""
        # Convert JAX arrays to numpy for rendering (CPU operation)
        grid = onp.array(state.grid)
        agent_locs = onp.array(state.agent_locs)
        freeze = onp.array(state.freeze)
        
        # Prepare padded grid (matches your original render logic)
        grid = onp.pad(
            grid, 
            ((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)), 
            constant_values=1  # Items.wall
        )
        
        # Compute highlight mask (for agent observations)
        highlight_mask = onp.zeros_like(grid, dtype=bool)
        for a in range(self.num_agents):
            if freeze[a] > 0:
                continue
            
            # Use your existing get_obs_point function
            startx, starty = get_obs_point_fn(jnp.array(agent_locs[a]))
            startx, starty = int(startx), int(starty)
            
            highlight_mask[startx:startx+self.OBS_SIZE, starty:starty+self.OBS_SIZE] = True
        
        # Create output image
        h, w = grid.shape
        img = onp.zeros((h * self.tile_size, w * self.tile_size, 3), 
                       dtype=onp.uint8)
        
        # OPTIMIZED: Vectorized tile placement instead of nested loops
        # Build lookup arrays for all positions at once
        rows, cols = onp.meshgrid(onp.arange(h), onp.arange(w), indexing='ij')
        
        # Pre-compute agent positions for fast lookup
        agent_positions = {}
        for a in range(self.num_agents):
            # Adjust for padding
            pos = (int(agent_locs[a, 0]) + self.PADDING, 
                   int(agent_locs[a, 1]) + self.PADDING)
            agent_positions[pos] = (int(agent_locs[a, 2]), a)
        
        # Fast tile placement with minimal lookups
        for i in range(h):
            for j in range(w):
                cell = int(grid[i, j])
                if cell == 0:
                    cell = None
                
                # Fast agent lookup
                agent_dir = None
                if (i, j) in agent_positions:
                    agent_dir, _ = agent_positions[(i, j)]
                
                highlight = bool(highlight_mask[i, j])
                
                # Get pre-rendered tile (fast cache lookup)
                tile = self.get_tile(cell, agent_dir, False, highlight)
                
                # Place tile in output image
                y0, y1 = i * self.tile_size, (i + 1) * self.tile_size
                x0, x1 = j * self.tile_size, (j + 1) * self.tile_size
                img[y0:y1, x0:x1] = tile
        
        # Crop to actual game area (matches your original render logic)
        crop_size = self.tile_size * (self.PADDING - 1)
        img = img[crop_size:-crop_size, crop_size:-crop_size]
        
        # Rotate to match your coordinate system (matches your rot90 at the end)
        img = onp.rot90(img, 2)
        
        return img

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


        self.renderer = OptimizedRenderer(
            num_agents=self.num_agents,
            player_colours=self.PLAYER_COLOURS,
            grid_size_row=self.GRID_SIZE_ROW,
            grid_size_col=self.GRID_SIZE_COL,
            obs_size=self.OBS_SIZE,
            padding=self.PADDING,
            tile_size=16,
            subdivs=3
        )

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
                (current_items == Items.empty),
                interact_idx,   # Draw beam on empty
                current_items   # Keep everything else (agents, apples, walls)
            )
            qualified_to_zap = zaps_expanded.squeeze()

            def update_grid_batch(locs, items, qualified, grid):
                # Only write if qualified AND valid index
                in_bounds_r = (locs[:, 0] >= 0) & (locs[:, 0] < self.GRID_SIZE_ROW)
                in_bounds_c = (locs[:, 1] >= 0) & (locs[:, 1] < self.GRID_SIZE_COL)
                should_write = qualified & in_bounds_r & in_bounds_c

                # Safe clip for the write operation index (only executes if should_write is True)
                target_r = jnp.where(should_write, locs[:, 0], self.GRID_SIZE_ROW)
                target_c = jnp.where(should_write, locs[:, 1], self.GRID_SIZE_COL)

                # This ensures any index at GRID_SIZE_ROW is simply ignored.
                return grid.at[target_r, target_c].set(items, mode='drop')

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
            
            # --- SETUP ---
            locs_at_start = state.agent_locs # Capture starting locations for cleanup
            
            # Purgatory Coordinates
            base_purgatory = -self.OBS_SIZE * 2
            p_cols = jnp.int16(base_purgatory - jnp.arange(self.num_agents))
            p_rows = jnp.full((self.num_agents,), base_purgatory, dtype=jnp.int16)
            p_dirs = jnp.zeros((self.num_agents,), dtype=jnp.int16)
            purgatory_locs = jnp.stack([p_rows, p_cols, p_dirs], axis=-1)
            
            key, subkey = jax.random.split(key)

            # ============================================================
            ## PHASE 1: ZAPPING
            # (Identify newly frozen agents and draw beams onto state.grid)
            # ============================================================
            
            # 1. Apply Zapping interaction. _interact clears old beams, calculates, and draws new beams.
            state = _interact(subkey, state, actions)
            freeze_after_zap = state.freeze
            is_frozen_now = freeze_after_zap > 0 

            # 2. Agents frozen by Zap are prevented from moving
            actions = jnp.where(is_frozen_now, Actions.stay, actions)

            # ============================================================
            ## PHASE 2: MOVEMENT & APPLE COLLECTION
            # (Calculate final move locations, including forcing frozen to purgatory)
            # ============================================================
            
            key, subkey = jax.random.split(key)

            # --- Calculate Proposed Locations (Rotation, Forward Movement, Clipping) ---
            rotated_locs = jax.vmap(
                lambda p, a: jnp.int16(p + ROTATIONS[a]) % jnp.array(
                    [self.GRID_SIZE_ROW + 1, self.GRID_SIZE_COL + 1, 4], 
                    dtype=jnp.int16
                )
            )(p=state.agent_locs, a=actions).squeeze()            
            
            is_move_forward = (actions == Actions.move_forward)
            agent_move = is_move_forward
            
            def apply_forward(p, move_bool):
                current_dir = p[2]
                forward_delta = STEP[current_dir]
                return jnp.where(move_bool, p + forward_delta, p)

            proposed_locs = jax.vmap(apply_forward)(rotated_locs, is_move_forward)

            proposed_locs = jax.vmap(
                jnp.clip, in_axes=(0, None, None)
            )(
                proposed_locs,
                jnp.array([0, 0, 0], dtype=jnp.int16),
                jnp.array([self.GRID_SIZE_ROW - 1, self.GRID_SIZE_COL - 1, 3], dtype=jnp.int16),
            ).squeeze()

            # --- Enforce Purgatory for all frozen agents (new and old) ---
            locs_for_collision = jnp.where(
                is_frozen_now[:, None],
                purgatory_locs,
                proposed_locs
            )

            # --- Collision Resolution (Agent-Agent and Agent-Wall/Obstacle) ---
            collision_matrix = check_collision(locs_for_collision)
            collisions = jnp.sum(collision_matrix, axis=-1, dtype=jnp.int8) - 1
            collisions = jnp.minimum(collisions, 1)
            collided_moved = jnp.maximum(collisions - ~agent_move, 0)

            new_locs = jax.lax.cond(
                jnp.max(collided_moved) > 0,
                lambda: fix_collisions(
                    subkey, 
                    collided_moved,
                    collision_matrix,
                    state.agent_locs,
                    locs_for_collision
                ),
                lambda: locs_for_collision
            )
            key, subkey = jax.random.split(key)

            # Wall/Object Collision Check (Reverting agents hitting walls)
            on_grid_mask = (new_locs[:, 0] >= 0) & (new_locs[:, 1] >= 0)
            target_vals = jnp.where(
                on_grid_mask,
                state.grid[new_locs[:, 0], new_locs[:, 1]],
                jnp.int16(Items.wall)
            )
            
            hit_obstacle = (target_vals != Items.empty) & (target_vals != Items.apple)
            condition_3d = jnp.stack([hit_obstacle, hit_obstacle, hit_obstacle], axis=-1)
            move_mask_3d = jnp.stack([agent_move, agent_move, agent_move], axis=-1)

            final_move_locs = jnp.where(
                move_mask_3d & condition_3d & (~is_frozen_now[:, None]),
                state.agent_locs,
                new_locs
            )
            
            # Final Purgatory enforcement
            final_move_locs = jnp.where(
                is_frozen_now[:, None],
                purgatory_locs,
                final_move_locs
            )

            # --- Apple Collection & Inventory Update (Based on final_move_locs) ---
            def apple_matcher(p: jnp.ndarray) -> jnp.ndarray:
                return jnp.logical_and(
                    p[0] >= 0, 
                    state.grid[p[0], p[1]] == Items.apple
                )
            
            apple_matches = jax.vmap(apple_matcher)(p=final_move_locs).reshape(self.num_agents, 1)
            apple_matches = apple_matches & (~is_frozen_now[:, None]) # Only unfrozen can collect

            new_invs = state.agent_invs + apple_matches
            state = state.replace(agent_invs=new_invs)

            # ============================================================
            ## PHASE 3: GRID CLEANUP, RESPAWN, & FINAL DRAW (Clear Old -> Draw New)
            # ============================================================
            
            # 1. Decrement Freeze
            new_freeze = jnp.maximum(freeze_after_zap - 1, 0)
            just_unfrozen = (freeze_after_zap > 0) & (new_freeze == 0)

            # 2. Respawn Calculation
            already_on_grid = (freeze_after_zap == 0)

            def is_spawn_occupied(spawn_loc):
                at_spawn = (final_move_locs[:, 0] == spawn_loc[0]) & \
                           (final_move_locs[:, 1] == spawn_loc[1])
                return jnp.any(at_spawn & already_on_grid)

            spawns_occupied = jax.vmap(is_spawn_occupied)(self.SPAWNS_PLAYERS)
            available_spawns = ~spawns_occupied

            # Get all available spawn indices
            num_spawns = len(self.SPAWNS_PLAYERS)
            spawn_indices = jnp.arange(num_spawns)

            # Shuffle ALL spawn indices
            key, subkey = jax.random.split(key)
            shuffled_all = jax.random.permutation(subkey, num_spawns)

            # Filter: keep only available spawns from shuffled order
            def is_available(idx):
                return available_spawns[idx]

            shuffled_available_mask = jax.vmap(is_available)(shuffled_all)

            # Push unavailable spawns to end using large dummy value
            dummy_val = num_spawns + 100
            shuffled_with_priority = jnp.where(
                shuffled_available_mask,
                shuffled_all,
                dummy_val
            )

            valid_shuffled = jnp.clip(shuffled_with_priority, 0, num_spawns - 1)

            # Assign to unfrozen agents
            unfrozen_indices = jnp.where(
                just_unfrozen, 
                jnp.cumsum(just_unfrozen.astype(jnp.int32)) - 1, 
                0
            )

            assigned_spawn_idx = jnp.where(
                just_unfrozen,
                valid_shuffled[unfrozen_indices],
                0
            )

            selected_spawns = self.SPAWNS_PLAYERS[assigned_spawn_idx]

            key, subkey = jax.random.split(key)
            player_dir = jax.random.randint(subkey, shape=(self.num_agents,), minval=0, maxval=4, dtype=jnp.int16)

            # CRITICAL: Ensure respawn_locs has the same dtype as state.agent_locs (int16)
            respawn_locs = jnp.stack([
                selected_spawns[:, 0].astype(jnp.int16), 
                selected_spawns[:, 1].astype(jnp.int16), 
                player_dir
            ], axis=-1)

            final_locs_post_respawn = jnp.where(
                just_unfrozen[:, None],
                respawn_locs,
                final_move_locs
            )

            # --- 2a. CLEAR STEP: Remove Old Agent Bodies and Collected Apples ---

            # 2a.1. Clear agent starting positions (Remove Old Positions)
            def clear_old_body(grid, loc):
                # Clear only if the agent was actually on the grid
                return jax.lax.cond(
                    loc[0] >= 0, 
                    lambda g: g.at[loc[0], loc[1]].set(jnp.int16(Items.empty)),
                    lambda g: g,
                    grid
                )
            
            # Use scan to clear all agent bodies at their starting locations
            # The grid still contains beams and apples at this point.
            grid_temp, _ = jax.lax.scan(
                lambda grid, loc: (clear_old_body(grid, loc), None),
                state.grid, 
                locs_at_start
            )
                
            # 2a.2. Clear collected apples (Remove apple item, not agent spot)
            def clear_collected_apple(grid, loc, collected):
                return jax.lax.cond(
                    jnp.logical_and(collected, loc[0] >= 0),
                    lambda g: g.at[loc[0], loc[1]].set(jnp.int16(Items.empty)),
                    lambda g: g,
                    grid
                )

            grid_temp, _ = jax.lax.scan(
                lambda grid, inputs: (clear_collected_apple(grid, inputs[0], inputs[1]), None),
                grid_temp,
                (final_move_locs, apple_matches.flatten())
            )
            
            # 2a.3
            grid_clean_static = grid_temp 
            
            state = state.replace(grid=grid_clean_static)

            # --- 2b.  DRAW STEP: Draw New Agent Positions ---
            
            def place_final_agent(grid, loc, agent_id, frozen):
                # Only draw if agent is NOT frozen AND location is on the grid (NOT in purgatory)
                should_draw = jnp.logical_and(frozen == 0, loc[0] >= 0)
                
                return jax.lax.cond(
                    should_draw,
                    lambda g: g.at[loc[0], loc[1]].set(agent_id),
                    lambda g: g,
                    grid
                )

            def final_draw_body(carry_grid, inputs):
                loc, agent_id, frozen = inputs
                updated_grid = place_final_agent(carry_grid, loc, agent_id, frozen)
                return updated_grid, None

            final_grid_with_agents, _ = jax.lax.scan(
                final_draw_body, 
                state.grid, # Use the clean grid (static map, beams, no old agents/apples)
                (final_locs_post_respawn, self._agents, new_freeze) # Use final locs and final freeze status
            )

            state = state.replace(
                agent_locs=final_locs_post_respawn, 
                freeze=new_freeze, 
                grid=final_grid_with_agents
            )

            # ============================================================
            ## PHASE 4: APPLE REGROWTH
            # ============================================================
            
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
            
            prob = jax.random.uniform(subkey, shape=(len(self.SPAWNS_APPLE),))
            new_apple = jax.vmap(regrow_apple)(self.SPAWNS_APPLE, near_apple_nums, prob)
            key, subkey = jax.random.split(key)

            # Check what's currently at each spawn
            current_at_spawn = grid_apple[self.SPAWNS_APPLE[:, 0], self.SPAWNS_APPLE[:, 1]]
            is_occupied = current_at_spawn != Items.empty

            # Only update if not occupied (empty or already an apple)
            final_apple_state = jnp.where(is_occupied, current_at_spawn, new_apple)

            new_apple_grid = grid_apple.at[self.SPAWNS_APPLE[:, 0], self.SPAWNS_APPLE[:, 1]].set(final_apple_state[:])
            state = state.replace(grid=new_apple_grid)

            # ============================================================
            ## PHASE 5: REWARDS, INFO, & TIMESTEP
            # ============================================================
            
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
                    new_smooth_rewards = 0.99 * 0.01* state.smooth_rewards + original_rewards
                    rewards,disadvantageous,advantageous = self.get_inequity_aversion_rewards_immediate(
                        new_smooth_rewards, self.inequity_aversion_target_agents, 
                        state.inner_t, self.inequity_aversion_alpha, self.inequity_aversion_beta
                    )
                    state = state.replace(smooth_rewards=new_smooth_rewards)
                    info = {
                        "original_rewards": original_rewards.squeeze(),
                        "smooth_rewards": state.smooth_rewards.squeeze(),
                        "shaped_rewards": rewards.squeeze(),
                    }
                else:
                    rewards,disadvantageous,advantageous = self.get_inequity_aversion_rewards_immediate(
                        original_rewards, self.inequity_aversion_target_agents, 
                        state.inner_t, self.inequity_aversion_alpha, self.inequity_aversion_beta
                    )
                    info = {
                        "original_rewards": original_rewards.squeeze(),
                        "shaped_rewards": rewards.squeeze(),
                    }
            elif self.svo:
                rewards = jnp.zeros((self.num_agents, 1))
                original_rewards = jnp.where(apple_matches, 1, rewards) * self.num_agents
                rewards, theta = self.get_svo_rewards(
                    original_rewards, self.svo_w, self.svo_ideal_angle_degrees, self.svo_target_agents
                )
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

            inner_t = state_nxt.inner_t
            outer_t = state_nxt.outer_t
            reset_inner = inner_t == num_inner_steps

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
        tile_size: int = 16,
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

    def render(self, state) -> onp.ndarray:
        """Optimized rendering using pre-cached tiles"""
        return self.renderer.render_grid(state, self._agents, self.get_obs_point)
    
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