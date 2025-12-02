"""
Test script for SocialJax environments
Supports manual control or random actions, saves episode as MP4
"""
import sys
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import socialjax
from pathlib import Path
import imageio
import time
from datetime import datetime 

RENDER = True

def visualize_grid_terminal(state, env):
    """
    Visualizes the grid in the terminal with agents and their directions.
    
    Legend:
    - Wall: ██
    - Empty: ..
    - Apple: 🍎
    - Agent: A0-A9 with direction arrow (↑↓←→)
    - Frozen Agent: (A0)-(A9) with direction arrow
    """
    grid = state.grid
    height, width = grid.shape
    
    # Direction symbols (adjusted for top-left origin grid)
    # In this coordinate system: increasing row = moving down, increasing col = moving right
    DIR_SYMBOLS = {
        0: 'v',  # North in code = Down visually (increasing row)
        1: '>',  # East (increasing col) 
        2: '^',  # South in code = Up visually (decreasing row)
        3: '<'   # West (decreasing col)
    }
    
    # Object type symbols (socialjax convention: 0-4 for static items, 5+ for agents)
    # Based on Items enum: empty=0, wall=1, interact=2, apple=3, spawn_point=4
    SYMBOLS = {
        0: ' _ ',  # Empty
        1: '███',  # Wall
        2: ' ⚡',  # Interact (zap beam)
        3: ' A ',  # Apple
        4: ' S '   # Spawn point
    }
    
    print("\n" + "="*60)
    print("Grid Visualization")
    print("="*60)
    
    # Print column numbers
    print("    ", end="")
    for c in range(width):
        print(f" {c:2d} ", end="")
    print()
    print("    " + "----" * width)
    
    # Print each row
    for r in range(height):
        print(f"{r:2d} |", end=" ")
        for c in range(width):
            cell_id = int(grid[r, c])  # Convert JAX array to Python int
            
            # Check if it's an agent (ID >= 5)
            if cell_id >= 5:
                agent_idx = cell_id - 5
                direction = int(state.agent_locs[agent_idx, 2])  # Convert to int
                dir_symbol = DIR_SYMBOLS[direction]
                
                # Check if agent is frozen
                if int(state.freeze[agent_idx]) > 0:
                    # Frozen agent (shouldn't normally be on grid, but show if present)
                    print(f"({agent_idx}{dir_symbol})", end="")
                else:
                    # Active agent
                    if agent_idx < 10:
                        print(f" {agent_idx}{dir_symbol}", end=" ")
                    else:
                        print(f"{agent_idx}{dir_symbol}", end=" ")
            else:
                # Static object
                symbol = SYMBOLS.get(cell_id, '??')
                print(f"{symbol}", end=" ")
        print()
    
    print()
    
    # Print agent info
    print("Agent Information:")
    print("-" * 60)
    for i in range(env.num_agents):
        r, c, d = state.agent_locs[i]
        dir_symbol = DIR_SYMBOLS[int(d)]
        frozen_status = f" [FROZEN: {state.freeze[i]} steps]" if state.freeze[i] > 0 else ""
        print(f"Agent {i}: Position=({r}, {c}), Direction={dir_symbol}{frozen_status}")
    
    print("="*60 + "\n")

def perform_consistency_checks(step, state, env):
    """
    Performs checks on the environment state for internal consistency.
    
    Checks:
    1. Collision: If more than one agent occupies the same (r, c) location.
    2. Grid Representation: If the grid contains the correct agent IDs at the
       locations specified in state.agent_locs.
    """
    print(f"\n--- Consistency Checks (Step {step}) ---")
    
    # Extract only the (r, c) coordinates (first two columns)
    # The agent_locs structure is assumed to be (r, c, d)
    agent_positions = state.agent_locs[:, :2]
    
    # 1. Check for Collisions (More than one agent at the same (r, c))
    # We use np.unique with return_counts to find duplicated (r, c) pairs.
    unique_positions, counts = np.unique(agent_positions, axis=0, return_counts=True)
    
    # Find positions where count > 1 (i.e., collisions)
    collision_indices = np.where(counts > 1)[0]
    
    if collision_indices.size > 0:
        collision_locs = unique_positions[collision_indices]
        print(f"\t\t!!! COLLISION ERROR: Multiple agents occupy the following locations:")
        for loc in collision_locs:
            # Find the indices of agents at this location for better debugging
            agent_indices = np.where(np.all(agent_positions == loc, axis=1))[0]
            print(f"   Location ({loc[0]}, {loc[1]}) occupied by agents: {list(agent_indices)}")
    else:
        print("OK: Collision Check passed. No two agents occupy the same (r, c) location.")

    # 2. Check Grid Representation
    # Agent IDs start from 5 (socialjax convention: 0-4 for static items)
    AGENT_ID_START = 5
    grid_ok = True
    
    for i in range(env.num_agents):
        r, c, d = state.agent_locs[i]
        
        # Check if agent is frozen. Frozen agents should NOT be on the grid.
        if state.freeze[i] > 0:
            if state.grid[r, c] == AGENT_ID_START + i:
                print(f"\t\tWARNING: Agent {i} is frozen but is still present on the grid at ({r}, {c})")
            continue
        
        # Check if the grid cell at (r, c) contains the correct agent ID (5 + index)
        expected_id = AGENT_ID_START + i
        actual_id = state.grid[r, c]
        
        if actual_id != expected_id:
            grid_ok = False
            print(f"\t\t!!! GRID ERROR: Agent {i} is at ({r}, {c}) but grid cell contains {actual_id} (Expected: {expected_id})")

    if grid_ok:
        print("OK: Grid Check passed. All active agents are correctly placed on the grid.")
    print("-" * 40)
    
    return grid_ok

def render_agent_view(env, state, agent_idx):
    """
    Renders the egocentric view of a specific agent using the environment's assets.
    """
    tile_size = 32
    obs_size = env.OBS_SIZE
    padding = env.PADDING
    
    # 1. Prepare Grid (Pad and Crop)
    # We use numpy for CPU-based rendering logic
    full_grid = np.array(state.grid)
    # Pad with Wall (ID=1)
    padded_grid = np.pad(full_grid, ((padding, padding), (padding, padding)), constant_values=1)
    
    # Get agent location
    r, c, d = state.agent_locs[agent_idx]
    center_r, center_c = r + padding, c + padding
    
    # Calculate window (Top-Left corner)
    r_start = center_r - obs_size // 2
    c_start = center_c - obs_size // 2
    
    # Extract local view
    local_view = padded_grid[r_start : r_start + obs_size, c_start : c_start + obs_size]
    
    # 2. Rotate View (Ego-centric)
    # k=d rotates the grid so the agent's "Forward" direction points "Up" in the image
    k = (-int(d) + 2) % 4
    rotated_view = np.rot90(local_view, k=k)
    
    # 3. Render Tiles
    img = np.zeros((obs_size * tile_size, obs_size * tile_size, 3), dtype=np.uint8)
    
    for row in range(obs_size):
        for col in range(obs_size):
            obj_id = rotated_view[row, col]
            
            # Calculate relative direction for other agents visible in the view
            render_dir = None
            # 5 is start of agent IDs (0-4 are static items)
            if obj_id >= 5: 
                target_agent_idx = obj_id - 5
                target_global_dir = state.agent_locs[target_agent_idx, 2].item()
                # Rotate their direction relative to our view
                render_dir = (target_global_dir + k) % 4
            
            # Use the environment's native tile renderer
            tile = env.render_tile(obj_id, agent_dir=render_dir, tile_size=tile_size)
            
            # Place tile on canvas
            y_pos = row * tile_size
            x_pos = col * tile_size
            img[y_pos : y_pos + tile_size, x_pos : x_pos + tile_size] = tile
            
    return img

def get_manual_action(agent_idx, num_actions, action_names=None):
    """Get action from user input"""
    print(f"\nAgent {agent_idx} - Choose action:")
    if action_names:
        for i, name in enumerate(action_names):
            print(f"  {i}: {name}")
    else:
        print(f"  Enter action (0-{num_actions-1})")
    
    while True:
        try:
            action = int(input(f"Agent {agent_idx} action: "))
            if 0 <= action < num_actions:
                return action
            else:
                print(f"Invalid action. Must be between 0 and {num_actions-1}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def run_episode(
    env_name,
    env_kwargs=None,
    mode="random",
    max_steps=1000,
    seed=None,
    save_path="test_episode.mp4",
    fps=5
):
    """
    Run a single episode in the environment
    
    Args:
        env_name: Name of the environment (e.g., 'harvest_common_open')
        env_kwargs: Dict of environment kwargs
        mode: 'manual' or 'random'
        max_steps: Maximum number of steps
        seed: Random seed
        save_path: Path to save the video (supports .mp4, .gif, .avi)
        fps: Frames per second for video
    """
    # Create environment
    env_kwargs = env_kwargs or {}
    env = socialjax.make(env_name, **env_kwargs)
    
    print(f"\n{'='*60}")
    print(f"Testing Environment: {env_name}")
    print(f"Number of agents: {env.num_agents}")
    print(f"Action space: {env.num_actions}")
    print(f"Mode: {mode}")
    print(f"{'='*60}\n")
    
    # Get action names if available
    try:
        from socialjax.environments.common_harvest.harvest_common import Actions
        action_names = [action.name for action in Actions]
    except:
        action_names = None
    
    # Initialize
    rng = jax.random.PRNGKey(int(time.time()) if seed is None else seed)
    rng, reset_rng = jax.random.split(rng)
    
    obs, state = env.reset(reset_rng)
    print("Agent locations:")
    print(state.agent_locs)
    print("Observation shape:", obs.shape)
    
    visualize_grid_terminal(state, env)
    
    perform_consistency_checks(0, state, env)
    
    # Storage for rendering
    if RENDER:
        frames = []
        agent_frames = [[] for _ in range(env.num_agents)]

        frames.append(env.render(state))

    for i in range(env.num_agents):
        agent_key = env.agents[i]
        # Skip rendering if agent is frozen (optional, matches your global render logic)
        if state.freeze[i] > 0:
            is_empty = np.all(obs[agent_key] == 0)
            if not is_empty:
                print(f"❌ ERROR: Agent {i} is frozen but received non-zero observation!")
            # Append black frame or last frame, or just skip. 
            # Here we append a black frame to keep video sync.
            if RENDER:
                black_frame = np.zeros((env.OBS_SIZE*32, env.OBS_SIZE*32, 3), dtype=np.uint8)
                agent_frames[i].append(black_frame)
        else:
            if RENDER:
                a_frame = render_agent_view(env, state, i)
                agent_frames[i].append(a_frame)
    
    # Episode info
    episode_rewards = np.zeros(env.num_agents)
    done = False
    step = 0
    
    print("Starting episode...")
    if mode == "manual":
        print("(Press Ctrl+C to stop early)\n")
    
    try:
        while not done and step < max_steps:
            print(f"\n--- Step {step} ---")
            
            # Get actions
            if mode == "manual":
                # Manual control
                actions = []
                for i in range(env.num_agents):
                    action = get_manual_action(i, env.num_actions, action_names)
                    actions.append(action)
            else:
                # Random actions
                rng, *action_rngs = jax.random.split(rng, env.num_agents + 1)
                probs = jnp.array([0.2, 0.2, 0.5, 0, 0.1])
                actions = [
                    jax.random.choice(action_rngs[i], a=env.num_actions, p=probs).item()
                    for i in range(env.num_agents)
                ]
                print(f"Actions: {actions}")
            
            # Step environment
            rng, step_rng = jax.random.split(rng)
            obs, state, rewards, done_dict, info = env.step(step_rng, state, actions)
            
            visualize_grid_terminal(state, env)
            
            # Update episode info
            episode_rewards += np.array(rewards)
            done = done_dict["__all__"]
            
            # Print step info
            print(f"Rewards: {rewards}")
            if info:
                print(f"Info: {info}")
            
            if not perform_consistency_checks(step + 1, state, env):
                break
            
            # Render
            if RENDER:
                frame = env.render(state)
                frames.append(frame)

            for i in range(env.num_agents):
                agent_key = env.agents[i]
                # Skip rendering if agent is frozen (optional, matches your global render logic)
                if state.freeze[i] > 0:
                    is_empty = np.all(obs[agent_key] == 0)
                    if not is_empty:
                        print(f"❌ ERROR: Agent {i} is frozen but received non-zero observation!")
                    # Append black frame or last frame, or just skip. 
                    # Here we append a black frame to keep video sync.
                    if RENDER:
                        black_frame = np.zeros((env.OBS_SIZE*32, env.OBS_SIZE*32, 3), dtype=np.uint8)
                        agent_frames[i].append(black_frame)
                else:
                    if RENDER:
                        a_frame = render_agent_view(env, state, i)
                        agent_frames[i].append(a_frame)
            
            step += 1
            
    except KeyboardInterrupt:
        print("\n\nEpisode interrupted by user.")
    
    # Episode summary
    print(f"\n{'='*60}")
    print(f"Episode Summary")
    print(f"{'='*60}")
    print(f"Steps: {step}")
    print(f"Episode done: {done}")
    for i in range(env.num_agents):
        print(f"Agent {i} total reward: {episode_rewards[i]:.2f}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"{'='*60}\n")
    
    if RENDER:
        # Save video
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")  
        save_path = f"test_{timestamp}/test_{env_name}.mp4"
        print(f"Saving episode to {save_path}...")
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert frames to numpy arrays
        frames_np = [np.array(frame) for frame in frames]
        
        # Determine format from extension
        extension = save_path.suffix.lower()
        
        # Save as video (MP4, AVI, etc.) using imageio
        # For MP4, use ffmpeg codec
        if extension == '.mp4':
            imageio.mimsave(
                save_path,
                frames_np,
                fps=fps,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p'
            )
        else:
            # For other formats, let imageio choose codec
            imageio.mimsave(save_path, frames_np, fps=fps)
        
        agent_dir = save_path.parent / f"{save_path.stem}_agent_views"
        agent_dir.mkdir(exist_ok=True)

        print(f"Saving agent observations to {agent_dir}...")
        for i in range(env.num_agents):
            if len(agent_frames[i]) > 0:
                agent_vid_path = agent_dir / f"agent_{i}_obs.mp4"
                imageio.mimsave(str(agent_vid_path), agent_frames[i], fps=fps)      

        print(f"✓ Saved episode video with {len(frames)} frames at {fps} FPS")


def main():
    parser = argparse.ArgumentParser(description="Test SocialJax environments")
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Environment name (e.g., 'harvest_common_open')"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["manual", "random"],
        default="random",
        help="Control mode: 'manual' for user input, 'random' for random actions"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of steps per episode"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="test_episode.mp4",
        help="Path to save the episode video (supports .mp4, .gif, .avi)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second for the video"
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=None,
        help="Number of agents (if applicable)"
    )
    parser.add_argument(
        "--num-inner-steps",
        type=int,
        default=None,
        help="Number of inner steps (if applicable)"
    )
    parser.add_argument(
        "--shared_rewards",
        type=bool,
        default=False,
        help="whether to use common rewards"
    )


    args = parser.parse_args()
    
    # Build env_kwargs from arguments
    env_kwargs = {}
    if args.num_agents is not None:
        env_kwargs["num_agents"] = args.num_agents
    if args.num_inner_steps is not None:
        env_kwargs["num_inner_steps"] = args.num_inner_steps
    if args.shared_rewards is not None:
        env_kwargs["shared_rewards"] = args.shared_rewards

    # Run episode
    run_episode(
        env_name=args.env,
        env_kwargs=env_kwargs,
        mode=args.mode,
        max_steps=args.max_steps,
        seed=args.seed,
        save_path=args.save_path,
        fps=args.fps
    )


if __name__ == "__main__":
    main()


# Example usage:
"""
# Random actions (MP4)
python test_env.py --env harvest_common_open --mode random --max-steps 500 --save-path test.mp4

# Manual control (MP4)
python test_env.py --env harvest_common_open --mode manual --max-steps 100 --save-path manual_test.mp4

# With custom FPS
python test_env.py --env harvest_common_open --mode random --fps 10 --save-path fast_test.mp4

# Still works with GIF
python test_env.py --env harvest_common_open --mode random --save-path test.gif

# With custom parameters
python test_env.py --env harvest_common_open --mode random --num-agents 4 --num-inner-steps 200 --save-path videos/test.mp4
"""