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
    
    # Storage for rendering
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
            black_frame = np.zeros((env.OBS_SIZE*32, env.OBS_SIZE*32, 3), dtype=np.uint8)
            agent_frames[i].append(black_frame)
        else:
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
                actions = [
                    jax.random.randint(action_rngs[i], (), 0, env.num_actions).item()
                    for i in range(env.num_agents)
                ]
                print(f"Actions: {actions}")
            
            # Step environment
            rng, step_rng = jax.random.split(rng)
            obs, state, rewards, done_dict, info = env.step(step_rng, state, actions)
            
            # Update episode info
            episode_rewards += np.array(rewards)
            done = done_dict["__all__"]
            
            # Print step info
            print(f"Rewards: {rewards}")
            if info:
                print(f"Info: {info}")
            
            # Render
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
                    black_frame = np.zeros((env.OBS_SIZE*32, env.OBS_SIZE*32, 3), dtype=np.uint8)
                    agent_frames[i].append(black_frame)
                else:
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