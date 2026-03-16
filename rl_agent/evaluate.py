"""
Evaluation Script for Trained RL Agent
Compares against baseline load balancing policies
"""

import logging
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# Handle imports whether script is run directly or as module
try:
    from rl_agent.envs.sdn_env import SDNEnvironment
    from utils.visualizer import plot_comparison
except ModuleNotFoundError:
    # If running directly from rl_agent directory, add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from rl_agent.envs.sdn_env import SDNEnvironment
    from utils.visualizer import plot_comparison


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("evaluate")


def evaluate_agent(
    model_path: str,
    n_episodes: int = 10,
    deterministic: bool = True,
    use_mock: bool = True,
    ryu_api_url: str = "http://127.0.0.1:8080",
) -> Dict[str, float]:
    """
    Evaluate trained DQN agent.
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic policy
        use_mock: If True, use synthetic data (no Ryu controller required)
        ryu_api_url: Ryu controller API URL
        
    Returns:
        Dictionary with evaluation metrics
    """
    
    logger.info(f"Loading model from {model_path}")
    model = DQN.load(model_path)
    
    # Create environment directly with mock mode
    base_env = SDNEnvironment(
        ryu_api_url=ryu_api_url,
        num_ports=4,
        lookback_window=5,
        use_mock=use_mock
    )
    env = DummyVecEnv([lambda: base_env])
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            # Handle vectorized environment returns
            total_reward += float(reward[0]) if isinstance(reward, np.ndarray) else reward
            done = done[0] if isinstance(done, np.ndarray) else done
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        logger.info(f"Episode {episode+1}: reward={total_reward:.2f}, length={steps}")
    
    env.close()
    
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "max_reward": np.max(episode_rewards),
        "min_reward": np.min(episode_rewards),
    }
    
    return metrics


def evaluate_baseline_policies(
    env_id: str = "SDNEnv-v0",
    n_episodes: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate baseline load balancing policies.
    
    Baselines:
    - Random: Random port selection
    - Round-robin: Cycle through ports
    - Least-loaded: Always select least utilized port
    """
    
    env = make_vec_env(env_id, n_envs=1)
    
    baselines = {
        "random": [],
        "round_robin": [],
        "least_loaded": [],
    }
    
    for baseline_name in baselines:
        logger.info(f"Evaluating {baseline_name} baseline...")
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            step_count = 0
            rr_counter = 0
            
            while not done:
                # Select action based on policy
                if baseline_name == "random":
                    action = env.action_space.sample()
                elif baseline_name == "round_robin":
                    action = rr_counter % env.action_space.n
                    rr_counter += 1
                elif baseline_name == "least_loaded":
                    # Assume first few obs elements are link utils
                    action = np.argmin(obs[0, :env.action_space.n])
                
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                step_count += 1
            
            baselines[baseline_name].append(total_reward)
    
    env.close()
    
    baseline_metrics = {}
    for baseline_name, rewards in baselines.items():
        baseline_metrics[baseline_name] = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
        }
    
    return baseline_metrics


def compare_all(model_path: str, output_dir: str = "data/"):
    """Compare trained agent against baselines."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Evaluating trained agent...")
    agent_metrics = evaluate_agent(model_path, n_episodes=10)
    
    logger.info("Evaluating baseline policies...")
    baseline_metrics = evaluate_baseline_policies(n_episodes=10)
    
    # Log results
    logger.info("\n=== Agent Metrics ===")
    for key, val in agent_metrics.items():
        logger.info(f"{key}: {val:.4f}")
    
    logger.info("\n=== Baseline Metrics ===")
    for baseline_name, metrics in baseline_metrics.items():
        logger.info(f"{baseline_name}:")
        for key, val in metrics.items():
            logger.info(f"  {key}: {val:.4f}")
    
    # Plot comparison
    plot_comparison(agent_metrics, baseline_metrics, output_dir=output_dir)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/best_model"
    
    compare_all(model_path)