"""
DQN Training Script using Stable-Baselines3
Trains RL agent on SDN load balancing environment
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Handle imports whether script is run directly or as module
try:
    from rl_agent.envs.sdn_env import SDNEnvironment
except ModuleNotFoundError:
    # If running directly from rl_agent directory, add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from rl_agent.envs.sdn_env import SDNEnvironment


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_dqn")


def train_dqn(
    total_timesteps: int = 100000,
    learning_rate: float = 1e-3,
    exploration_fraction: float = 0.1,
    batch_size: int = 32,
    buffer_size: int = 50000,
    model_dir: str = "models/",
    log_dir: str = "logs/",
    use_mock: bool = True,
    ryu_api_url: str = "http://127.0.0.1:8080",
):
    """
    Train DQN agent on SDN environment.
    
    Args:
        total_timesteps: Total training steps
        learning_rate: Learning rate for DQN
        exploration_fraction: Epsilon decay fraction
        batch_size: Batch size for training
        buffer_size: Replay buffer size
        model_dir: Directory to save trained models
        log_dir: Directory for TensorBoard logs
        use_mock: If True, use synthetic data (no Ryu controller required)
        ryu_api_url: Ryu controller API URL
    """
    
    # Create directories
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating training environment...")
    # Create environment directly with mock mode enabled
    base_env = SDNEnvironment(
        ryu_api_url=ryu_api_url,
        num_ports=4,
        lookback_window=5,
        use_mock=use_mock
    )
    # Wrap with DummyVecEnv for vectorized interface
    env = DummyVecEnv([lambda: base_env])
    
    logger.info("Initializing DQN agent...")
    
    # Check if tensorboard is available for logging
    tensorboard_log = None
    try:
        import tensorboard
        tensorboard_log = log_dir
        logger.info("TensorBoard logging enabled")
    except ImportError:
        logger.warning("TensorBoard not available - training will proceed without TensorBoard logging")
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        batch_size=batch_size,
        buffer_size=buffer_size,
        train_freq=4,
        target_update_interval=10000,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )
    
    # Evaluation environment
    eval_base_env = SDNEnvironment(
        ryu_api_url=ryu_api_url,
        num_ports=4,
        lookback_window=5,
        use_mock=use_mock
    )
    eval_env = DummyVecEnv([lambda: eval_base_env])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback],
        progress_bar=True,
    )
    
    # Save final model
    final_model_path = os.path.join(model_dir, "dqn_final")
    model.save(final_model_path)
    logger.info(f"Training completed. Model saved to {final_model_path}")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    train_dqn(
        total_timesteps=100000,
        learning_rate=1e-3,
        batch_size=32,
    )
