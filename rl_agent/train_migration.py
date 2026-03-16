"""
DQN Training Script for Controller Migration Environment

Trains an RL agent to learn optimal controller migration policies.
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# Handle imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_agent.envs.controller_migration_env import ControllerMigrationEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_migration")


def train_migration_dqn(
    total_timesteps: int = 50000,
    learning_rate: float = 1e-3,
    exploration_fraction: float = 0.1,
    batch_size: int = 32,
    buffer_size: int = 50000,
    num_switches: int = 4,
    num_controllers: int = 2,
    model_dir: str = "models/",
    log_dir: str = "logs/",
    use_mock: bool = True,
):
    """
    Train DQN agent on controller migration environment.
    
    Args:
        total_timesteps: Total training steps
        learning_rate: Learning rate for DQN
        exploration_fraction: Epsilon decay fraction
        batch_size: Batch size for training
        buffer_size: Replay buffer size
        num_switches: Number of switches in topology
        num_controllers: Number of controllers
        model_dir: Directory to save trained models
        log_dir: Directory for TensorBoard logs
        use_mock: Use synthetic metrics (no Ryu required)
    """
    
    # Create directories
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("TRAINING CONTROLLER MIGRATION AGENT")
    logger.info("="*70)
    logger.info(f"Config: {num_switches} switches, {num_controllers} controllers")
    logger.info(f"Training for {total_timesteps} timesteps")
    
    # Create environment
    logger.info("\nCreating training environment...")
    base_env = ControllerMigrationEnv(
        num_switches=num_switches,
        num_controllers=num_controllers,
        use_mock=use_mock
    )
    env = DummyVecEnv([lambda: base_env])
    
    logger.info("Initializing DQN agent...")
    
    # Check tensorboard availability
    tensorboard_log = None
    try:
        import tensorboard
        tensorboard_log = log_dir
        logger.info("✓ TensorBoard logging enabled")
    except ImportError:
        logger.warning("⚠ TensorBoard not available - training without TensorBoard logs")
    
    # Create DQN model
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
        target_update_interval=5000,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )
    
    # Create evaluation environment
    logger.info("Setting up evaluation callback...")
    eval_base_env = ControllerMigrationEnv(
        num_switches=num_switches,
        num_controllers=num_controllers,
        use_mock=use_mock
    )
    eval_env = DummyVecEnv([lambda: eval_base_env])
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        eval_freq=5000,
        n_eval_episodes=3,
        deterministic=True,
        verbose=1,
    )
    
    # Train
    logger.info(f"\nStarting training for {total_timesteps} timesteps...")
    logger.info("-"*70)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback],
        progress_bar=True,
    )
    
    # Save final model
    final_model_path = os.path.join(model_dir, "migration_dqn_final")
    model.save(final_model_path)
    logger.info(f"\n✓ Training completed")
    logger.info(f"Final model saved to: {final_model_path}")
    logger.info(f"Best model saved to: {os.path.join(model_dir, 'best_model')}")
    
    env.close()
    eval_env.close()
    
    logger.info("="*70)
    
    return model


def evaluate_trained_model(
    model_path: str,
    num_episodes: int = 5,
    num_switches: int = 4,
    num_controllers: int = 2,
):
    """
    Evaluate a trained migration agent.
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of evaluation episodes
        num_switches: Number of switches
        num_controllers: Number of controllers
    """
    logger.info("\n" + "="*70)
    logger.info(f"EVALUATING MODEL: {model_path}")
    logger.info("="*70)
    
    # Load model
    model = DQN.load(model_path)
    
    # Create environment
    env = ControllerMigrationEnv(
        num_switches=num_switches,
        num_controllers=num_controllers,
        use_mock=True
    )
    
    episodes_data = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_return = 0
        episode_migrations = 0
        step_count = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_return += reward
            step_count += 1
            
            if "old_controller" in info:
                episode_migrations += 1
            
            if terminated or truncated:
                break
        
        episode_data = {
            "episode": episode + 1,
            "return": episode_return,
            "migrations": episode_migrations,
            "steps": step_count,
        }
        episodes_data.append(episode_data)
        
        logger.info(f"Episode {episode+1}: return={episode_return:.4f}, migrations={episode_migrations}, steps={step_count}")
    
    # Print statistics
    returns = [e["return"] for e in episodes_data]
    migrations = [e["migrations"] for e in episodes_data]
    
    logger.info("\n" + "-"*70)
    logger.info("EVALUATION STATISTICS")
    logger.info("-"*70)
    logger.info(f"Mean episode return: {np.mean(returns):.4f} ± {np.std(returns):.4f}")
    logger.info(f"Mean migrations: {np.mean(migrations):.2f} ± {np.std(migrations):.2f}")
    logger.info(f"Best episode return: {np.max(returns):.4f}")
    logger.info(f"Worst episode return: {np.min(returns):.4f}")
    logger.info("="*70)
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train controller migration agent")
    parser.add_argument("--train", action="store_true", help="Train new model")
    parser.add_argument("--evaluate", type=str, default=None, help="Evaluate trained model (path)")
    parser.add_argument("--timesteps", type=int, default=50000, help="Training timesteps")
    parser.add_argument("--switches", type=int, default=4, help="Number of switches")
    parser.add_argument("--controllers", type=int, default=2, help="Number of controllers")
    parser.add_argument("--no-mock", action="store_true", help="Use real Ryu (requires it running)")
    
    args = parser.parse_args()
    
    if args.train:
        model = train_migration_dqn(
            total_timesteps=args.timesteps,
            num_switches=args.switches,
            num_controllers=args.controllers,
            use_mock=not args.no_mock,
        )
    
    if args.evaluate:
        evaluate_trained_model(
            model_path=args.evaluate,
            num_switches=args.switches,
            num_controllers=args.controllers,
        )
    
    if not args.train and not args.evaluate:
        # Default: train and then evaluate
        model = train_migration_dqn(
            total_timesteps=args.timesteps,
            num_switches=args.switches,
            num_controllers=args.controllers,
            use_mock=not args.no_mock,
        )
        evaluate_trained_model(
            model_path="models/best_model.zip",
            num_switches=args.switches,
            num_controllers=args.controllers,
        )
