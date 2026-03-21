"""
Multi-Agent RL System for Large-Scale WAN

Coordinates multiple DQN agents (one per region) to balance control plane load
across a large-scale geographically distributed SDN network.
"""

import sys
import logging
from pathlib import Path
from typing import Dict
import numpy as np

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# Handle imports - add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rl_agent.envs.wan_env import LargeScaleWANEnv
except ImportError:
    # If running from rl_agent directory, try relative import
    from envs.wan_env import LargeScaleWANEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("multi_agent_wan")


class RegionalWANEnvWrapper(gym.Env):
    """
    Wrapper to adapt LargeScaleWANEnv for standard gymnasium interface.
    Binds region_id to environment for single-agent training.
    """
    
    def __init__(self, base_env: 'LargeScaleWANEnv', region_id: int):
        super().__init__()
        self.base_env = base_env
        self.region_id = region_id
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space
        self.metadata = base_env.metadata
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        return self.base_env.reset(seed=seed, options=options)
    
    def step(self, action: int):
        """Step with region_id bound."""
        return self.base_env.step(action, region_id=self.region_id)
    
    def render(self, mode='human'):
        """Render environment."""
        return self.base_env.render()
    
    def close(self):
        """Close environment."""
        pass


class MultiAgentWANCoordinator:
    """
    Coordinates multiple regional DQN agents for WAN load balancing.
    
    Architecture:
    ├─ Region 0 Agent (North America)
    ├─ Region 1 Agent (Europe)
    └─ Region 2 Agent (Asia)
    
    Each agent learns independently but observations are shared globally.
    """
    
    def __init__(
        self,
        num_regions: int = 3,
        model_dir: str = "models/",
    ):
        """
        Initialize multi-agent coordinator.
        
        Args:
            num_regions: Number of geographic regions
            model_dir: Directory to save models
        """
        self.num_regions = num_regions
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create shared environment
        self.env = LargeScaleWANEnv(
            num_regions=num_regions,
            switches_per_region=30,
            controllers_per_region=4,
            use_mock=True
        )
        
        # Create agents (one per region)
        self.agents = {}
        self.agent_models = {}
        
        for region_id in range(num_regions):
            region_name = self.env.regions[region_id]
            
            # Create individual environment for each agent
            base_env = LargeScaleWANEnv(
                num_regions=num_regions,
                switches_per_region=30,
                controllers_per_region=4,
                use_mock=True
            )
            # Wrap to bind region_id
            wrapped_env = RegionalWANEnvWrapper(base_env, region_id)
            env = DummyVecEnv([lambda e=wrapped_env: e])
            
            # Create DQN agent
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=1e-3,
                exploration_fraction=0.1,
                batch_size=32,
                buffer_size=50000,
                verbose=0,
                tensorboard_log=str(Path("logs") / f"region_{region_id}"),
            )
            
            self.agents[region_id] = {
                "name": region_name,
                "env": env,
                "model": model,
                "episodes": 0,
                "total_reward": 0,
                "migrations": 0,
            }
            
            logger.info(f"Created agent for region {region_id} ({region_name})")
    
    def train(
        self,
        total_timesteps_per_agent: int = 50000,
        update_frequency: int = 5000,
    ) -> None:
        """
        Train all agents with coordination.
        
        Args:
            total_timesteps_per_agent: Per-agent training steps
            update_frequency: Frequency of inter-agent coordination
        """
        logger.info("="*70)
        logger.info("TRAINING MULTI-AGENT WAN SYSTEM")
        logger.info("="*70)
        logger.info(f"Regions: {self.num_regions}")
        logger.info(f"Timesteps per agent: {total_timesteps_per_agent}")
        logger.info(f"Coordination frequency: every {update_frequency} steps")
        
        # Train each agent
        for region_id, agent_info in self.agents.items():
            logger.info(f"\n--- Training Region {region_id} ({agent_info['name']}) ---")
            
            agent_info["model"].learn(
                total_timesteps=total_timesteps_per_agent,
                progress_bar=True,
            )
            
            # Save model
            model_path = self.model_dir / f"agent_region_{region_id}"
            agent_info["model"].save(str(model_path))
            logger.info(f"Saved model: {model_path}")
            
            # Periodic coordination
            if region_id % update_frequency == 0:
                self._coordinate_agents()
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETED")
        logger.info("="*70)
    
    def _coordinate_agents(self) -> None:
        """Coordinate between agents (knowledge sharing)."""
        logger.info("Coordinating between agents...")
        
        # Collect metrics from each agent
        metrics = {}
        for region_id, agent_info in self.agents.items():
            env = agent_info["env"]
            obs = env.reset()
            
            # Run agent on current state
            action, _ = agent_info["model"].predict(obs)
            obs, reward, done, info = env.step(action)
            
            metrics[region_id] = {
                "reward": float(reward[0]) if isinstance(reward, np.ndarray) else reward,
                "action": action,
            }
        
        # Log coordination results
        avg_reward = np.mean([m["reward"] for m in metrics.values()])
        logger.info(f"  Average reward across regions: {avg_reward:.4f}")
    
    def evaluate(self, num_episodes: int = 5) -> Dict:
        """
        Evaluate all agents.
        
        Args:
            num_episodes: Number of evaluation episodes per agent
            
        Returns:
            Dictionary of results
        """
        logger.info("\n" + "="*70)
        logger.info("EVALUATING MULTI-AGENT SYSTEM")
        logger.info("="*70)
        
        results = {}
        
        for region_id, agent_info in self.agents.items():
            logger.info(f"\nEvaluating Region {region_id} ({agent_info['name']})...")
            
            episodes_data = []
            
            for episode in range(num_episodes):
                obs = agent_info["env"].reset()
                episode_return = 0
                episode_migrations = 0
                step_count = 0
                
                for step in range(500):
                    action, _ = agent_info["model"].predict(obs, deterministic=True)
                    obs, reward, done, info = agent_info["env"].step(action)
                    
                    episode_return += float(reward[0]) if isinstance(reward, np.ndarray) else reward
                    step_count += 1
                    
                    if done:
                        break
                
                episodes_data.append({
                    "episode": episode + 1,
                    "return": episode_return,
                    "steps": step_count,
                })
                
                logger.info(
                    f"  Episode {episode+1}: return={episode_return:.4f}, steps={step_count}"
                )
            
            returns = [e["return"] for e in episodes_data]
            results[region_id] = {
                "region": agent_info["name"],
                "mean_return": np.mean(returns),
                "std_return": np.std(returns),
                "best_return": np.max(returns),
                "worst_return": np.min(returns),
            }
            
            logger.info(f"  Mean return: {results[region_id]['mean_return']:.4f}")
            logger.info(f"  Std dev: {results[region_id]['std_return']:.4f}")
        
        # Global statistics
        all_returns = [r["mean_return"] for r in results.values()]
        
        logger.info("\n" + "-"*70)
        logger.info("GLOBAL STATISTICS")
        logger.info("-"*70)
        logger.info(f"Average return across all regions: {np.mean(all_returns):.4f}")
        logger.info(f"Best regional performance: {np.max(all_returns):.4f}")
        logger.info(f"Worst regional performance: {np.min(all_returns):.4f}")
        logger.info("="*70)
        
        return results
    
    def deploy(self, num_steps: int = 100) -> None:
        """
        Deploy trained agents on WAN environment.
        
        Args:
            num_steps: Number of deployment steps
        """
        logger.info("\n" + "="*70)
        logger.info("DEPLOYING AGENTS TO WAN")
        logger.info("="*70)
        
        obs = self.env.reset()[0]
        
        total_reward = 0
        total_migrations = 0
        
        for step in range(num_steps):
            # Each region agent makes decision
            actions_by_region = {}
            
            for region_id, agent_info in self.agents.items():
                action, _ = agent_info["model"].predict(obs, deterministic=True)
                actions_by_region[region_id] = action
            
            # Execute first region's action (demonstration)
            action = actions_by_region[0]
            obs, reward, terminated, truncated, info = self.env.step(action, region_id=0)
            
            total_reward += reward
            total_migrations += 1
            
            if (step + 1) % 20 == 0:
                logger.info(
                    f"Step {step+1}/{num_steps}: "
                    f"reward={reward:.4f}, total_reward={total_reward:.4f}"
                )
            
            if terminated:
                break
        
        logger.info(f"\nDeployment complete:")
        logger.info(f"  Total steps: {step+1}")
        logger.info(f"  Total reward: {total_reward:.4f}")
        logger.info(f"  Average reward/step: {total_reward/(step+1):.4f}")
        logger.info("="*70)


def main():
    """Main training and evaluation loop."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Agent RL for Large-Scale WAN Control Plane Balancing"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train agents"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate agents"
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy agents"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Timesteps per agent (default: 50000)"
    )
    parser.add_argument(
        "--regions",
        type=int,
        default=3,
        help="Number of regions (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Create coordinator
    coordinator = MultiAgentWANCoordinator(
        num_regions=args.regions,
        model_dir="models/wan/"
    )
    
    # Train
    if args.train:
        coordinator.train(total_timesteps_per_agent=args.timesteps)
    
    # Evaluate
    if args.evaluate:
        coordinator.evaluate(num_episodes=5)
    
    # Deploy
    if args.deploy:
        coordinator.deploy(num_steps=100)
    
    # Default: train and evaluate
    if not args.train and not args.evaluate and not args.deploy:
        coordinator.train(total_timesteps_per_agent=args.timesteps)
        coordinator.evaluate(num_episodes=5)


if __name__ == "__main__":
    main()
