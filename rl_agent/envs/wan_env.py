"""
Large-Scale WAN Environment for Multi-Agent RL

Models a realistic geo-distributed WAN with:
- Multiple regions (North America, Europe, Asia)
- Regional controllers and switches
- Inter-region latency
- Hierarchical agent coordination
- Shared control plane metrics
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import logging

logger = logging.getLogger("WAN_Env")


class LargeScaleWANEnv(gym.Env):
    """
    Large-scale WAN environment with multiple regions and controllers.
    
    Simulates a continental WAN with:
    - 3 geographic regions (NA, EU, ASIA)
    - 4-6 controllers per region
    - 30-40 switches per region
    - Inter-region latency (typical: 50-150ms)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        num_regions: int = 3,
        switches_per_region: int = 30,
        controllers_per_region: int = 4,
        use_mock: bool = True,
        inter_region_latency: Dict[str, float] = None,
    ):
        """
        Initialize large-scale WAN environment.
        
        Args:
            num_regions: Number of geographic regions
            switches_per_region: Switches per region
            controllers_per_region: Controllers per region
            use_mock: Use synthetic metrics
            inter_region_latency: Latency between regions (ms)
        """
        self.num_regions = num_regions
        self.switches_per_region = switches_per_region
        self.controllers_per_region = controllers_per_region
        self.total_switches = num_regions * switches_per_region
        self.total_controllers = num_regions * controllers_per_region
        self.use_mock = use_mock
        
        # Define regions
        self.regions = ["North_America", "Europe", "Asia"][:num_regions]
        self.region_ids = {name: i for i, name in enumerate(self.regions)}
        
        # Inter-region latency (milliseconds)
        if inter_region_latency is None:
            inter_region_latency = {
                ("North_America", "Europe"): 80,
                ("North_America", "Asia"): 150,
                ("Europe", "Asia"): 100,
            }
        self.inter_region_latency = inter_region_latency
        
        # Initialize regional states
        self.regional_states = {}
        self.switch_assignments = {}  # switch_id → controller_id
        self.controller_loads = {}     # controller_id → CPU load
        self.regional_latencies = {}   # region → avg latency to Ryu
        
        for region_idx, region in enumerate(self.regions):
            self.regional_states[region] = {
                "switches": list(range(
                    region_idx * switches_per_region,
                    (region_idx + 1) * switches_per_region
                )),
                "controllers": list(range(
                    region_idx * controllers_per_region,
                    (region_idx + 1) * controllers_per_region
                )),
                "cpu_loads": np.zeros(controllers_per_region, dtype=np.float32),
                "memory_loads": np.zeros(controllers_per_region, dtype=np.float32),
            }
            
            # Initialize switch assignments (spread evenly)
            for switch_id in self.regional_states[region]["switches"]:
                controller_idx = (switch_id % controllers_per_region)
                self.switch_assignments[switch_id] = (
                    region_idx * controllers_per_region + controller_idx
                )
            
            self.regional_latencies[region] = np.random.uniform(5, 20)  # ms
        
        # Action space: each region acts independently
        # Regional action: (switch_id, target_controller_id) migration
        self.action_space = spaces.Discrete(
            switches_per_region * controllers_per_region
        )
        
        # Observation space: global state visible to all agents
        # [regional_cpu_loads, regional_memory_loads, regional_latencies, assignments_sample]
        obs_dim = (
            num_regions * controllers_per_region +  # CPU per regional controller
            num_regions * controllers_per_region +  # Memory per regional controller
            num_regions +                           # Avg latency per region
            20  # Sample of switch assignments
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 500
        self.migration_history = []
        self.episode_rewards = []
        
        logger.info(
            f"Initialized WAN with {num_regions} regions, "
            f"{self.total_switches} switches, {self.total_controllers} controllers"
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.migration_history = []
        self.episode_rewards = []
        
        # Reset switch assignments evenly
        for region_idx, region in enumerate(self.regions):
            for local_idx, switch_id in enumerate(
                self.regional_states[region]["switches"]
            ):
                controller_idx = (local_idx % self.controllers_per_region)
                self.switch_assignments[switch_id] = (
                    region_idx * self.controllers_per_region + controller_idx
                )
        
        observation = self._get_observation()
        info = {
            "step": 0,
            "region_assignments": self._get_regional_assignments(),
        }
        
        logger.info("WAN environment reset")
        return observation, info
    
    def step(self, action: int, region_id: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step for a region.
        
        Args:
            action: Discrete action from regional policy
            region_id: Which region is acting
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Decode action
        region = self.regions[region_id]
        region_switches = self.regional_states[region]["switches"]
        region_controllers = self.regional_states[region]["controllers"]
        
        local_switch_idx = action // self.controllers_per_region
        local_controller_idx = action % self.controllers_per_region
        
        if local_switch_idx >= len(region_switches):
            return self._get_observation(), -1.0, False, False, {"error": "invalid"}
        
        switch_id = region_switches[local_switch_idx]
        global_controller_id = region_controllers[local_controller_idx]
        
        # Apply migration
        old_controller = self.switch_assignments[switch_id]
        self.switch_assignments[switch_id] = global_controller_id
        
        self.migration_history.append({
            "step": self.current_step,
            "region": region,
            "switch": switch_id,
            "old_controller": old_controller,
            "new_controller": global_controller_id,
        })
        
        # Update loads (simulate)
        self._update_controller_loads()
        
        # Calculate reward
        reward = self._calculate_reward(region_id)
        self.episode_rewards.append(reward)
        
        observation = self._get_observation()
        terminated = self.current_step >= self.max_steps
        
        info = {
            "step": self.current_step,
            "region": region,
            "reward": reward,
            "total_migrations": len(self.migration_history),
        }
        
        return observation, reward, terminated, False, info
    
    def _get_observation(self) -> np.ndarray:
        """Get global observation (visible to all agents)."""
        obs = []
        
        # CPU loads per region
        for region in self.regions:
            region_idx = self.region_ids[region]
            cpu_loads = self.regional_states[region]["cpu_loads"]
            obs.extend(cpu_loads)
        
        # Memory loads per region
        for region in self.regions:
            memory_loads = self.regional_states[region]["memory_loads"]
            obs.extend(memory_loads)
        
        # Latencies per region
        for region in self.regions:
            obs.append(self.regional_latencies[region] / 100.0)  # Normalize
        
        # Sample of assignments
        for i in range(20):
            switch_id = i % self.total_switches
            controller_id = self.switch_assignments[switch_id]
            obs.append(controller_id / self.total_controllers)
        
        return np.array(obs, dtype=np.float32)
    
    def _update_controller_loads(self) -> None:
        """Update controller loads based on switch assignments."""
        for region in self.regions:
            self.regional_states[region]["cpu_loads"] = np.zeros(
                self.controllers_per_region, dtype=np.float32
            )
        
        # Count switches per controller
        for switch_id, controller_id in self.switch_assignments.items():
            region_idx = controller_id // self.controllers_per_region
            local_controller_idx = controller_id % self.controllers_per_region
            
            region = self.regions[region_idx]
            self.regional_states[region]["cpu_loads"][local_controller_idx] += 0.25
        
        # Normalize and add noise
        for region in self.regions:
            cpu = self.regional_states[region]["cpu_loads"]
            cpu = np.minimum(cpu / (self.switches_per_region * 0.3), 1.0)
            cpu += np.random.normal(0, 0.03, len(cpu))
            self.regional_states[region]["cpu_loads"] = np.clip(cpu, 0.1, 1.0)
    
    def _calculate_reward(self, region_id: int) -> float:
        """Calculate reward for regional agent."""
        region = self.regions[region_id]
        cpu_loads = self.regional_states[region]["cpu_loads"]
        
        # Reward: minimize variance (balance)
        variance = float(np.var(cpu_loads))
        overload_count = float(np.sum(cpu_loads > 0.9))
        
        reward = (
            -2.0 * variance +
            -5.0 * overload_count +
            -0.1  # Small migration cost
        )
        
        # Bonus for good balance
        if variance < 0.2:
            reward += 0.5
        
        return max(reward, -2.0)  # Clip to avoid very negative rewards
    
    def _get_regional_assignments(self) -> Dict[str, List[int]]:
        """Get switch-to-controller assignments per region."""
        assignments = {}
        for region in self.regions:
            assignments[region] = [
                self.switch_assignments[s]
                for s in self.regional_states[region]["switches"]
            ]
        return assignments
    
    def get_info(self) -> Dict[str, Any]:
        """Get detailed environment information."""
        return {
            "step": self.current_step,
            "total_switches": self.total_switches,
            "total_controllers": self.total_controllers,
            "regions": self.regions,
            "switch_assignments": self.switch_assignments,
            "total_migrations": len(self.migration_history),
            "episode_rewards": self.episode_rewards,
            "avg_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
        }
