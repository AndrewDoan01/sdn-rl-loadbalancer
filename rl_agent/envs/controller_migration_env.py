"""
Gymnasium Environment for SDN Controller Migration & Load Balancing

Objective: Train RL agent to migrate switches between controllers
to balance load and minimize latency while avoiding cascade failures.

State: CPU/Memory/Latency metrics for each controller
Action: Discrete index representing (switch_id, target_controller_id) migration
Reward: -CPU_variance - latency_penalty + latency_improvement - migration_cost
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional
import requests
import logging
import time

logger = logging.getLogger("ControllerMigrationEnv")


class ControllerMigrationEnv(gym.Env):
    """
    Custom RL environment for SDN controller migration.
    
    Manages which switches are connected to which controllers,
    aiming to balance CPU load and minimize latency.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        ryu_api_url: str = "http://127.0.0.1:8080",
        num_switches: int = 4,
        num_controllers: int = 2,
        use_mock: bool = True,
        ryu_api_timeout: float = 2.0,
    ):
        """
        Initialize Controller Migration Environment.
        
        Args:
            ryu_api_url: URL to Ryu REST API
            num_switches: Number of switches in topology
            num_controllers: Number of controllers
            use_mock: Use synthetic metrics if True
            ryu_api_timeout: Timeout for Ryu API calls
        """
        self.ryu_api_url = ryu_api_url
        self.num_switches = num_switches
        self.num_controllers = num_controllers
        self.use_mock = use_mock
        self.ryu_api_timeout = ryu_api_timeout
        
        # Try to connect to Ryu if not using mock
        self._ryu_available = False
        self._mock_mode_active = use_mock
        if not use_mock:
            self._check_ryu_availability()
        
        # State tracking
        self.current_step = 0
        self.max_steps = 1000
        self.last_cpu_load = np.zeros(num_controllers, dtype=np.float32)
        self.last_memory_load = np.zeros(num_controllers, dtype=np.float32)
        self.last_latency = np.zeros(num_switches, dtype=np.float32)
        
        # Switch-to-controller assignments (controller_id for each switch)
        # Initialize: spread evenly across controllers
        self.switch_assignments = np.array(
            [i % num_controllers for i in range(num_switches)],
            dtype=np.int32
        )
        
        # Migration history for cost calculation
        self.migration_history = []  # List of (step, switch_id, old_controller, new_controller)
        
        # Action space: num_switches * num_controllers possible migrations
        # Action i represents: migrate switch (i // num_controllers) to controller (i % num_controllers)
        self.action_space = spaces.Discrete(num_switches * num_controllers)
        
        # Observation space:
        # [cpu_loads (num_controllers), memory_loads (num_controllers), 
        #  latencies (num_switches), current_assignments (num_switches)]
        obs_dim = num_controllers + num_controllers + num_switches + num_switches
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        logger.info(f"Initialized ControllerMigrationEnv: {num_switches} switches, {num_controllers} controllers")
        if self._mock_mode_active:
            logger.info("✓ Running in MOCK mode (synthetic metrics)")
        else:
            logger.info("✓ Connected to Ryu")
    
    def _check_ryu_availability(self) -> None:
        """Check if Ryu controller is accessible."""
        try:
            response = requests.get(
                f"{self.ryu_api_url}/stats/switches",
                timeout=self.ryu_api_timeout
            )
            if response.status_code == 200:
                self._ryu_available = True
                self._mock_mode_active = False
                logger.info(f"✓ Ryu controller available at {self.ryu_api_url}")
            else:
                logger.warning("Ryu unavailable, using mock mode")
                self._mock_mode_active = True
        except Exception as e:
            logger.warning(f"Cannot connect to Ryu ({e}), using mock mode")
            self._mock_mode_active = True
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Reinitializes switch assignments and clears migration history.
        """
        super().reset(seed=seed)
        self.current_step = 0
        
        # Reset assignments: spread switches evenly
        self.switch_assignments = np.array(
            [i % self.num_controllers for i in range(self.num_switches)],
            dtype=np.int32
        )
        
        # Clear history
        self.migration_history = []
        
        # Get initial observation
        observation = self._get_observation()
        info = {
            "step": self.current_step,
            "switch_assignments": self.switch_assignments.tolist(),
            "migrations": len(self.migration_history)
        }
        
        logger.info("Environment reset. Initial switch assignments:")
        for switch_id, controller_id in enumerate(self.switch_assignments):
            logger.info(f"  Switch {switch_id} → Controller {controller_id}")
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Integer 0 to (num_switches * num_controllers - 1)
                   representing migration of a switch to a target controller
                   
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Decode action: which switch and target controller
        switch_id = action // self.num_controllers
        target_controller = action % self.num_controllers
        
        # Validate action
        if switch_id >= self.num_switches:
            logger.warning(f"Invalid action {action}: switch_id {switch_id} out of range")
            return self._get_observation(), -1.0, False, False, {"error": "invalid_action"}
        
        # Check if already assigned to target controller
        current_controller = self.switch_assignments[switch_id]
        if current_controller == target_controller:
            logger.debug(f"Switch {switch_id} already on controller {target_controller}")
            reward = -0.1  # Small penalty for no-op
            observation = self._get_observation()
            return observation, reward, False, False, {"action": "no_op"}
        
        # Record attempted migration
        old_controller = self.switch_assignments[switch_id]
        logger.info(f"Step {self.current_step}: Migrating switch {switch_id} from controller {old_controller} to {target_controller}")
        
        # Apply migration (update assignment)
        self.switch_assignments[switch_id] = target_controller
        self.migration_history.append((self.current_step, switch_id, old_controller, target_controller))
        
        # In real scenario: notify Ryu to perform migration
        if not self._mock_mode_active:
            self._apply_migration(switch_id, old_controller, target_controller)
        
        # Wait for network to stabilize (reduced for testing, can increase for real deployment)
        time.sleep(0.1)
        
        # Get new observation after migration
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(observation, action, old_controller)
        
        # Check terminal conditions
        terminated = self.current_step >= self.max_steps
        truncated = False  # Can add early stopping conditions
        
        info = {
            "step": self.current_step,
            "switch": switch_id,
            "old_controller": old_controller,
            "new_controller": target_controller,
            "reward": reward,
            "switch_assignments": self.switch_assignments.tolist(),
            "total_migrations": len(self.migration_history)
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Fetch current network state (mock or from Ryu)."""
        if self._mock_mode_active:
            return self._get_mock_observation()
        
        try:
            return self._get_real_observation()
        except Exception as e:
            logger.warning(f"Error fetching real observation: {e}, falling back to mock")
            self._mock_mode_active = True
            return self._get_mock_observation()
    
    def _get_real_observation(self) -> np.ndarray:
        """Fetch metrics from Ryu API."""
        cpu_loads = np.zeros(self.num_controllers, dtype=np.float32)
        memory_loads = np.zeros(self.num_controllers, dtype=np.float32)
        latencies = np.zeros(self.num_switches, dtype=np.float32)
        
        try:
            # Fetch controller metrics from Ryu
            response = requests.get(
                f"{self.ryu_api_url}/stats/controllers",
                timeout=self.ryu_api_timeout
            )
            if response.status_code == 200:
                controller_stats = response.json()
                for i, (ctrl_id, stats) in enumerate(controller_stats.items()):
                    if i < self.num_controllers:
                        cpu_loads[i] = min(stats.get("cpu", 0.5) / 100.0, 1.0)
                        memory_loads[i] = min(stats.get("memory", 0.5) / 100.0, 1.0)
            
            # Fetch latency from port stats
            response = requests.get(
                f"{self.ryu_api_url}/stats/ports",
                timeout=self.ryu_api_timeout
            )
            if response.status_code == 200:
                port_stats = response.json()
                # Estimate latency from packet loss rate (placeholder)
                for i in range(self.num_switches):
                    latencies[i] = np.random.uniform(0.05, 0.3)
        except Exception as e:
            logger.debug(f"Error fetching real metrics: {e}")
            # Fall through to mock metrics
            return self._get_mock_observation()
        
        self.last_cpu_load = cpu_loads
        self.last_memory_load = memory_loads
        self.last_latency = latencies
        
        return self._build_observation_vector(cpu_loads, memory_loads, latencies)
    
    def _get_mock_observation(self) -> np.ndarray:
        """Generate synthetic observation for testing."""
        # Simulate controller loads based on switch assignments
        cpu_loads = np.zeros(self.num_controllers, dtype=np.float32)
        memory_loads = np.zeros(self.num_controllers, dtype=np.float32)
        
        # Count switches per controller
        for switch_id in range(self.num_switches):
            controller_id = self.switch_assignments[switch_id]
            cpu_loads[controller_id] += 0.2  # Each switch adds load
        
        # Normalize to 0-1
        cpu_loads = np.minimum(cpu_loads / (self.num_switches * 0.3), 1.0)
        # Add noise
        cpu_loads += np.random.normal(0, 0.05, self.num_controllers)
        cpu_loads = np.clip(cpu_loads, 0.1, 1.0)
        
        # Memory loads are similar but slightly lower
        memory_loads = cpu_loads * 0.8 + np.random.normal(0, 0.03, self.num_controllers)
        memory_loads = np.clip(memory_loads, 0.05, 0.95)
        
        # Latencies vary by switch
        latencies = np.random.uniform(0.05, 0.4, self.num_switches)
        
        self.last_cpu_load = cpu_loads
        self.last_memory_load = memory_loads
        self.last_latency = latencies
        
        return self._build_observation_vector(cpu_loads, memory_loads, latencies)
    
    def _build_observation_vector(
        self,
        cpu_loads: np.ndarray,
        memory_loads: np.ndarray,
        latencies: np.ndarray
    ) -> np.ndarray:
        """Build observation vector from individual metrics."""
        # Normalize assignments to 0-1 (controller_id / num_controllers)
        assignments_normalized = self.switch_assignments.astype(np.float32) / self.num_controllers
        
        observation = np.concatenate([
            cpu_loads,
            memory_loads,
            latencies,
            assignments_normalized
        ]).astype(np.float32)
        
        return observation
    
    def _calculate_reward(
        self,
        observation: np.ndarray,
        action: int,
        old_controller: int
    ) -> float:
        """
        Calculate reward based on environment state.
        
        Reward = -CPU_variance - latency_penalty - migration_cost + latency_improvement
        """
        # Extract metrics from observation
        dim_per_controller = self.num_controllers
        cpu_loads = observation[:dim_per_controller]
        memory_loads = observation[dim_per_controller:2*dim_per_controller]
        
        # 1. Penalize CPU variance (load imbalance)
        cpu_variance = float(np.var(cpu_loads))
        variance_penalty = -2.0 * cpu_variance
        
        # 2. Penalize high memory loads (with safety check for NaN)
        high_memory = memory_loads[memory_loads > 0.8]
        memory_penalty = -1.0 * (float(np.mean(high_memory)) if len(high_memory) > 0 else 0.0)
        
        # 3. Penalize overloaded controllers (>90%)
        overload_count = float(np.sum(cpu_loads > 0.9))
        overload_penalty = -5.0 * overload_count
        
        # 4. Migration cost (discourage unnecessary migrations)
        migration_cost = -0.5
        
        # 5. Latency improvement (if target controller has lower load)
        new_controller = action % self.num_controllers
        load_improvement = float((cpu_loads[old_controller] - cpu_loads[new_controller]) * 0.5)
        
        # 6. Bonus for successful load balancing
        balance_bonus = 1.0 if cpu_variance < 0.2 else 0.0
        
        total_reward = (
            variance_penalty +
            memory_penalty +
            overload_penalty +
            migration_cost +
            load_improvement +
            balance_bonus
        )
        
        # Ensure reward is valid (no NaN or Inf)
        if np.isnan(total_reward) or np.isinf(total_reward):
            logger.warning(f"Invalid reward calculated: {total_reward}, resetting to -1.0")
            total_reward = -1.0
        
        return float(total_reward)
    
    def _apply_migration(self, switch_id: int, old_controller: int, new_controller: int) -> None:
        """Apply switch migration via Ryu API (placeholder)."""
        try:
            payload = {
                "switch_id": int(switch_id),
                "old_controller": int(old_controller),
                "new_controller": int(new_controller),
                "timestamp": self.current_step
            }
            requests.post(
                f"{self.ryu_api_url}/migration/apply",
                json=payload,
                timeout=self.ryu_api_timeout
            )
            logger.info(f"Migration request sent to Ryu")
        except Exception as e:
            logger.warning(f"Failed to apply migration via Ryu: {e}")
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render environment state (text representation)."""
        render_str = f"\n=== Step {self.current_step} ===\n"
        render_str += f"CPU Loads: {self.last_cpu_load}\n"
        render_str += f"Memory Loads: {self.last_memory_load}\n"
        render_str += f"Assignments: {self.switch_assignments}\n"
        render_str += f"Migrations: {len(self.migration_history)}\n"
        
        if mode == "human":
            print(render_str)
        
        return render_str
    
    def get_info(self) -> Dict[str, Any]:
        """Get detailed environment information."""
        return {
            "step": self.current_step,
            "num_switches": self.num_switches,
            "num_controllers": self.num_controllers,
            "switch_assignments": self.switch_assignments.tolist(),
            "cpu_loads": self.last_cpu_load.tolist(),
            "memory_loads": self.last_memory_load.tolist(),
            "latencies": self.last_latency.tolist(),
            "total_migrations": len(self.migration_history),
            "migration_history": self.migration_history,
            "mock_mode": self._mock_mode_active
        }
