"""
Custom Gymnasium Environment for SDN Load Balancing
Interfaces with Ryu controller and Mininet topology
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any
import requests
import logging
from requests.exceptions import ConnectionError, Timeout

logger = logging.getLogger("SDNEnv")


class SDNEnvironment(gym.Env):
    """
    Custom RL environment for SDN load balancing.
    
    State: Network metrics (link utilization, latency, packet loss)
    Action: Routing decisions (select output port for flows)
    Reward: Negative latency + balance metric
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        ryu_api_url: str = "http://127.0.0.1:8080",
        num_ports: int = 4,
        lookback_window: int = 5,
        use_mock: bool = True
    ):
        """
        Initialize SDN Environment.
        
        Args:
            ryu_api_url: URL to Ryu REST API
            num_ports: Number of output ports to choose from
            lookback_window: History window for state observations
            use_mock: If True, use synthetic data when Ryu unavailable (default: True)
        """
        self.ryu_api_url = ryu_api_url
        self.num_ports = num_ports
        self.lookback_window = lookback_window
        self.use_mock = use_mock
        self._ryu_available = False
        self._mock_mode_active = False
        
        # Check Ryu availability on init
        if not self.use_mock:
            self._check_ryu_availability()
        else:
            self._mock_mode_active = True
            logger.info("SDN Environment initialized in MOCK mode (Ryu integration disabled)")
        
        # Action space: Select one port (0 to num_ports-1)
        self.action_space = spaces.Discrete(num_ports)
        
        # Observation space: [link_util_1, ..., link_util_n, latency, packet_loss]
        obs_dim = num_ports + 2  # All link utils + latency + packet loss
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 1000
        
    def _check_ryu_availability(self) -> None:
        """Check if Ryu controller is accessible."""
        try:
            response = requests.get(f"{self.ryu_api_url}/stats/switches", timeout=2)
            if response.status_code == 200:
                self._ryu_available = True
                logger.info(f"✓ Ryu controller available at {self.ryu_api_url}")
            else:
                logger.warning(f"Ryu returned status {response.status_code}, using mock mode")
                self._mock_mode_active = True
        except (ConnectionError, Timeout, Exception):
            logger.warning(f"Ryu unavailable at {self.ryu_api_url}, using mock mode with synthetic data")
            self._mock_mode_active = True
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        
        observation = self._get_observation()
        info = {}
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Port selection (0 to num_ports-1)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Apply routing action to network
        self._apply_action(action)
        
        # Get new state
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(observation, action)
        
        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            "step": self.current_step,
            "action": action,
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Fetch current network observation (mock or from Ryu)."""
        if self._mock_mode_active:
            return self._get_mock_observation()
        
        try:
            # Example: GET network stats from Ryu
            response = requests.get(
                f"{self.ryu_api_url}/stats/flowentry/all/table0",
                timeout=2
            )
            metrics = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            
            # Parse response and fill metrics
            # This is a placeholder - actual implementation depends on Ryu API format
            return metrics
        except (ConnectionError, Timeout) as e:
            logger.warning(f"Ryu connection lost, switching to mock mode: {e}")
            self._mock_mode_active = True
            return self._get_mock_observation()
        except Exception as e:
            logger.error(f"Error fetching observation: {e}")
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def _get_mock_observation(self) -> np.ndarray:
        """Generate synthetic observation for testing/training without Ryu."""
        metrics = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Synthetic link utilization (0.3-0.8)
        for i in range(self.num_ports):
            metrics[i] = np.random.uniform(0.3, 0.8)
        
        # Synthetic latency (0.1-0.4, normalized to 0-1)
        metrics[-2] = np.random.uniform(0.1, 0.4)
        
        # Synthetic packet loss (0.0-0.1, normalized to 0-1)
        metrics[-1] = np.random.uniform(0.0, 0.1)
        
        return metrics
    
    def _apply_action(self, action: int) -> None:
        """Apply routing action on network (or mock if unavailable)."""
        if self._mock_mode_active:
            return  # No-op in mock mode
        
        try:
            # POST routing command to Ryu
            payload = {
                "action": int(action),
                "timestamp": self.current_step
            }
            requests.post(
                f"{self.ryu_api_url}/routing/apply",
                json=payload,
                timeout=1
            )
        except (ConnectionError, Timeout) as e:
            logger.debug(f"Ryu connection failed, switching to mock mode: {e}")
            self._mock_mode_active = True
        except Exception as e:
            logger.debug(f"Error applying action: {e}")
    
    def _calculate_reward(self, observation: np.ndarray, action: int) -> float:
        """
        Calculate reward based on:
        - Link utilization balance
        - Latency
        - Packet loss
        """
        link_utils = observation[:-2]
        latency = observation[-2]
        packet_loss = observation[-1]
        
        # Penalize high latency and packet loss
        reward = -latency - (2 * packet_loss)
        
        # Bonus for balanced link utilization
        balance_penalty = np.std(link_utils) * 0.5
        reward -= balance_penalty
        
        return float(reward)
    
    def render(self) -> None:
        """Render environment (logging)."""
        logger.info(f"Step {self.current_step}")
    
    def close(self) -> None:
        """Clean up resources."""
        pass
