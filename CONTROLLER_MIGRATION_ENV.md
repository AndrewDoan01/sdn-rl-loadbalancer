# Controller Migration Environment - Documentation

## Overview

This is a custom **Gymnasium environment** for training RL agents to optimize SDN controller assignment and load balancing. Instead of routing packets (traditional load balancing), the agent learns to migrate switches between controllers to balance computational load and minimize latency.

## Problem Statement

In multi-controller SDN deployments:
- Each switch needs to be assigned to a master controller
- Controllers have limited CPU/memory capacity
- Overloaded controllers become bottlenecks
- Latency varies based on switch-controller distance
- Unnecessary migrations waste network resources

**Goal**: Train an RL agent to dynamically reassign switches to controllers in response to changing network conditions.

## Architecture

### State Space

Observation vector (12 elements for 2 controllers, 4 switches):

```
[cpu_load_0, cpu_load_1,                    # CPU utilization per controller
 memory_load_0, memory_load_1,              # Memory per controller
 latency_0, latency_1, latency_2, latency_3, # Switch latencies
 switch_0_ctrl, switch_1_ctrl, switch_2_ctrl, switch_3_ctrl] # Current assignments (normalized 0-1)
```

Each metric is normalized to [0, 1]:
- CPU/Memory loads: 0.0 (idle) to 1.0 (fully loaded)
- Latencies: Estimates based on network conditions
- Assignments: normalized controller IDs

### Action Space

**Discrete(num_switches × num_controllers)**

Each action represents a migration:
- Action `i` = migrate switch `(i // num_controllers)` to controller `(i % num_controllers)`
- Example with 4 switches, 2 controllers (8 total actions):
  - Action 0: migrate switch 0 to controller 0
  - Action 1: migrate switch 0 to controller 1
  - Action 2: migrate switch 1 to controller 0
  - Action 3: migrate switch 1 to controller 1
  - ... etc

### Reward Function

Designed to encourage good load balancing while penalizing unnecessary migrations:

```python
reward = -2.0 * variance(cpu_loads)        # Penalize imbalance
         - 1.0 * mean(high_memory)         # Penalize overload
         - 5.0 * count(overloaded > 90%)   # Strong penalty for failures
         - 0.5                             # Migration cost
         + 0.5 * (old_load - new_load)     # Bonus for improving switch's controller
         + 1.0 if variance < 0.2           # Bonus for well-balanced system
```

**Key design choices:**
- CPU variance is key metric for load balance
- Overload (>90%) has 5x penalty to avoid cascade failures
- Each migration costs -0.5 to encourage consolidation
- Balancing bonus incentivizes reaching stable state

## Environment Files

### 1. `rl_agent/envs/controller_migration_env.py`

Main environment implementation:

```python
class ControllerMigrationEnv(gym.Env):
    def __init__(
        self,
        ryu_api_url: str = "http://127.0.0.1:8080",
        num_switches: int = 4,
        num_controllers: int = 2,
        use_mock: bool = True,
    )
```

**Key methods:**

- `reset()`: Initialize switches evenly across controllers
- `step(action)`: Execute migration, return new state & reward
- `_get_observation()`: Fetch metrics (mock or from Ryu)
- `_calculate_reward()`: Compute reward from current state
- `render()`: Print current environment state
- `get_info()`: Return detailed environment information

**Mock mode features:**
- Simulates controller loads based on switch assignments
- Each switch adds 0.2 to controller's CPU load
- Realistic noise in metrics
- Works offline without Ryu/Mininet
- Gracefully falls back to mock if Ryu unavailable

### 2. `rl_agent/train_migration.py`

Training script with evaluation:

```bash
# Train with default settings
python3 rl_agent/train_migration.py --train

# Train with custom parameters
python3 rl_agent/train_migration.py --train --timesteps 100000 --switches 6 --controllers 3

# Evaluate trained model
python3 rl_agent/train_migration.py --evaluate models/best_model.zip

# Train then evaluate
python3 rl_agent/train_migration.py --train --evaluate models/best_model.zip
```

**Features:**
- Configurable number of switches/controllers
- DQN with MLP policy
- EvalCallback for model checkpointing
- TensorBoard logging
- Deterministic evaluation mode

### 3. `quick_test_migration.py`

Quick validation test (no Ryu required):

```bash
python3 quick_test_migration.py
```

**Tests:**
- Space validity (action/observation dimensions)
- Reset functionality
- Random actions produce valid observations
- Reward values are reasonable (no NaN/Inf)
- Full episode execution
- Environment info consistency

### 4. `test_migration_env.py`

Comprehensive test suite:

```bash
# Full test (slower, more thorough)
timeout 60 python3 test_migration_env.py
```

**Tests include:**
- Space validity
- Reset operation
- Random actions
- State transitions
- Multiple episodes
- Observation bounds
- Reward distribution statistics

## Usage Examples

### Basic Training

```python
from rl_agent.envs.controller_migration_env import ControllerMigrationEnv
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# Create environment (mock mode - no Ryu needed)
base_env = ControllerMigrationEnv(
    num_switches=4,
    num_controllers=2,
    use_mock=True
)
env = DummyVecEnv([lambda: base_env])

# Create and train DQN
model = DQN("MlpPolicy", env, learning_rate=1e-3)
model.learn(total_timesteps=50000)

# Save model
model.save("migration_agent.zip")

# Load and use
model = DQN.load("migration_agent.zip")
obs = env.reset()
action, _ = model.predict(obs)
```

### Standalone Environment Test

```python
from rl_agent.envs.controller_migration_env import ControllerMigrationEnv

env = ControllerMigrationEnv(num_switches=4, num_controllers=2)

obs, info = env.reset()
print(f"Initial state: {env.switch_assignments}")
print(f"CPU loads: {env.last_cpu_load}")

for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: action={action}, reward={reward:.4f}")
    
    if terminated:
        break

env.render()
info = env.get_info()
print(f"Total migrations: {info['total_migrations']}")
```

## Metrics and Monitoring

### Training Metrics (TensorBoard)

Enable TensorBoard monitoring:
```bash
tensorboard --logdir logs/
# Open http://localhost:6006
```

Tracks:
- Episode reward per iteration
- Q-network loss
- Exploration epsilon decay
- Evaluation performance

### Environment Statistics

Access via `get_info()`:
```python
{
    "step": 5000,
    "num_switches": 4,
    "num_controllers": 2,
    "switch_assignments": [0, 1, 0, 1],
    "cpu_loads": [0.3, 0.35],
    "memory_loads": [0.25, 0.3],
    "latencies": [0.1, 0.15, 0.12, 0.18],
    "total_migrations": 42,
    "migration_history": [(step, switch, old_ctrl, new_ctrl), ...]
}
```

## Performance Results

### Quick Test (10 steps):
```
✓ Reward stats: mean=0.1061, std=0.2588
✓ Reward range: [-0.1000, 0.5059]
✓ No NaN/Inf values
✓ Observation in bounds [0.0, 1.0]
```

### Training (5000 timesteps):
```
- Training throughput: ~120 steps/sec (CPU)
- Typical episode return: -2.0 to +5.0
- Migrations per episode: 10-15
- Convergence: Usually within 5000-10000 steps
```

## Customization Guide

### Modify Reward Function

Edit `_calculate_reward()` in `controller_migration_env.py`:

```python
def _calculate_reward(self, observation, action, old_controller):
    # Extract your metrics
    cpu_loads = observation[:self.num_controllers]
    
    # Custom reward formula
    reward = (
        -1.0 * np.var(cpu_loads) +  # Penalize imbalance
        0.1 * migration_benefit +     # Custom improvement metric
        -0.1 * migration_cost         # Custom migration cost
    )
    
    return reward
```

### Scale to Larger Networks

```python
env = ControllerMigrationEnv(
    num_switches=16,      # Larger topology
    num_controllers=4,    # More controllers
    use_mock=True
)
# Action space becomes 16 * 4 = 64 actions
```

### Connect to Real Ryu Controller

```python
env = ControllerMigrationEnv(
    ryu_api_url="http://192.168.1.100:8080",  # Real controller IP
    num_switches=4,
    num_controllers=2,
    use_mock=False  # Use real metrics
)
```

## Limitations & Future Improvements

**Current limitations:**
- Simplified reward function (can be made more sophisticated)
- Synthetic metrics in mock mode (realistic but not real traffic)
- No migration delay/cost modeling
- Single evaluation per step (could batch)
- No network constraints (bandwidth, latency SLAs)

**Future enhancements:**
- Multi-objective reward (balance + latency + throughput)
- Network flow simulation
- Migration time/bandwith cost
- Hierarchical controller assignment
- Online learning with experience replay
- Policy distillation for inference speed

## Testing Checklist

Before using in production:

- [ ] Run `quick_test_migration.py` - all tests pass
- [ ] Train for 10K steps - model improves
- [ ] Evaluate on 10 episodes - reward convergence
- [ ] Check TensorBoard logs - no NaN spikes
- [ ] Test with different num_switches/num_controllers
- [ ] Verify reward function design matches objectives
- [ ] Validate observation normalization (all [0,1])

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
- [Ryu Framework](https://ryu.readthedocs.io/)
