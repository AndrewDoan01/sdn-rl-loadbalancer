# SDN Reinforcement Learning Load Balancer

A software-defined network (SDN) load balancing system using deep reinforcement learning (DQN) with Ryu controller and Mininet network emulation.

## Project Overview

This project implements an intelligent SDN load balancer that uses **Deep Q-Network (DQN)** training to optimize traffic routing decisions in a network. Instead of using static load balancing policies, the system learns dynamic routing strategies through interaction with a Mininet-emulated network controlled by Ryu.

### Key Components

- **Ryu SDN Controller**: OpenFlow controller for network management
- **Mininet**: Network emulation framework
- **Gymnasium (OpenAI Gym)**: RL environment interface
- **Stable-Baselines3**: DQN implementation
- **Custom REST API**: Monitor network metrics and apply RL decisions

## Project Structure

```
sdn-rl-loadbalancer/
├── controllers/              # Ryu controller applications
│   ├── load_balancer_app.py  # Main load balancing logic
│   └── monitor_api.py        # REST API for monitoring
├── mininet/                  # Network topology & traffic generation
│   ├── custom_topo.py        # Custom topologies (tree, linear, etc.)
│   └── traffic_generator.py  # iperf/scapy traffic tools
├── rl_agent/                 # RL training and evaluation
│   ├── envs/
│   │   ├── __init__.py
│   │   └── sdn_env.py        # Gymnasium environment
│   ├── train_dqn.py          # DQN training script
│   └── evaluate.py           # Agent evaluation
├── utils/                    # Utility modules
│   ├── api_client.py         # Ryu API client
│   ├── system_monitor.py     # CPU/memory monitoring
│   └── visualizer.py         # Plotting and visualization
├── models/                   # Trained models (.zip)
├── logs/                     # TensorBoard logs
├── data/                     # Raw data, plots, results
├── requirements.txt
├── README.md
└── main_integration.py       # Main orchestration script
```

## Prerequisites

### System Requirements

- Linux (Ubuntu 20.04+ recommended)
- Python 3.8+
- Root/sudo access (required for Mininet)

### Dependencies

All dependencies are listed in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

**Note**: Some components may require system-level packages:
- For Mininet: `sudo apt-get install mininet openvswitch-switch`
- For Ryu: Already in pip requirements

## Installation

### 1. Clone the Repository

```bash
cd ~/ryu-env
# Repository should already be here
cd sdn-rl-loadbalancer
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Ryu Controller

In one terminal:

```bash
# Run the load balancer app
ryu-manager controllers/load_balancer_app.py controllers/monitor_api.py
```

### 4. Create Network (Mininet)

In another terminal:

```bash
# Create tree topology
sudo python mininet/custom_topo.py tree

# Or linear topology
sudo python mininet/custom_topo.py linear
```

## Usage

### Training the RL Agent

```bash
python rl_agent/train_dqn.py \
    --total-timesteps 100000 \
    --learning-rate 1e-3 \
    --batch-size 32
```

This will:
- Create a custom SDN environment
- Train DQN agent for specified timesteps
- Save model to `models/` directory
- Log metrics to `logs/` directory (view with TensorBoard)

**View Training Progress**:
```bash
tensorboard --logdir logs/
```

### Evaluating Trained Agent

```bash
python rl_agent/evaluate.py models/best_model
```

This compares the trained agent against baseline policies:
- **Random**: Random port selection
- **Round-robin**: Cycle through ports
- **Least-loaded**: Always select least-utilized port

### Running Full Integration

```bash
python main_integration.py \
    --controller-ip 127.0.0.1 \
    --ryu-port 6633 \
    --topology tree \
    --traffic-scenario multi \
    --train \
    --evaluate
```

### Manual Network Testing

**Test connectivity**:
```bash
# In Mininet CLI
mininet> h1 ping h2
mininet> h1 iperf -s &
mininet> h2 iperf -c h1
```

**Monitor from controller**:
```bash
# In Python
from utils.api_client import RyuAPIClient
client = RyuAPIClient()
print(client.get_flow_stats())
print(client.get_port_stats())
```

## Environment Details

### State Space (Observation)

Network metrics collected from Ryu:
- Link utilization for each port (0-1)
- Average latency (normalized, 0-1)
- Packet loss rate (0-1)
- **Vector size**: `num_ports + 2`

### Action Space

Discrete selection of output port:
- **Options**: 0 to `num_ports - 1`
- **Example**: For 4 ports, actions are [0, 1, 2, 3]

### Reward Function

Negative reward based on:
- Network latency (lower is better)
- Packet loss (lower is better)
- Link utilization balance (standard deviation)

**Formula**:
```
reward = -latency - 2*packet_loss - 0.5*std(link_utils)
```

## Configuration

### Network Parameters (mininet/custom_topo.py)

```python
create_network(
    topo_type='tree',           # 'tree' or 'linear'
    controller_ip='127.0.0.1',
    controller_port=6633,
    num_hosts=4,
    link_bw=10,                 # Mbps
    enable_cli=False
)
```

### RL Training Parameters (rl_agent/train_dqn.py)

```python
train_dqn(
    total_timesteps=100000,
    learning_rate=1e-3,
    exploration_fraction=0.1,
    batch_size=32,
    buffer_size=50000,
    model_dir="models/",
    log_dir="logs/"
)
```

### API Endpoints (controllers/monitor_api.py)

- `GET /stats/flow/<dpid>` - Flow statistics
- `GET /stats/port/<dpid>` - Port statistics
- `POST /routing/apply` - Apply routing action
- `GET /switches` - List all switches
- `GET /topology` - Network topology

## Monitoring

### System Resources

Monitor CPU and memory usage:

```python
from utils.system_monitor import SystemMonitor

monitor = SystemMonitor()
monitor.monitor_ryu_processes(interval=5, duration=300)
summary = monitor.get_summary()
```

### Visualization

Plot training curves, comparisons, and network metrics:

```python
from utils.visualizer import plot_comparison, plot_network_metrics

plot_comparison(agent_metrics, baseline_metrics, output_dir="data/")
plot_network_metrics(metrics_data, title="Network Performance")
```

## Troubleshooting

### Ryu Controller Won't Start

```bash
# Check if port 6633 is in use
lsof -i :6633

# Use different port if needed
ryu-manager --ofp-tcp-listen-port 6634 controllers/load_balancer_app.py
```

### Mininet Startup Issues

```bash
# Clean up any existing Mininet processes
sudo mn -c

# Check permissions
sudo whoami  # Should print 'root'
```

### API Connection Errors

```python
# Verify controller is running and accessible
curl http://127.0.0.1:8080/
```

### Out of Memory During Training

- Reduce `buffer_size` in train_dqn.py
- Reduce `batch_size`
- Use fewer environments (n_envs=1)

## Results and Reports

### Output Files

- **models/**: Saved trained DQN models
- **logs/**: TensorBoard event logs
- **data/**:
  - `training_curve.png` - DQN reward over time
  - `comparison.png` - Agent vs baselines
  - `link_utilization.png` - Network metrics
  - `metrics.csv` - Raw results data

### Example Metrics

After training and evaluation, you should see:

```
Agent:
  mean_reward: -0.45
  std_reward: 0.23
  
Random Baseline:
  mean_reward: -0.78
  
Round-robin Baseline:
  mean_reward: -0.65
```

## Performance Tips

1. **Increase training steps** for convergence
   ```python
   train_dqn(total_timesteps=500000)
   ```

2. **Adjust reward function** in `sdn_env.py`
   - Increase penalty for latency if too slow
   - Increase bonus for balanced utilization

3. **Use hyperparameter tuning**
   - Learning rate: 1e-4 to 1e-2
   - Exploration fraction: 0.05 to 0.2
   - Batch size: 32 to 256

4. **Generate diverse traffic** for better generalization
   - Use `traffic_scenario='varying'` in traffic_generator.py

## Contributing

To extend this project:

1. **Custom topologies**: Add to `mininet/custom_topo.py`
2. **New RL algorithms**: Implement in `rl_agent/` (A2C, PPO, etc.)
3. **Advanced routing**: Enhance `controllers/load_balancer_app.py`
4. **Network metrics**: Add to REST API in `controllers/monitor_api.py`

## References

- [Ryu Documentation](https://ryu.readthedocs.io/)
- [Mininet](http://mininet.org/)
- [Gymnasium](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [OpenFlow 1.3](https://www.opennetworking.org/)

## License

[Add your license here]

## Authors

- Project Title: SDN RL Load Balancer
- Initial Setup: [Your Name]

## Support

For issues, questions, or suggestions:

1. Check existing documentation
2. Review error logs in `logs/`
3. Test components individually (Ryu, Mininet, RL)
4. Verify all dependencies are installed
