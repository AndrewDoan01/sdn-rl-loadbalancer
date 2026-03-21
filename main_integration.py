"""
Main Integration Script
Orchestrates the complete SDN RL load balancer workflow
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/integration.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("main_integration")


def setup_directories() -> None:
    """Create necessary directories."""
    dirs = ["models", "logs", "data"]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    logger.info("Created directories: " + ", ".join(dirs))


def start_mininet_network(
    topo_type: str = "tree",
    controller_ip: str = "127.0.0.1",
    controller_port: int = 6633
) -> Optional[object]:
    """
    Start Mininet network.
    
    Args:
        topo_type: Topology type ('tree' or 'linear')
        controller_ip: Ryu controller IP
        controller_port: Ryu controller port
        
    Returns:
        Mininet network instance
    """
    try:
        from mininet.custom_topo import create_network
        
        logger.info(f"Starting Mininet network ({topo_type} topology)...")
        net = create_network(
            topo_type=topo_type,
            controller_ip=controller_ip,
            controller_port=controller_port,
            enable_cli=False
        )
        logger.info("Mininet network started successfully")
        return net
    except ImportError as e:
        logger.warning(f"Mininet not installed: {e}")
        logger.warning("To install: pip install mininet (requires system setup)")
        return None
    except Exception as e:
        logger.error(f"Failed to start Mininet: {e}")
        return None


def generate_traffic(
    net: Optional[object],
    scenario: str = "basic",
    duration: int = 60
) -> None:
    """
    Generate network traffic.
    
    Args:
        net: Mininet network instance
        scenario: Traffic scenario type
        duration: Duration in seconds
    """
    if not net:
        logger.warning("Network not available, skipping traffic generation")
        return
    
    try:
        from mininet.traffic_generator import generate_traffic_scenario
        
        logger.info(f"Generating traffic ({scenario} scenario)...")
        tgen = generate_traffic_scenario(net, scenario=scenario)
        
        if tgen.flows:
            tgen.wait_for_completion(timeout=duration + 10)
            logger.info("Traffic generation completed")
        
    except Exception as e:
        logger.error(f"Failed to generate traffic: {e}")


def train_rl_agent(
    total_timesteps: int = 100000,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    model_dir: str = "models/"
) -> Optional[str]:
    """
    Train DQN agent.
    
    Args:
        total_timesteps: Number of training steps
        learning_rate: Learning rate
        batch_size: Batch size
        model_dir: Directory to save models
        
    Returns:
        Path to best model
    """
    try:
        from rl_agent.train_dqn import train_dqn
        
        logger.info("Starting DQN training...")
        logger.info(
            f"  Timesteps: {total_timesteps}, "
            f"LR: {learning_rate}, BS: {batch_size}"
        )
        
        train_dqn(
            total_timesteps=total_timesteps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            model_dir=model_dir
        )
        
        model_path = Path(model_dir) / "best_model"
        if model_path.exists():
            logger.info(f"Training completed. Best model: {model_path}")
            return str(model_path)
        else:
            logger.warning("No best model found after training")
            return None
            
    except Exception as e:
        logger.error(f"Failed to train agent: {e}")
        return None


def evaluate_agent(model_path: str, n_episodes: int = 10) -> None:
    """
    Evaluate trained agent.
    
    Args:
        model_path: Path to trained model
        n_episodes: Number of evaluation episodes
    """
    try:
        from rl_agent.evaluate import compare_all
        
        logger.info(f"Starting evaluation ({n_episodes} episodes)...")
        compare_all(model_path, output_dir="data/")
        logger.info("Evaluation completed")
        
    except Exception as e:
        logger.error(f"Failed to evaluate agent: {e}")


def monitor_system(duration: int = 60, interval: int = 5) -> None:
    """
    Monitor system resources.
    
    Args:
        duration: Monitoring duration in seconds
        interval: Monitoring interval in seconds
    """
    try:
        from utils.system_monitor import SystemMonitor
        
        logger.info(f"Monitoring system for {duration}s...")
        monitor = SystemMonitor()
        monitor.monitor_ryu_processes(interval=interval, duration=duration)
        
        summary = monitor.get_summary()
        logger.info("Monitoring Summary:")
        for key, metrics in summary.items():
            logger.info(f"  {key}: {metrics}")
        
    except Exception as e:
        logger.error(f"Failed to monitor system: {e}")


def main():
    """Main orchestration function."""
    
    parser = argparse.ArgumentParser(
        description="SDN RL Load Balancer Integration"
    )
    parser.add_argument(
        "--controller-ip",
        default="127.0.0.1",
        help="Ryu controller IP (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--controller-port",
        type=int,
        default=6633,
        help="Ryu controller port (default: 6633)"
    )
    parser.add_argument(
        "--topology",
        choices=["tree", "linear"],
        default="tree",
        help="Network topology (default: tree)"
    )
    parser.add_argument(
        "--traffic-scenario",
        choices=["basic", "multi", "varying"],
        default="basic",
        help="Traffic generation scenario (default: basic)"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,
        help="Total training timesteps (default: 100000)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor system resources"
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Skip network startup (for training only)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("SDN RL Load Balancer - Integration Script")
    logger.info("=" * 60)
    
    # Setup
    setup_directories()
    
    net = None
    
    # Start network
    if not args.no_network:
        net = start_mininet_network(
            topo_type=args.topology,
            controller_ip=args.controller_ip,
            controller_port=args.controller_port
        )
        
        if net:
            time.sleep(2)  # Wait for network to stabilize
            
            # Generate traffic
            if args.train or args.evaluate:
                generate_traffic(net, scenario=args.traffic_scenario)
    
    # Train
    best_model = None
    if args.train:
        best_model = train_rl_agent(
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size
        )
    
    # Evaluate
    if args.evaluate:
        if best_model:
            evaluate_agent(best_model)
        else:
            logger.warning("No model available for evaluation")
    
    # Monitor
    if args.monitor:
        monitor_system(duration=30)
    
    # Cleanup
    if net:
        logger.info("Stopping network...")
        net.stop()
    
    logger.info("=" * 60)
    logger.info("Integration script completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
