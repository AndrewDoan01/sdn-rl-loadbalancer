"""
Visualization Tools
Plot metrics, reward curves, and network statistics
"""

import matplotlib.pyplot as plt
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def plot_training_curves(
    log_dir: str,
    output_dir: str = "data/",
    title: str = "DQN Training"
) -> None:
    """
    Plot training reward curves from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        output_dir: Directory to save plots
        title: Plot title
    """
    
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        # Extract reward data
        if 'rollout/ep_rew_mean' in ea.scalars.Keys():
            rewards = ea.scalars.Items('rollout/ep_rew_mean')
            steps = [r.step for r in rewards]
            values = [r.value for r in rewards]
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(steps, values, linewidth=2)
            plt.xlabel('Timesteps')
            plt.ylabel('Episode Reward')
            plt.title(title)
            plt.grid(True, alpha=0.3)
            
            output_path = Path(output_dir) / "training_curve.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved training curve to {output_path}")
            plt.close()
    except ImportError:
        logger.warning("TensorBoard not installed, skipping training curve plot")


def plot_comparison(
    agent_metrics: Dict[str, float],
    baseline_metrics: Dict[str, Dict[str, float]],
    output_dir: str = "data/"
) -> None:
    """
    Plot comparison between agent and baselines.
    
    Args:
        agent_metrics: Agent performance metrics
        baseline_metrics: Baseline performance metrics
        output_dir: Directory to save plots
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    agent_reward = agent_metrics.get('mean_reward', 0)
    
    baselines = list(baseline_metrics.keys())
    baseline_rewards = [
        baseline_metrics[b].get('mean_reward', 0) for b in baselines
    ]
    
    all_names = ['Agent'] + baselines
    all_rewards = [agent_reward] + baseline_rewards
    
    # Plot bar chart
    plt.figure(figsize=(10, 6))
    colors = ['green'] + ['gray'] * len(baselines)
    bars = plt.bar(all_names, all_rewards, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, reward in zip(bars, all_rewards):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{reward:.2f}',
            ha='center',
            va='bottom'
        )
    
    plt.xlabel('Method')
    plt.ylabel('Mean Episode Reward')
    plt.title('Performance Comparison: Agent vs Baselines')
    plt.grid(True, alpha=0.3, axis='y')
    
    output_path = Path(output_dir) / "comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved comparison plot to {output_path}")
    plt.close()


def plot_network_metrics(
    metrics_data: Dict[str, List[float]],
    output_dir: str = "data/",
    title: str = "Network Metrics"
) -> None:
    """
    Plot network metrics over time.
    
    Args:
        metrics_data: Dictionary of metric name -> values
        output_dir: Directory to save plots
        title: Plot title
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    for metric_name, values in metrics_data.items():
        plt.plot(values, label=metric_name, linewidth=2)
    
    plt.xlabel('Time (samples)')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / "network_metrics.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved network metrics plot to {output_path}")
    plt.close()


def plot_link_utilization(
    link_stats: Dict[str, List[float]],
    output_dir: str = "data/"
) -> None:
    """
    Plot link utilization.
    
    Args:
        link_stats: Dictionary of link -> utilization values
        output_dir: Directory to save plots
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    for link_name, utils in link_stats.items():
        plt.plot(utils, label=link_name, linewidth=2)
    
    plt.xlabel('Time (samples)')
    plt.ylabel('Link Utilization (%)')
    plt.title('Link Utilization Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 100])
    
    output_path = Path(output_dir) / "link_utilization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved link utilization plot to {output_path}")
    plt.close()


def plot_latency(
    latencies: Dict[str, List[float]],
    output_dir: str = "data/"
) -> None:
    """
    Plot latency metrics.
    
    Args:
        latencies: Dictionary of flow -> latency values (ms)
        output_dir: Directory to save plots
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    for flow_name, lats in latencies.items():
        plt.plot(lats, label=flow_name, linewidth=2)
    
    plt.xlabel('Time (samples)')
    plt.ylabel('Latency (ms)')
    plt.title('Flow Latency Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / "latency.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved latency plot to {output_path}")
    plt.close()


def export_metrics_to_csv(
    metrics: Dict[str, Any],
    output_path: str
) -> None:
    """
    Export metrics to CSV format.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save CSV file
    """
    import csv
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    
    logger.info(f"Saved metrics to {output_path}")


def export_metrics_to_json(
    metrics: Dict[str, Any],
    output_path: str
) -> None:
    """
    Export metrics to JSON format.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save JSON file
    """
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved metrics to {output_path}")
