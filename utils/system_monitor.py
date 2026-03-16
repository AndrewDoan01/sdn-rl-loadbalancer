"""
System Monitor for Ryu Processes
Monitor CPU, memory, and other resource usage
"""

import psutil
import logging
import time
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor system resources for Ryu processes."""
    
    def __init__(self):
        """Initialize system monitor."""
        self.process_data = defaultdict(list)
        self.start_time = time.time()
    
    def find_ryu_processes(self) -> List[psutil.Process]:
        """
        Find all Ryu controller processes.
        
        Returns:
            List of psutil.Process objects for Ryu
        """
        ryu_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.cmdline() or [])
                if 'ryu' in cmdline.lower() or 'controller' in cmdline.lower():
                    ryu_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return ryu_processes
    
    def get_process_metrics(self, proc: psutil.Process) -> Dict:
        """
        Get metrics for a specific process.
        
        Args:
            proc: psutil.Process object
            
        Returns:
            Dictionary with CPU%, memory, etc.
        """
        try:
            with proc.oneshot():
                metrics = {
                    'pid': proc.pid,
                    'name': proc.name(),
                    'status': proc.status(),
                    'cpu_percent': proc.cpu_percent(interval=0.1),
                    'memory_percent': proc.memory_percent(),
                    'num_threads': proc.num_threads(),
                    'create_time': proc.create_time(),
                }
                
                # Memory info
                mem_info = proc.memory_info()
                metrics['rss'] = mem_info.rss / (1024 * 1024)  # MB
                metrics['vms'] = mem_info.vms / (1024 * 1024)  # MB
                
                return metrics
        except Exception as e:
            logger.error(f"Error getting process metrics: {e}")
            return {}
    
    def get_system_metrics(self) -> Dict:
        """Get overall system metrics."""
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'cpu_count': psutil.cpu_count(),
            'virtual_memory': {
                'total': psutil.virtual_memory().total / (1024 * 1024 * 1024),  # GB
                'available': psutil.virtual_memory().available / (1024 * 1024 * 1024),  # GB
                'percent': psutil.virtual_memory().percent,
            },
            'disk_usage': {
                'total': psutil.disk_usage('/').total / (1024 * 1024 * 1024),  # GB
                'used': psutil.disk_usage('/').used / (1024 * 1024 * 1024),  # GB
                'percent': psutil.disk_usage('/').percent,
            },
        }
        return metrics
    
    def monitor_ryu_processes(self, interval: int = 5, duration: int = 300) -> None:
        """
        Monitor Ryu processes periodically.
        
        Args:
            interval: Monitoring interval in seconds
            duration: Total monitoring duration in seconds
        """
        start_time = time.time()
        
        while time.time() - start_time < duration:
            processes = self.find_ryu_processes()
            
            if processes:
                logger.info(f"Found {len(processes)} Ryu processes")
                
                for proc in processes:
                    metrics = self.get_process_metrics(proc)
                    if metrics:
                        key = f"{metrics['pid']}_{metrics['name']}"
                        self.process_data[key].append({
                            'timestamp': time.time(),
                            'metrics': metrics,
                        })
                        
                        logger.info(
                            f"PID {metrics['pid']}: "
                            f"CPU={metrics['cpu_percent']:.1f}%, "
                            f"MEM={metrics['memory_percent']:.1f}% "
                            f"({metrics['rss']:.1f}MB)"
                        )
            else:
                logger.warning("No Ryu processes found")
            
            # Also log system metrics
            sys_metrics = self.get_system_metrics()
            logger.info(
                f"System: CPU={sys_metrics['cpu_percent']:.1f}%, "
                f"MEM={sys_metrics['virtual_memory']['percent']:.1f}%"
            )
            
            time.sleep(interval)
    
    def get_summary(self) -> Dict:
        """Get summary of monitored processes."""
        summary = {}
        
        for key, data_list in self.process_data.items():
            if data_list:
                metrics_list = [d['metrics'] for d in data_list]
                
                cpu_values = [m['cpu_percent'] for m in metrics_list]
                mem_values = [m['memory_percent'] for m in metrics_list]
                
                summary[key] = {
                    'samples': len(metrics_list),
                    'avg_cpu': sum(cpu_values) / len(cpu_values),
                    'max_cpu': max(cpu_values),
                    'avg_mem': sum(mem_values) / len(mem_values),
                    'max_mem': max(mem_values),
                }
        
        return summary
    
    def get_top_processes(self, metric: str = 'cpu_percent', top_n: int = 5) -> List[Dict]:
        """
        Get top N processes by metric.
        
        Args:
            metric: Metric to sort by ('cpu_percent', 'memory_percent')
            top_n: Number of processes to return
            
        Returns:
            List of process metrics sorted by metric
        """
        processes = self.find_ryu_processes()
        metrics_list = []
        
        for proc in processes:
            metrics = self.get_process_metrics(proc)
            if metrics:
                metrics_list.append(metrics)
        
        # Sort by metric
        metrics_list.sort(key=lambda x: x.get(metric, 0), reverse=True)
        
        return metrics_list[:top_n]


class ResourceAlert:
    """Alert on resource threshold violations."""
    
    def __init__(
        self,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 80.0
    ):
        """
        Initialize resource alert.
        
        Args:
            cpu_threshold: CPU usage threshold (%)
            memory_threshold: Memory usage threshold (%)
        """
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.alerts = []
    
    def check_alerts(self, monitor: SystemMonitor) -> List[str]:
        """
        Check for resource alerts.
        
        Args:
            monitor: SystemMonitor instance
            
        Returns:
            List of alert messages
        """
        alerts = []
        
        sys_metrics = monitor.get_system_metrics()
        
        if sys_metrics['cpu_percent'] > self.cpu_threshold:
            msg = f"HIGH CPU: {sys_metrics['cpu_percent']:.1f}%"
            alerts.append(msg)
            logger.warning(msg)
        
        if sys_metrics['virtual_memory']['percent'] > self.memory_threshold:
            msg = f"HIGH MEMORY: {sys_metrics['virtual_memory']['percent']:.1f}%"
            alerts.append(msg)
            logger.warning(msg)
        
        processes = monitor.find_ryu_processes()
        for proc in processes:
            metrics = monitor.get_process_metrics(proc)
            if metrics:
                if metrics['cpu_percent'] > self.cpu_threshold:
                    msg = f"HIGH CPU (PID {metrics['pid']}): {metrics['cpu_percent']:.1f}%"
                    alerts.append(msg)
                    logger.warning(msg)
                
                if metrics['memory_percent'] > self.memory_threshold:
                    msg = f"HIGH MEMORY (PID {metrics['pid']}): {metrics['memory_percent']:.1f}%"
                    alerts.append(msg)
                    logger.warning(msg)
        
        self.alerts.extend(alerts)
        return alerts
