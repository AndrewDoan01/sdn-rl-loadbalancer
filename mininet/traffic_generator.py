"""
Traffic Generator for Mininet
Generates realistic network traffic using iperf and scapy
"""

import time
import logging
from typing import Tuple
from mininet.net import Mininet

logger = logging.getLogger(__name__)


class TrafficGenerator:
    """Generate traffic between hosts in Mininet."""
    
    def __init__(self, net: Mininet):
        """
        Initialize traffic generator.
        
        Args:
            net: Mininet network instance
        """
        self.net = net
        self.flows = []
    
    def add_iperf_flow(
        self,
        src_host: str,
        dst_host: str,
        duration: int = 10,
        bandwidth: str = '1M',
        protocol: str = 'TCP'
    ) -> Tuple[str, str]:
        """
        Add iperf flow.
        
        Args:
            src_host: Source host name
            dst_host: Destination host name
            duration: Flow duration in seconds
            bandwidth: Target bandwidth (e.g., '1M', '10M')
            protocol: 'TCP' or 'UDP'
            
        Returns:
            (server_proc, client_proc)
        """
        
        src = self.net.get(src_host)
        dst = self.net.get(dst_host)
        
        if not src or not dst:
            logger.error(f"Host {src_host} or {dst_host} not found")
            return None
        
        logger.info(
            f"Starting iperf: {src_host} -> {dst_host} "
            f"({bandwidth}, {duration}s, {protocol})"
        )
        
        # Start server on destination
        protocol_flag = '-u' if protocol == 'UDP' else ''
        server_cmd = f'iperf -s {protocol_flag} -p 5001'
        server_proc = dst.popen(server_cmd)
        
        time.sleep(1)  # Wait for server to start
        
        # Start client on source
        bandwidth_flag = f'-b {bandwidth}' if protocol == 'UDP' else f'-B {bandwidth}'
        client_cmd = (
            f'iperf -c {dst.IP()} {protocol_flag} '
            f'{bandwidth_flag} -t {duration} -p 5001'
        )
        client_proc = src.popen(client_cmd)
        
        self.flows.append({
            'src': src_host,
            'dst': dst_host,
            'server': server_proc,
            'client': client_proc,
            'duration': duration,
        })
        
        return server_proc, client_proc
    
    def add_ping_flow(self, src_host: str, dst_host: str, count: int = 5) -> None:
        """
        Add ping flow.
        
        Args:
            src_host: Source host name
            dst_host: Destination host name
            count: Number of ICMP packets
        """
        
        src = self.net.get(src_host)
        dst = self.net.get(dst_host)
        
        logger.info(f"Starting ping: {src_host} -> {dst_host}")
        
        # Run ping
        cmd = f'ping -c {count} {dst.IP()}'
        src.cmd(cmd)
    
    def wait_for_completion(self, timeout: int = 300) -> None:
        """Wait for all flows to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_done = True
            
            for flow in self.flows:
                if flow['client']:
                    retcode = flow['client'].poll()
                    if retcode is None:
                        all_done = False
            
            if all_done:
                break
            
            time.sleep(0.5)
        
        logger.info("All flows completed")
    
    def start_constant_load(
        self,
        src_host: str,
        dst_host: str,
        bandwidth: str = '1M'
    ) -> None:
        """
        Start constant load between two hosts.
        
        Args:
            src_host: Source host
            dst_host: Destination host
            bandwidth: Bandwidth to use
        """
        # Use long duration for continuous load
        self.add_iperf_flow(
            src_host,
            dst_host,
            duration=3600,  # 1 hour
            bandwidth=bandwidth
        )
    
    def stop_all_flows(self) -> None:
        """Stop all active flows."""
        for src in self.net.hosts:
            src.cmd('pkill iperf')
        
        logger.info("Stopped all flows")
    
    def get_host_bandwidth_usage(self, host_name: str) -> dict:
        """
        Get current bandwidth usage for a host.
        
        Args:
            host_name: Host name
            
        Returns:
            Dictionary with tx_bytes, rx_bytes, etc.
        """
        host = self.net.get(host_name)
        if not host:
            return {}
        
        # Use ifstat to get interface stats
        result = host.cmd('cat /proc/net/dev | grep eth')
        # Parse and return stats (simplified)
        return {'raw': result}


def generate_traffic_scenario(
    net: Mininet,
    scenario: str = 'basic'
) -> TrafficGenerator:
    """
    Generate predefined traffic scenarios.
    
    Scenarios:
    - 'basic': Single flow
    - 'multi': Multiple concurrent flows
    - 'varying': Traffic varies over time
    
    Args:
        net: Mininet network
        scenario: Scenario type
        
    Returns:
        TrafficGenerator instance
    """
    
    tgen = TrafficGenerator(net)
    hosts = [h.name for h in net.hosts]
    
    if scenario == 'basic':
        if len(hosts) >= 2:
            tgen.add_iperf_flow(hosts[0], hosts[1], duration=10)
    
    elif scenario == 'multi':
        # Multiple flows
        for i in range(len(hosts) // 2):
            src = hosts[i]
            dst = hosts[-(i+1)]
            tgen.add_iperf_flow(src, dst, duration=10)
    
    elif scenario == 'varying':
        # Starting with different bandwidths
        for i, bw in enumerate(['1M', '5M', '10M']):
            if i < len(hosts) // 2:
                tgen.add_iperf_flow(hosts[i], hosts[-(i+1)], bandwidth=bw)
    
    return tgen


if __name__ == '__main__':
    from custom_topo import create_network
    
    net = create_network(topo_type='tree')
    tgen = generate_traffic_scenario(net, scenario='multi')
    tgen.wait_for_completion()
    tgen.stop_all_flows()
    net.stop()
