"""
Custom Mininet Topology for SDN Load Balancing
Creates various network topologies (tree, linear, etc.)
"""

from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.util import dumpNodeConnections
import logging

logger = logging.getLogger(__name__)


class TreeTopology(Topo):
    """
    Tree topology:
    
         Controller
              |
             S1 (Root)
            /  \
          S2   S3
         / \   / \
        H1 H2 H3 H4
    """
    
    def __init__(self, depth=2, fanout=2):
        """
        Create tree topology.
        
        Args:
            depth: Tree depth (number of switch levels)
            fanout: Number of children per switch
        """
        super(TreeTopology, self).__init__()
        self.depth = depth
        self.fanout = fanout
        self.host_count = 0
        self.switch_count = 0
        
        self._create_tree(None, 0)
    
    def _create_tree(self, parent_switch, level):
        """Recursively create tree structure."""
        if level >= self.depth:
            return
        
        # Create switches at this level
        num_switches = self.fanout ** level if level > 0 else 1
        
        for i in range(num_switches):
            self.switch_count += 1
            switch = f"s{self.switch_count}"
            self.addSwitch(switch)
            
            if parent_switch:
                self.addLink(parent_switch, switch)
            
            # Add hosts at leaf level
            if level == self.depth - 1:
                for j in range(self.fanout):
                    self.host_count += 1
                    host = f"h{self.host_count}"
                    self.addHost(host)
                    self.addLink(switch, host)
            else:
                # Recursively add child switches
                for j in range(self.fanout):
                    self._create_tree(switch, level + 1)


class LinearTopology(Topo):
    """
    Linear topology: H1 -- S1 -- S2 -- S3 -- H2
    """
    
    def __init__(self, num_switches=3):
        """
        Create linear topology.
        
        Args:
            num_switches: Number of switches in chain
        """
        super(LinearTopology, self).__init__()
        self.num_switches = num_switches
        
        # Create hosts
        self.addHost('h1')
        self.addHost('h2')
        
        # Create switch chain
        prev_switch = None
        for i in range(1, num_switches + 1):
            switch = f's{i}'
            self.addSwitch(switch)
            
            if prev_switch:
                self.addLink(prev_switch, switch)
            else:
                self.addLink('h1', switch)
            
            prev_switch = switch
        
        self.addLink(prev_switch, 'h2')


def create_network(
    topo_type='tree',
    controller_ip='127.0.0.1',
    controller_port=6633,
    num_hosts=4,
    link_bw=10,
    enable_cli=False
):
    """
    Create and start Mininet network.
    
    Args:
        topo_type: Topology type ('tree' or 'linear')
        controller_ip: IP of Ryu controller
        controller_port: Port of Ryu controller
        num_hosts: Number of hosts (for tree topo)
        link_bw: Link bandwidth in Mbps
        enable_cli: Start Mininet CLI
        
    Returns:
        net: Mininet network instance
    """
    
    logger.info(f"Creating {topo_type} topology...")
    
    # Create topology
    if topo_type == 'tree':
        topo = TreeTopology(depth=2, fanout=2)
    elif topo_type == 'linear':
        topo = LinearTopology(num_switches=3)
    else:
        raise ValueError(f"Unknown topology type: {topo_type}")
    
    # Create controller
    controller = RemoteController(
        'c0',
        ip=controller_ip,
        port=controller_port,
        protocol='tcp'
    )
    
    # Create network
    net = Mininet(
        topo=topo,
        controller=controller,
        switch=OVSSwitch,
        link=TCLink
    )
    
    # Configure links
    for link in net.links:
        link.intf1.config(bw=link_bw)
        link.intf2.config(bw=link_bw)
    
    logger.info("Starting network...")
    net.start()
    
    # Print network info
    dumpNodeConnections(net.hosts)
    dumpNodeConnections(net.switches)
    
    if enable_cli:
        cli = CLI(net)
    
    return net


if __name__ == '__main__':
    import sys
    topo = sys.argv[1] if len(sys.argv) > 1 else 'tree'
    net = create_network(topo_type=topo, enable_cli=True)
    net.stop()
