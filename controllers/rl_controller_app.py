"""
Ryu SDN Controller Application for RL-based Load Balancing
Integrates RL agent decisions with OpenFlow switches
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, arp
from ryu.lib import dpctl
import logging
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RLControllerApp(app_manager.RyuApp):
    """
    RL-based Load Balancer Ryu Application.
    
    Features:
    - Manages switch topology and ports
    - Tracks flow-to-port mappings
    - Applies routing actions from RL agent
    - Collects network statistics
    - Supports multi-switch scenarios
    """
    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    def __init__(self, *args, **kwargs):
        """Initialize RL Controller App."""
        super(RLControllerApp, self).__init__(*args, **kwargs)
        
        # Datapath registry
        self.datapaths = {}
        
        # Flow tracking: dpid -> {flow_id -> action}
        self.flow_routing = defaultdict(dict)
        
        # Statistics cache: dpid -> stats
        self.flow_stats = defaultdict(dict)
        self.port_stats = defaultdict(dict)
        
        # Port information: dpid -> port_no -> port_data
        self.port_info = defaultdict(dict)
        
        # Action counter for generating unique flow IDs
        self.action_counter = 0
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """
        Handle switch connection and register datapath.
        
        Args:
            ev: EventOFPSwitchFeatures event
        """
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        dpid = datapath.id
        self.datapaths[dpid] = datapath
        
        logger.info(f"Switch {dpid} connected")
        logger.info(f"  Ports: {msg.capabilities}")
        
        # Install default table-miss flow entry
        match = parser.OFPMatch()
        actions = [
            parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                  ofproto.OFPCML_NO_BUFFER)
        ]
        self.add_flow(datapath, 0, match, actions, 
                     buffer_id=ofproto.OFP_NO_BUFFER)
        
        logger.info(f"Installed table-miss flow on switch {dpid}")
    
    @set_ev_cls(ofp_event.EventOFPStateChange, MAIN_DISPATCHER)
    def state_change_handler(self, ev):
        """Handle switch state changes."""
        datapath = ev.datapath
        
        if ev.state == ofp_event.MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                logger.info(f"Switch {datapath.id} entered MAIN_DISPATCHER")
                self.datapaths[datapath.id] = datapath
                # Request port descriptions
                self.request_port_desc(datapath)
        
        elif ev.state == ofp_event.DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                logger.info(f"Switch {datapath.id} disconnected")
                del self.datapaths[datapath.id]
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """
        Handle incoming packets.
        
        Args:
            ev: EventOFPPacketIn event
        """
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.in_port
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        
        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        
        logger.debug(f"PacketIn: dpid={dpid}, src={src}, dst={dst}, in_port={in_port}")
        
        # Handle ARP
        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt:
            self._handle_arp(datapath, in_port, eth, arp_pkt, msg)
            return
        
        # Forward packet (simple learning)
        self._forward_packet(datapath, in_port, eth, msg)
    
    @set_ev_cls(ofp_event.EventOFPPortDesc, MAIN_DISPATCHER)
    def port_desc_handler(self, ev):
        """Handle port description information."""
        msg = ev.msg
        datapath = msg.datapath
        dpid = datapath.id
        
        for port in msg.body:
            logger.debug(f"Port {port.port_no} on switch {dpid}: {port.name}")
            self.port_info[dpid][port.port_no] = {
                'name': port.name.decode('utf-8') if isinstance(port.name, bytes) else port.name,
                'hw_addr': port.hw_addr,
                'state': port.state,
                'curr': port.curr,
            }
    
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        """Handle flow statistics reply."""
        datapath = ev.msg.datapath
        dpid = datapath.id
        
        for stat in ev.msg.body:
            key = (stat.table_id, stat.priority, stat.match_to_dict())
            self.flow_stats[dpid][key] = {
                'packet_count': stat.packet_count,
                'byte_count': stat.byte_count,
                'duration_sec': stat.duration_sec,
            }
    
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        """Handle port statistics reply."""
        datapath = ev.msg.datapath
        dpid = datapath.id
        
        for stat in ev.msg.body:
            self.port_stats[dpid][stat.port_no] = {
                'tx_bytes': stat.tx_bytes,
                'rx_bytes': stat.rx_bytes,
                'tx_packets': stat.tx_packets,
                'rx_packets': stat.rx_packets,
                'tx_errors': stat.tx_errors,
                'rx_errors': stat.rx_errors,
            }
    
    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        """
        Install a flow rule on a switch.
        
        Args:
            datapath: Datapath object
            priority: Flow priority
            match: Match criteria
            actions: Actions to apply
            buffer_id: Buffer ID (if any)
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        if buffer_id is not None:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                buffer_id=buffer_id,
                priority=priority,
                match=match,
                instructions=inst
            )
        else:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                priority=priority,
                match=match,
                instructions=inst
            )
        
        datapath.send_msg(mod)
        logger.debug(f"Flow added on switch {datapath.id}")
    
    def delete_flow(self, datapath, match, priority=0):
        """
        Delete a flow rule from a switch.
        
        Args:
            datapath: Datapath object
            match: Match criteria
            priority: Flow priority (default: 0)
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        mod = parser.OFPFlowMod(
            datapath=datapath,
            command=ofproto.OFPFC_DELETE,
            priority=priority,
            match=match
        )
        
        datapath.send_msg(mod)
        logger.debug(f"Flow deleted on switch {datapath.id}")
    
    def apply_rl_action(self, dpid: int, flow_match: Dict, 
                       action_port: int) -> bool:
        """
        Apply routing action from RL agent.
        
        Args:
            dpid: Datapath ID
            flow_match: Flow match fields (e.g., {'eth_dst': 'mac', 'eth_type': 2048})
            action_port: Output port selected by RL agent
            
        Returns:
            True if successful, False otherwise
        """
        if dpid not in self.datapaths:
            logger.error(f"Switch {dpid} not found")
            return False
        
        datapath = self.datapaths[dpid]
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        try:
            # Convert flow_match dict to OFPMatch
            match = parser.OFPMatch(**flow_match)
            
            # Create output action
            actions = [parser.OFPActionOutput(action_port)]
            
            # Install flow with priority 100 (higher than default 0)
            self.add_flow(datapath, priority=100, match=match, actions=actions)
            
            # Track routing decision
            flow_id = self._generate_flow_id()
            self.flow_routing[dpid][flow_id] = {
                'match': flow_match,
                'action_port': action_port,
                'timestamp': self._get_timestamp()
            }
            
            logger.info(
                f"RL action applied: DPID={dpid}, port={action_port}, "
                f"match={flow_match}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply RL action: {e}")
            return False
    
    def apply_rl_action_to_switch(self, dpid: int, src_mac: str, 
                                  dst_mac: str, action_port: int) -> bool:
        """
        Apply RL routing action for a specific MAC flow.
        
        Args:
            dpid: Datapath ID
            src_mac: Source MAC address
            dst_mac: Destination MAC address
            action_port: Output port (RL decision)
            
        Returns:
            True if successful
        """
        flow_match = {
            'eth_src': src_mac,
            'eth_dst': dst_mac
        }
        
        return self.apply_rl_action(dpid, flow_match, action_port)
    
    def get_network_topology(self) -> Dict:
        """
        Get current network topology.
        
        Returns:
            Dictionary with switches and ports
        """
        topology = {
            'switches': [],
            'ports': {}
        }
        
        for dpid in self.datapaths:
            topology['switches'].append(dpid)
            topology['ports'][dpid] = list(self.port_info[dpid].keys())
        
        return topology
    
    def get_switch_stats(self, dpid: int) -> Dict:
        """
        Get statistics for a specific switch.
        
        Args:
            dpid: Datapath ID
            
        Returns:
            Dictionary with flow and port statistics
        """
        return {
            'dpid': dpid,
            'flow_stats': dict(self.flow_stats.get(dpid, {})),
            'port_stats': dict(self.port_stats.get(dpid, {})),
        }
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all switches."""
        stats = {}
        for dpid in self.datapaths:
            stats[dpid] = self.get_switch_stats(dpid)
        return stats
    
    def request_stats(self, datapath, stat_type: str):
        """
        Request statistics from a datapath.
        
        Args:
            datapath: Datapath object
            stat_type: Type of stats ('flow' or 'port')
        """
        if stat_type == 'flow':
            ofproto = datapath.ofproto
            parser = datapath.ofproto_parser
            req = parser.OFPFlowStatsRequest(datapath)
            datapath.send_msg(req)
        
        elif stat_type == 'port':
            ofproto = datapath.ofproto
            parser = datapath.ofproto_parser
            req = parser.OFPPortStatsRequest(datapath, ofproto.OFPP_ALL)
            datapath.send_msg(req)
    
    def request_port_desc(self, datapath):
        """Request port descriptions from datapath."""
        parser = datapath.ofproto_parser
        req = parser.OFPPortDescStatsRequest(datapath)
        datapath.send_msg(req)
    
    def get_available_ports(self, dpid: int) -> List[int]:
        """
        Get available ports for a switch.
        
        Args:
            dpid: Datapath ID
            
        Returns:
            List of available port numbers
        """
        if dpid not in self.port_info:
            return []
        
        ports = []
        for port_no, port_data in self.port_info[dpid].items():
            # Exclude special ports (CONTROLLER, LOCAL, etc.)
            if port_no < 65280:  # 65280 is OFPP_LOCAL
                ports.append(port_no)
        
        return sorted(ports)
    
    # Private helper methods
    
    def _forward_packet(self, datapath, in_port, eth, msg):
        """
        Forward packet using simple MAC learning.
        
        Args:
            datapath: Datapath object
            in_port: Incoming port
            eth: Ethernet packet
            msg: OFPPacketIn message
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        
        # Determine output port
        out_port = ofproto.OFPP_FLOOD
        
        # Send packet out
        actions = [parser.OFPActionOutput(out_port)]
        data = None if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=msg.data
        )
        datapath.send_msg(out)
    
    def _handle_arp(self, datapath, in_port, eth, arp_pkt, msg):
        """
        Handle ARP packets.
        
        Args:
            datapath: Datapath object
            in_port: Incoming port
            eth: Ethernet packet
            arp_pkt: ARP packet
            msg: OFPPacketIn message
        """
        logger.debug(f"ARP packet: src_ip={arp_pkt.src_ip}, dst_ip={arp_pkt.dst_ip}")
        
        # Forward ARP by default
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        
        data = None if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=msg.data
        )
        datapath.send_msg(out)
    
    def _generate_flow_id(self) -> int:
        """Generate unique flow ID."""
        self.action_counter += 1
        return self.action_counter
    
    @staticmethod
    def _get_timestamp() -> float:
        """Get current timestamp."""
        import time
        return time.time()
