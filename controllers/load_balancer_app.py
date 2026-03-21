"""
Ryu SDN Controller - Load Balancer Application
Implements basic load balancing and packet routing
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_0
from ryu.lib.packet import packet, ethernet, arp
import logging

logger = logging.getLogger(__name__)


class LoadBalancerApp(app_manager.RyuApp):
    """
    Load Balancer SDN Application.
    
    Handles:
    - ARP requests
    - Flow routing with load balancing
    - Basic statistics collection
    """
    
    OFP_VERSIONS = [ofproto_v1_0.OFP_VERSION]
    
    def __init__(self, *args, **kwargs):
        super(LoadBalancerApp, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.port_stats = {}
        
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle switch features."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # Install table miss
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                         ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        
        logger.info(f"Switch {datapath.id} registered")
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle incoming packets."""
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
        
        self.mac_to_port.setdefault(dpid, {})
        
        # Learn source MAC
        self.mac_to_port[dpid][src] = in_port
        
        # Handle ARP
        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt:
            self.handle_arp(datapath, in_port, eth, arp_pkt, msg)
            return
        
        # Forward packet
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD
        
        actions = [parser.OFPActionOutput(out_port)]
        
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            self.add_flow(datapath, 1, match, actions)
        
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data
        )
        datapath.send_msg(out)
    
    def handle_arp(self, datapath, in_port, eth, arp_pkt, msg):
        """Handle ARP requests."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        if arp_pkt.opcode != arp.ARP_REQUEST:
            return
        
        # Simple ARP reply (in real implementation, handle proper routing)
        logger.debug(f"ARP request for {arp_pkt.dst_ip}")
    
    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        """Install flow entry."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        if buffer_id:
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
