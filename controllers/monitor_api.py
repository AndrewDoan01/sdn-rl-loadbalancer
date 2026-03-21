"""
Ryu Custom REST API for Monitoring
Provides REST endpoints to:
- Collect network statistics (flow stats, port stats)
- Apply routing decisions from RL agent
"""

from ryu.controller import ofp_event
from ryu.controller.handler import set_ev_cls, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.base import app_manager
from ryu.ofproto import ofproto_v1_3

import json
import logging
from webob import Response
from collections import defaultdict

logger = logging.getLogger(__name__)


class MonitorAPI(app_manager.RyuApp):
    """
    Monitoring API for SDN controller.
    
    REST Endpoints:
    - GET /stats/flow/<dpid>: Flow statistics
    - GET /stats/port/<dpid>: Port statistics
    - POST /routing/apply: Apply routing action
    - GET /topology: Network topology info
    """
    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    _INSTANCES = {}
    
    def __init__(self, *args, **kwargs):
        super(MonitorAPI, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.flow_stats = defaultdict(dict)
        self.port_stats = defaultdict(dict)
        
        # Register WSGI app
        wsgi = kwargs['wsgi']
        wsgi.register_instance(self)
    
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        """Track switch state changes."""
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                logger.info(f"Register datapath {datapath.id}")
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                logger.info(f"Unregister datapath {datapath.id}")
                del self.datapaths[datapath.id]
    
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        """Handle flow statistics from switches."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        
        for stat in ev.msg.body:
            key = (datapath.id, stat.table_id)
            self.flow_stats[key] = {
                'packet_count': stat.packet_count,
                'byte_count': stat.byte_count,
                'duration_sec': stat.duration_sec,
                'duration_nsec': stat.duration_nsec,
            }
    
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        """Handle port statistics from switches."""
        datapath = ev.msg.datapath
        
        for stat in ev.msg.body:
            key = (datapath.id, stat.port_no)
            self.port_stats[key] = {
                'tx_bytes': stat.tx_bytes,
                'rx_bytes': stat.rx_bytes,
                'tx_packets': stat.tx_packets,
                'rx_packets': stat.rx_packets,
                'tx_errors': stat.tx_errors,
                'rx_errors': stat.rx_errors,
            }
    
    def rest_stats_flow(self, req, **kwargs):
        """REST endpoint: GET /stats/flow/<dpid>"""
        dpid = kwargs.get('dpid')
        
        if dpid:
            dpid = int(dpid)
            if dpid in self.datapaths:
                self.request_stats(self.datapaths[dpid], 'flow')
        
        # Return cached stats
        body = json.dumps(dict(self.flow_stats))
        return Response(content_type='application/json', body=body)
    
    def rest_stats_port(self, req, **kwargs):
        """REST endpoint: GET /stats/port/<dpid>"""
        dpid = kwargs.get('dpid')
        
        if dpid:
            dpid = int(dpid)
            if dpid in self.datapaths:
                self.request_stats(self.datapaths[dpid], 'port')
        
        body = json.dumps(dict(self.port_stats))
        return Response(content_type='application/json', body=body)
    
    def rest_routing_apply(self, req, **kwargs):
        """REST endpoint: POST /routing/apply"""
        try:
            data = json.loads(req.body.decode())
            action = data.get('action', -1)
            
            logger.info(f"Applying routing action: {action}")
            
            # In real implementation, apply action to datapaths
            # Update flow tables based on RL agent's decision
            
            return Response(
                content_type='application/json',
                body=json.dumps({'status': 'success', 'action': action})
            )
        except Exception as e:
            logger.error(f"Error applying routing: {e}")
            return Response(
                status=400,
                content_type='application/json',
                body=json.dumps({'error': str(e)})
            )
    
    def request_stats(self, datapath, stat_type):
        """Request statistics from datapath."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        if stat_type == 'flow':
            req = parser.OFPFlowStatsRequest(datapath)
        elif stat_type == 'port':
            req = parser.OFPPortStatsRequest(datapath, ofproto.OFPP_ALL)
        else:
            return
        
        datapath.send_msg(req)
