"""
API Client for Ryu REST API
Provides Python functions to call Ryu monitoring endpoints
"""

import requests
import logging
from typing import Dict, Any, Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class RyuAPIClient:
    """Client for Ryu REST API."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        """
        Initialize Ryu API client.
        
        Args:
            base_url: Base URL of Ryu controller
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.timeout = 5
    
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request to Ryu API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests
            
        Returns:
            JSON response or None on error
        """
        
        url = urljoin(self.base_url, endpoint)
        
        try:
            response = self.session.request(
                method,
                url,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def get_flow_stats(self, dpid: Optional[int] = None) -> Dict[str, Any]:
        """
        Get flow statistics from switch(es).
        
        Args:
            dpid: Datapath ID of specific switch (None for all)
            
        Returns:
            Dictionary of flow statistics
        """
        if dpid is not None:
            endpoint = f"/stats/flow/{dpid}"
        else:
            endpoint = "/stats/flow/all"
        
        return self._request("GET", endpoint) or {}
    
    def get_port_stats(self, dpid: Optional[int] = None) -> Dict[str, Any]:
        """
        Get port statistics from switch(es).
        
        Args:
            dpid: Datapath ID of specific switch (None for all)
            
        Returns:
            Dictionary of port statistics
        """
        if dpid is not None:
            endpoint = f"/stats/port/{dpid}"
        else:
            endpoint = "/stats/port/all"
        
        return self._request("GET", endpoint) or {}
    
    def apply_routing_action(self, action: int, metadata: Optional[Dict] = None) -> bool:
        """
        Apply routing action from RL agent.
        
        Args:
            action: Action to apply (e.g., port selection)
            metadata: Additional metadata (timestamp, etc.)
            
        Returns:
            True if successful
        """
        payload = {"action": int(action)}
        if metadata:
            payload.update(metadata)
        
        response = self._request(
            "POST",
            "/routing/apply",
            json=payload
        )
        
        return response is not None
    
    def get_switch_status(self) -> Dict[str, Any]:
        """Get status of all switches."""
        return self._request("GET", "/switches") or {}
    
    def get_switch_by_id(self, dpid: int) -> Dict[str, Any]:
        """Get specific switch information."""
        return self._request("GET", f"/switches/{dpid}") or {}
    
    def get_topology(self) -> Dict[str, Any]:
        """Get network topology information."""
        return self._request("GET", "/topology") or {}
    
    def install_flow_rule(
        self,
        dpid: int,
        priority: int,
        match: Dict,
        actions: list
    ) -> bool:
        """
        Install a flow rule on a switch.
        
        Args:
            dpid: Switch datapath ID
            priority: Rule priority
            match: Match fields
            actions: List of actions
            
        Returns:
            True if successful
        """
        payload = {
            "dpid": dpid,
            "priority": priority,
            "match": match,
            "actions": actions,
        }
        
        response = self._request(
            "POST",
            "/flows/install",
            json=payload
        )
        
        return response is not None
    
    def delete_flow_rule(self, dpid: int, match: Dict) -> bool:
        """
        Delete a flow rule from a switch.
        
        Args:
            dpid: Switch datapath ID
            match: Match fields
            
        Returns:
            True if successful
        """
        payload = {
            "dpid": dpid,
            "match": match,
        }
        
        response = self._request(
            "POST",
            "/flows/delete",
            json=payload
        )
        
        return response is not None
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """Get aggregated network metrics."""
        metrics = {}
        
        flow_stats = self.get_flow_stats()
        port_stats = self.get_port_stats()
        
        metrics['flow_stats'] = flow_stats
        metrics['port_stats'] = port_stats
        
        return metrics
    
    def is_healthy(self) -> bool:
        """Check if controller is healthy."""
        try:
            response = self.session.get(
                urljoin(self.base_url, "/"),
                timeout=2
            )
            return response.status_code == 200
        except:
            return False


class ControllerMonitor:
    """Monitor Ryu controller health and performance."""
    
    def __init__(self, api_client: RyuAPIClient):
        """
        Initialize controller monitor.
        
        Args:
            api_client: RyuAPIClient instance
        """
        self.api_client = api_client
    
    def monitor_loop(self, interval: int = 5, duration: int = 300) -> None:
        """
        Monitor controller in a loop.
        
        Args:
            interval: Monitoring interval in seconds
            duration: Total monitoring duration in seconds
        """
        import time
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            if self.api_client.is_healthy():
                metrics = self.api_client.get_network_metrics()
                logger.info(f"Metrics collected: {len(metrics)} entries")
            else:
                logger.warning("Controller is not healthy")
            
            time.sleep(interval)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of controller status and metrics."""
        return {
            'healthy': self.api_client.is_healthy(),
            'topology': self.api_client.get_topology(),
            'switches': self.api_client.get_switch_status(),
            'metrics': self.api_client.get_network_metrics(),
        }
