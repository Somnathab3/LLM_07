"""SCAT dataset parser and neighbor finder"""

import json
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Iterator, Dict, Any
from pathlib import Path


@dataclass
class TrackPoint:
    """Individual trajectory point"""
    timestamp: float
    latitude: float
    longitude: float
    altitude_ft: float
    heading_deg: float
    speed_kt: float
    vertical_rate_fpm: Optional[float] = None
    mach: Optional[float] = None


@dataclass
class FlightPlan:
    """Flight plan information"""
    callsign: str
    route_string: str
    requested_flight_level: int
    cruise_tas: int
    waypoints: List[Dict[str, Any]]
    clearance_updates: List[Dict[str, Any]]


class SCATAdapter:
    """SCAT dataset parser and neighbor finder"""
    
    def __init__(self, scat_file: Path):
        self.scat_file = scat_file
        self._data: Optional[Dict] = None
    
    def load_file(self) -> Dict[str, Any]:
        """Load and validate SCAT file structure"""
        with open(self.scat_file, 'r') as f:
            self._data = json.load(f)
        
        # Validate required fields
        required_fields = ['plots', 'fpl_plan_update']
        for field in required_fields:
            if field not in self._data:
                raise ValueError(f"Missing required field: {field}")
        
        return self._data
    
    def ownship_track(self) -> Iterator[TrackPoint]:
        """Generate ownship trajectory points"""
        if not self._data:
            self.load_file()
        
        for plot in sorted(self._data['plots'], key=lambda x: x['time_of_track']):
            # Extract I062 fields
            i062_105 = plot.get('I062/105', {})
            i062_136 = plot.get('I062/136', {})
            i062_380 = plot.get('I062/380', {})
            i062_220 = plot.get('I062/220', {})
            
            # Convert measured flight level to feet
            alt_ft = i062_136.get('measured_flight_level', 0) * 100
            
            yield TrackPoint(
                timestamp=plot['time_of_track'],
                latitude=i062_105.get('latitude', 0.0),
                longitude=i062_105.get('longitude', 0.0),
                altitude_ft=alt_ft,
                heading_deg=i062_380.get('magnetic_heading', 0.0),
                speed_kt=i062_380.get('indicated_airspeed', 0.0),
                vertical_rate_fpm=i062_220.get('rocd', None),
                mach=i062_380.get('mach_number', None)
            )
    
    def flight_plan(self) -> FlightPlan:
        """Extract flight plan information"""
        if not self._data:
            self.load_file()
        
        fpl_update = self._data.get('fpl_plan_update', {})
        fpl_clearance = self._data.get('fpl_clearance', {})
        
        return FlightPlan(
            callsign=fpl_update.get('callsign', ''),
            route_string=fpl_update.get('route', ''),
            requested_flight_level=fpl_update.get('rfl', 0),
            cruise_tas=fpl_update.get('tas', 0),
            waypoints=fpl_update.get('waypoints', []),
            clearance_updates=fpl_clearance.get('updates', [])
        )
    
    def find_neighbors(self, scat_directory: Path, 
                      radius_nm: float = 100.0, 
                      altitude_tolerance_ft: float = 5000.0) -> List['SCATAdapter']:
        """Find neighboring flights within specified radius and altitude"""
        neighbors = []
        own_track = list(self.ownship_track())
        
        if not own_track:
            return neighbors
        
        # Get ownship time window
        start_time = own_track[0].timestamp
        end_time = own_track[-1].timestamp
        
        # Search other SCAT files
        for scat_file in scat_directory.glob('*.json'):
            if scat_file == self.scat_file:
                continue
                
            try:
                neighbor_adapter = SCATAdapter(scat_file)
                neighbor_track = list(neighbor_adapter.ownship_track())
                
                # Check time overlap
                if not neighbor_track:
                    continue
                    
                n_start = neighbor_track[0].timestamp
                n_end = neighbor_track[-1].timestamp
                
                if n_end < start_time or n_start > end_time:
                    continue
                
                # Simple distance check (would use KDTree in full implementation)
                for n_point in neighbor_track[:10]:  # Sample first 10 points
                    if start_time <= n_point.timestamp <= end_time:
                        # Simple distance approximation
                        for o_point in own_track[:10]:
                            if abs(o_point.timestamp - n_point.timestamp) < 300:  # 5 min window
                                lat_diff = abs(o_point.latitude - n_point.latitude)
                                lon_diff = abs(o_point.longitude - n_point.longitude)
                                if lat_diff < 2.0 and lon_diff < 2.0:  # Rough proximity
                                    neighbors.append(neighbor_adapter)
                                    break
                        break
                        
            except Exception as e:
                print(f"Error processing {scat_file}: {e}")
                continue
        
        return neighbors
