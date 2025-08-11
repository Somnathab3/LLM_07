"""Advanced conflict detection using geometric algorithms"""

import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConflictPrediction:
    """Conflict prediction result"""
    intruder_callsign: str
    time_to_cpa_minutes: float
    cpa_distance_nm: float
    cpa_altitude_difference_ft: float
    conflict_severity: float  # 0-1 scale
    conflict_type: str  # 'head_on', 'crossing', 'overtaking', 'vertical'
    recommended_resolution: Optional[str] = None


class ConflictDetector:
    """Advanced conflict detection using geometric algorithms"""
    
    def __init__(self, separation_min_nm: float = 5.0, 
                 separation_min_ft: float = 1000.0,
                 lookahead_minutes: float = 10.0):
        self.separation_min_nm = separation_min_nm
        self.separation_min_ft = separation_min_ft
        self.lookahead_minutes = lookahead_minutes
        self.lookahead_seconds = lookahead_minutes * 60
    
    def detect_conflicts(self, ownship_state: Any, intruders: List[Any],
                        lookahead_minutes: Optional[float] = None) -> List[ConflictPrediction]:
        """Detect potential conflicts with intruders"""
        
        if lookahead_minutes is None:
            lookahead_minutes = self.lookahead_minutes
        
        conflicts = []
        
        for intruder in intruders:
            conflict = self._predict_conflict_pair(
                ownship_state, intruder, lookahead_minutes
            )
            
            if conflict:
                conflicts.append(conflict)
        
        # Sort by time to conflict (most urgent first)
        conflicts.sort(key=lambda c: c.time_to_cpa_minutes)
        
        return conflicts
    
    def _predict_conflict_pair(self, ownship: Any, intruder: Any,
                              lookahead_minutes: float) -> Optional[ConflictPrediction]:
        """Predict conflict between ownship and single intruder"""
        
        # Extract current states
        own_lat = getattr(ownship, 'latitude', 0)
        own_lon = getattr(ownship, 'longitude', 0)
        own_alt = getattr(ownship, 'altitude_ft', 0)
        own_hdg = getattr(ownship, 'heading_deg', 0)
        own_spd = getattr(ownship, 'speed_kt', 0)
        
        int_lat = getattr(intruder, 'latitude', 0)
        int_lon = getattr(intruder, 'longitude', 0)
        int_alt = getattr(intruder, 'altitude_ft', 0)
        int_hdg = getattr(intruder, 'heading_deg', 0)
        int_spd = getattr(intruder, 'speed_kt', 0)
        
        # Calculate current separation
        current_distance_nm = self._great_circle_distance_nm(
            own_lat, own_lon, int_lat, int_lon
        )
        current_alt_diff = abs(int_alt - own_alt)
        
        # Simple conflict prediction using linear projection
        time_steps = [i * 60 for i in range(int(lookahead_minutes) + 1)]  # Every minute
        min_separation = float('inf')
        min_alt_diff = current_alt_diff
        conflict_time = 0
        
        for t in time_steps:
            # Project positions forward in time
            own_future_lat, own_future_lon = self._project_position(
                own_lat, own_lon, own_hdg, own_spd, t
            )
            int_future_lat, int_future_lon = self._project_position(
                int_lat, int_lon, int_hdg, int_spd, t
            )
            
            # Calculate separation at this time
            future_distance = self._great_circle_distance_nm(
                own_future_lat, own_future_lon, int_future_lat, int_future_lon
            )
            
            if future_distance < min_separation:
                min_separation = future_distance
                conflict_time = t
                min_alt_diff = current_alt_diff  # Simplified - assume constant altitude
        
        # Check if conflict occurs
        if (min_separation >= self.separation_min_nm or 
            min_alt_diff >= self.separation_min_ft):
            return None
        
        # Determine conflict type
        conflict_type = self._classify_conflict_geometry(
            own_lat, own_lon, own_hdg, int_lat, int_lon, int_hdg
        )
        
        # Calculate severity
        severity = 1.0 - min(min_separation / self.separation_min_nm, 1.0)
        
        return ConflictPrediction(
            intruder_callsign=getattr(intruder, 'callsign', 'UNKNOWN'),
            time_to_cpa_minutes=conflict_time / 60,
            cpa_distance_nm=min_separation,
            cpa_altitude_difference_ft=min_alt_diff,
            conflict_severity=severity,
            conflict_type=conflict_type
        )
    
    def _project_position(self, lat: float, lon: float, heading: float, 
                         speed_kt: float, time_seconds: float) -> Tuple[float, float]:
        """Project aircraft position forward in time"""
        if time_seconds == 0:
            return lat, lon
        
        # Convert speed to distance
        distance_nm = (speed_kt * time_seconds) / 3600  # knots to NM
        
        # Convert heading to radians (aviation to math convention)
        heading_rad = math.radians(90 - heading)
        
        # Calculate position change (simplified flat earth approximation)
        delta_lat = (distance_nm * math.sin(heading_rad)) / 60  # NM to degrees
        delta_lon = (distance_nm * math.cos(heading_rad)) / (60 * math.cos(math.radians(lat)))
        
        return lat + delta_lat, lon + delta_lon
    
    def _classify_conflict_geometry(self, lat1: float, lon1: float, hdg1: float,
                                   lat2: float, lon2: float, hdg2: float) -> str:
        """Classify conflict geometry based on relative positions and headings"""
        
        # Calculate bearing from aircraft 1 to aircraft 2
        bearing = self._calculate_bearing(lat1, lon1, lat2, lon2)
        
        # Calculate relative angles
        rel_angle_1 = abs(hdg1 - bearing)
        rel_angle_2 = abs(hdg2 - (bearing + 180) % 360)
        
        # Normalize angles
        if rel_angle_1 > 180:
            rel_angle_1 = 360 - rel_angle_1
        if rel_angle_2 > 180:
            rel_angle_2 = 360 - rel_angle_2
        
        # Classify based on approach angles
        if rel_angle_1 < 45 and rel_angle_2 < 45:
            return 'head_on'
        elif rel_angle_1 > 135 and rel_angle_2 > 135:
            return 'overtaking'
        else:
            return 'crossing'
    
    def _great_circle_distance_nm(self, lat1: float, lon1: float,
                                 lat2: float, lon2: float) -> float:
        """Calculate great circle distance in nautical miles"""
        
        # Convert to radians
        lat1_r = math.radians(lat1)
        lon1_r = math.radians(lon1)
        lat2_r = math.radians(lat2)
        lon2_r = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(max(0, min(1, a))))
        
        # Earth radius in nautical miles
        r_nm = 3440.065
        
        return r_nm * c
    
    def _calculate_bearing(self, lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
        """Calculate initial bearing from point 1 to point 2"""
        
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        dlon_r = math.radians(lon2 - lon1)
        
        y = math.sin(dlon_r) * math.cos(lat2_r)
        x = (math.cos(lat1_r) * math.sin(lat2_r) - 
             math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r))
        
        bearing_r = math.atan2(y, x)
        bearing_deg = (math.degrees(bearing_r) + 360) % 360
        
        return bearing_deg
