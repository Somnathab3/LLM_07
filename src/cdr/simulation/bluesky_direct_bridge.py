#!/usr/bin/env python3
"""
BlueSky Direct Access Bridge

This module provides direct access to BlueSky's internal API for improved 
data exchange between BlueSky and LLM, similar to the sector-based RL environments.

Key improvements over command-response parsing:
1. Direct access to bs.traf for aircraft states
2. Direct access to bs.traf.asas for conflict detection
3. Eliminates parsing errors and data loss
4. Faster and more reliable data exchange
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import BlueSky modules directly
try:
    import bluesky as bs
    from bluesky.traffic import Traffic
    BLUESKY_AVAILABLE = True
except ImportError:
    BLUESKY_AVAILABLE = False
    print("⚠️ BlueSky not available, using fallback mode")


@dataclass
class DirectAircraftState:
    """Aircraft state from direct BlueSky access"""
    callsign: str
    latitude: float
    longitude: float
    altitude_ft: float
    heading_deg: float
    speed_kt: float
    vertical_speed_fpm: float
    timestamp: float
    track_angle: Optional[float] = None
    ground_speed_kt: Optional[float] = None


@dataclass
class DirectConflictInfo:
    """Conflict info from direct BlueSky ASAS access"""
    aircraft1: str
    aircraft2: str
    time_to_conflict: float
    horizontal_distance: float
    vertical_distance: float
    conflict_type: str
    severity: str
    detection_source: str = "direct_asas"


class BlueSkyDirectBridge:
    """
    Direct bridge to BlueSky internal API
    
    Provides the missing bridge between BlueSky and LLM by accessing
    BlueSky's internal data structures directly, similar to the 
    sector-based RL environments.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.bs_available = BLUESKY_AVAILABLE
        self.last_states = {}
        
        if not self.bs_available:
            self.logger.warning("BlueSky not available - bridge will use fallback mode")
    
    def is_available(self) -> bool:
        """Check if direct BlueSky access is available"""
        return self.bs_available and hasattr(bs, 'traf') and bs.traf is not None
    
    def get_aircraft_states_direct(self) -> Dict[str, DirectAircraftState]:
        """
        Get aircraft states directly from BlueSky traffic module
        
        This bypasses the command-response parsing and accesses bs.traf directly,
        similar to how the sector-based RL environments work.
        """
        if not self.is_available():
            return {}
        
        states = {}
        current_time = time.time()
        
        try:
            # Direct access to BlueSky traffic arrays
            if hasattr(bs.traf, 'id') and len(bs.traf.id) > 0:
                
                for i, callsign in enumerate(bs.traf.id):
                    try:
                        # Get index for this aircraft
                        ac_idx = bs.traf.id2idx(callsign)
                        if ac_idx < 0:
                            continue
                        
                        # Direct access to aircraft data arrays (like RL environments)
                        lat = float(bs.traf.lat[ac_idx]) if hasattr(bs.traf, 'lat') else 0.0
                        lon = float(bs.traf.lon[ac_idx]) if hasattr(bs.traf, 'lon') else 0.0
                        alt_m = float(bs.traf.alt[ac_idx]) if hasattr(bs.traf, 'alt') else 0.0
                        hdg = float(bs.traf.hdg[ac_idx]) if hasattr(bs.traf, 'hdg') else 0.0
                        
                        # Speed handling (different units in BlueSky)
                        if hasattr(bs.traf, 'cas'):
                            # Calibrated airspeed in m/s
                            speed_ms = float(bs.traf.cas[ac_idx])
                            speed_kt = speed_ms * 1.944  # Convert m/s to knots
                        elif hasattr(bs.traf, 'gs'):
                            # Ground speed in m/s
                            speed_ms = float(bs.traf.gs[ac_idx])
                            speed_kt = speed_ms * 1.944  # Convert m/s to knots
                        else:
                            speed_kt = 250.0  # Default
                        
                        # Vertical speed
                        vs_ms = 0.0
                        if hasattr(bs.traf, 'vs'):
                            vs_ms = float(bs.traf.vs[ac_idx])
                        vs_fpm = vs_ms * 196.85  # Convert m/s to ft/min
                        
                        # Altitude conversion (BlueSky uses meters)
                        alt_ft = alt_m * 3.28084  # Convert meters to feet
                        
                        # Track angle if available
                        track = None
                        if hasattr(bs.traf, 'trk'):
                            track = float(bs.traf.trk[ac_idx])
                        
                        # Ground speed if different from cas
                        gs_kt = None
                        if hasattr(bs.traf, 'gs'):
                            gs_ms = float(bs.traf.gs[ac_idx])
                            gs_kt = gs_ms * 1.944
                        
                        # Create aircraft state
                        state = DirectAircraftState(
                            callsign=callsign,
                            latitude=lat,
                            longitude=lon,
                            altitude_ft=alt_ft,
                            heading_deg=hdg,
                            speed_kt=speed_kt,
                            vertical_speed_fpm=vs_fpm,
                            timestamp=current_time,
                            track_angle=track,
                            ground_speed_kt=gs_kt
                        )
                        
                        states[callsign] = state
                        
                        # Debug logging for first few aircraft
                        if len(states) <= 3:
                            self.logger.debug(f"Direct access: {callsign} at {lat:.4f},{lon:.4f}, "
                                            f"{alt_ft:.0f}ft, {hdg:.1f}°, {speed_kt:.0f}kt")
                        
                    except (IndexError, AttributeError, ValueError) as e:
                        self.logger.warning(f"Error getting state for {callsign}: {e}")
                        continue
                
                self.last_states = states
                
                # Debug: Always log aircraft count for troubleshooting
                self.logger.info(f"Direct bridge retrieved {len(states)} aircraft: {list(states.keys())}")
                
            else:
                self.logger.debug("No aircraft in BlueSky traffic")
                
        except Exception as e:
            self.logger.error(f"Error in direct aircraft state access: {e}")
            import traceback
            traceback.print_exc()
        
        return states
    
    def get_conflicts_direct(self) -> List[DirectConflictInfo]:
        """
        Get conflicts directly from BlueSky ASAS module
        
        This accesses BlueSky's internal conflict detection state directly,
        bypassing the SSD CONFLICTS command parsing.
        """
        if not self.is_available():
            return []
        
        conflicts = []
        
        try:
            # Direct access to ASAS conflict detection (like RL environments)
            if hasattr(bs.traf, 'asas') and bs.traf.asas is not None:
                asas = bs.traf.asas
                
                # Check for conflict detection arrays
                if hasattr(asas, 'inconf') and hasattr(asas, 'confpairs'):
                    
                    for i, in_conflict in enumerate(asas.inconf):
                        if in_conflict and i < len(asas.confpairs):
                            
                            # Get conflict pair indices
                            pair = asas.confpairs[i]
                            if len(pair) >= 2:
                                ac1_idx, ac2_idx = pair[0], pair[1]
                                
                                # Get aircraft callsigns
                                if (ac1_idx < len(bs.traf.id) and ac2_idx < len(bs.traf.id)):
                                    ac1_callsign = bs.traf.id[ac1_idx]
                                    ac2_callsign = bs.traf.id[ac2_idx]
                                    
                                    # Calculate distance and metrics
                                    h_dist, v_dist = self._calculate_separation_direct(ac1_idx, ac2_idx)
                                    
                                    # Determine time to conflict if available
                                    time_to_conflict = 0.0
                                    if hasattr(asas, 'tLOS') and i < len(asas.tLOS):
                                        time_to_conflict = float(asas.tLOS[i])
                                    
                                    # Determine conflict type and severity
                                    conflict_type = "horizontal"
                                    severity = "medium"
                                    
                                    if h_dist < 5.0 and v_dist < 1000.0:
                                        conflict_type = "both"
                                        severity = "high"
                                    elif h_dist < 5.0:
                                        conflict_type = "horizontal"
                                        severity = "medium"
                                    elif v_dist < 1000.0:
                                        conflict_type = "vertical" 
                                        severity = "medium"
                                    
                                    conflict = DirectConflictInfo(
                                        aircraft1=ac1_callsign,
                                        aircraft2=ac2_callsign,
                                        time_to_conflict=time_to_conflict,
                                        horizontal_distance=h_dist,
                                        vertical_distance=v_dist,
                                        conflict_type=conflict_type,
                                        severity=severity,
                                        detection_source="direct_asas"
                                    )
                                    
                                    conflicts.append(conflict)
                                    
                                    self.logger.info(f"Direct conflict: {ac1_callsign} vs {ac2_callsign}, "
                                                   f"h={h_dist:.1f}NM, v={v_dist:.0f}ft")
            
            # Alternative: Manual geometric detection using direct access
            if not conflicts:
                conflicts = self._detect_conflicts_geometric_direct()
                
        except Exception as e:
            self.logger.error(f"Error in direct conflict detection: {e}")
            import traceback
            traceback.print_exc()
        
        return conflicts
    
    def _calculate_separation_direct(self, ac1_idx: int, ac2_idx: int) -> tuple:
        """Calculate separation between aircraft using direct access"""
        try:
            # Get positions directly
            lat1 = bs.traf.lat[ac1_idx]
            lon1 = bs.traf.lon[ac1_idx] 
            alt1 = bs.traf.alt[ac1_idx]
            
            lat2 = bs.traf.lat[ac2_idx]
            lon2 = bs.traf.lon[ac2_idx]
            alt2 = bs.traf.alt[ac2_idx]
            
            # Calculate horizontal distance using BlueSky's tools if available
            if hasattr(bs.tools, 'geo'):
                h_dist_nm = bs.tools.geo.kwikdist(lat1, lon1, lat2, lon2)  # Returns NM
            else:
                # Fallback calculation
                import math
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                h_dist_nm = math.sqrt(dlat**2 + dlon**2) * 60  # Rough conversion
            
            # Vertical separation (convert meters to feet)
            v_dist_ft = abs(alt2 - alt1) * 3.28084
            
            return h_dist_nm, v_dist_ft
            
        except Exception as e:
            self.logger.warning(f"Error calculating separation: {e}")
            return 999.0, 999.0  # Safe default
    
    def _detect_conflicts_geometric_direct(self) -> List[DirectConflictInfo]:
        """Manual geometric conflict detection using direct access"""
        conflicts = []
        
        try:
            if not hasattr(bs.traf, 'id') or len(bs.traf.id) < 2:
                return conflicts
            
            # Check all aircraft pairs
            for i in range(len(bs.traf.id)):
                for j in range(i + 1, len(bs.traf.id)):
                    
                    ac1_callsign = bs.traf.id[i]
                    ac2_callsign = bs.traf.id[j]
                    
                    h_dist, v_dist = self._calculate_separation_direct(i, j)
                    
                    # Check if within conflict thresholds
                    h_violation = h_dist < 5.0  # 5 NM
                    v_violation = v_dist < 1000.0  # 1000 ft
                    
                    if h_violation or v_violation:
                        
                        conflict_type = "both" if (h_violation and v_violation) else \
                                      "horizontal" if h_violation else "vertical"
                        
                        severity = "high" if (h_violation and v_violation) else "medium"
                        
                        conflict = DirectConflictInfo(
                            aircraft1=ac1_callsign,
                            aircraft2=ac2_callsign,
                            time_to_conflict=0.0,
                            horizontal_distance=h_dist,
                            vertical_distance=v_dist,
                            conflict_type=conflict_type,
                            severity=severity,
                            detection_source="direct_geometric"
                        )
                        
                        conflicts.append(conflict)
                        
        except Exception as e:
            self.logger.error(f"Error in geometric detection: {e}")
        
        return conflicts
    
    def send_command_direct(self, command: str) -> bool:
        """
        Send command directly to BlueSky stack
        
        This provides a more reliable way to send commands compared to
        socket-based communication.
        """
        if not self.is_available():
            return False
        
        try:
            # Direct access to BlueSky command stack
            if hasattr(bs, 'stack'):
                result = bs.stack.stack(command)
                return result is not False  # BlueSky returns False on error
            else:
                self.logger.warning("BlueSky stack not available")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending direct command '{command}': {e}")
            return False
    
    def apply_heading_command_direct(self, callsign: str, heading_deg: float) -> bool:
        """Apply heading command using direct BlueSky access"""
        command = f"HDG {callsign} {heading_deg:.1f}"
        success = self.send_command_direct(command)
        
        if success:
            self.logger.info(f"Applied direct heading command: {command}")
        else:
            self.logger.error(f"Failed direct heading command: {command}")
        
        return success
    
    def apply_altitude_command_direct(self, callsign: str, altitude_ft: float) -> bool:
        """Apply altitude command using direct BlueSky access"""
        command = f"ALT {callsign} {altitude_ft:.0f}"
        success = self.send_command_direct(command)
        
        if success:
            self.logger.info(f"Applied direct altitude command: {command}")
        else:
            self.logger.error(f"Failed direct altitude command: {command}")
        
        return success
    
    def apply_speed_command_direct(self, callsign: str, speed_kt: float) -> bool:
        """Apply speed command using direct BlueSky access"""
        command = f"SPD {callsign} {speed_kt:.0f}"
        success = self.send_command_direct(command)
        
        if success:
            self.logger.info(f"Applied direct speed command: {command}")
        else:
            self.logger.error(f"Failed direct speed command: {command}")
        
        return success
    
    def convert_to_standard_format(self, direct_states: Dict[str, DirectAircraftState]) -> Dict[str, Any]:
        """Convert direct aircraft states to standard CDR pipeline format"""
        standard_states = {}
        
        for callsign, state in direct_states.items():
            standard_states[callsign] = {
                'callsign': state.callsign,
                'latitude': state.latitude,
                'longitude': state.longitude,
                'altitude': state.altitude_ft,
                'altitude_ft': state.altitude_ft,
                'heading': state.heading_deg,
                'heading_deg': state.heading_deg,
                'speed': state.speed_kt,
                'speed_kt': state.speed_kt,
                'vertical_speed_fpm': state.vertical_speed_fpm,
                'timestamp': state.timestamp,
                'track_angle': state.track_angle,
                'ground_speed_kt': state.ground_speed_kt
            }
        
        return standard_states
    
    def convert_conflicts_to_standard(self, direct_conflicts: List[DirectConflictInfo]) -> List[Dict[str, Any]]:
        """Convert direct conflicts to standard CDR pipeline format"""
        standard_conflicts = []
        
        for conflict in direct_conflicts:
            standard_conflicts.append({
                'aircraft1': conflict.aircraft1,
                'aircraft2': conflict.aircraft2,
                'time_to_conflict': conflict.time_to_conflict,
                'horizontal_distance': conflict.horizontal_distance,
                'vertical_distance': conflict.vertical_distance,
                'conflict_type': conflict.conflict_type,
                'severity': conflict.severity,
                'source': conflict.detection_source,
                'time_to_cpa_minutes': conflict.time_to_conflict / 60.0 if conflict.time_to_conflict > 0 else 0.0
            })
        
        return standard_conflicts


# Global bridge instance
_direct_bridge = None

def get_direct_bridge() -> BlueSkyDirectBridge:
    """Get global direct bridge instance"""
    global _direct_bridge
    if _direct_bridge is None:
        _direct_bridge = BlueSkyDirectBridge()
    return _direct_bridge


def test_direct_bridge():
    """Test the direct bridge functionality"""
    print("🔗 Testing BlueSky Direct Bridge...")
    
    bridge = get_direct_bridge()
    
    if not bridge.is_available():
        print("❌ BlueSky direct access not available")
        return False
    
    print("✅ BlueSky direct access available")
    
    # Test aircraft states
    states = bridge.get_aircraft_states_direct()
    print(f"📊 Found {len(states)} aircraft via direct access")
    
    for callsign, state in states.items():
        print(f"   {callsign}: {state.latitude:.4f},{state.longitude:.4f}, "
              f"{state.altitude_ft:.0f}ft, {state.heading_deg:.1f}°")
    
    # Test conflict detection
    conflicts = bridge.get_conflicts_direct()
    print(f"⚠️  Found {len(conflicts)} conflicts via direct access")
    
    for conflict in conflicts:
        print(f"   {conflict.aircraft1} vs {conflict.aircraft2}: "
              f"h={conflict.horizontal_distance:.1f}NM, v={conflict.vertical_distance:.0f}ft")
    
    return True


if __name__ == "__main__":
    test_direct_bridge()
