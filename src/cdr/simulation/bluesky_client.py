"""
Simplified BlueSky Client using direct BlueSky interface

"""
import bluesky as bs
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import math
import random


@dataclass
class AircraftState:
    """Aircraft state information"""
    id: str
    lat: float
    lon: float
    alt: float
    hdg: float
    tas: float
    vs: float
    x: float = 0.0
    y: float = 0.0


@dataclass
class Destination:
    """Fixed destination waypoint"""
    name: str
    lat: float
    lon: float
    alt: float = 35000  # Default cruise altitude


class SimpleBlueSkyClient:
    """Simplified BlueSky client using direct interface"""
    
    def __init__(self):
        self.aircraft_ids = []
        self.initialized = False
        self.destinations = {}  # Map aircraft_id to Destination
        
    def initialize(self):
        """Initialize BlueSky simulation"""
        if not self.initialized:
            bs.init(mode='sim', detached=True)
            self.initialized = True
    
    def reset(self):
        """Reset simulation"""
        bs.sim.reset()
        self.aircraft_ids = []
        self.destinations = {}
    
    def create_aircraft(self, acid: str, lat: float, lon: float, hdg: float = 0, 
                       alt: float = 10000, spd: float = 250) -> bool:
        """Create aircraft using direct BlueSky interface"""
        # Create aircraft using direct traf.cre - autopilot is automatically enabled
        try:
            success = bs.traf.cre(acid, actype="B738", aclat=lat, aclon=lon, 
                                 achdg=hdg, acalt=alt, acspd=spd)
            
            if success:
                self.aircraft_ids.append(acid)
                return True
            return False
        except Exception as e:
            print(f"Error creating aircraft {acid}: {e}")
            return False
    
    def direct_to(self, acid: str, lat: float, lon: float) -> bool:
        """Direct aircraft to coordinates"""
        # Direct to lat,lon coordinates
        cmd = f"DIRECT {acid},{lat},{lon}"
        return bs.stack.stack(cmd)
    
    def add_waypoint(self, acid: str, name: str, lat: float, lon: float) -> bool:
        """Add waypoint to route"""
        # Correct BlueSky syntax: ADDWPT acid, (wpname/lat,lon),[alt,spd,afterwp]
        cmd = f"ADDWPT {acid},{name},{lat},{lon}"
        return bs.stack.stack(cmd)
    
    def generate_fixed_destination(self, start_lat: float, start_lon: float, 
                                 current_heading: Optional[float] = None,
                                 min_distance_nm: float = 80, max_distance_nm: float = 100) -> Destination:
        """Generate a fixed destination 80-100 NM from starting position, considering current heading"""
        # Random distance within range
        distance_nm = random.uniform(min_distance_nm, max_distance_nm)
        
        if current_heading is not None:
            # Generate destination in the general direction of current heading (±45° spread)
            heading_spread = 45  # degrees
            min_bearing = (current_heading - heading_spread) % 360
            max_bearing = (current_heading + heading_spread) % 360
            
            if min_bearing <= max_bearing:
                bearing_deg = random.uniform(min_bearing, max_bearing)
            else:
                # Handle wrap-around case (e.g., heading 350°, range 305°-35°)
                if random.random() < 0.5:
                    bearing_deg = random.uniform(min_bearing, 360)
                else:
                    bearing_deg = random.uniform(0, max_bearing)
            
            print(f"🧭 Generated destination considering heading {current_heading:.0f}°: bearing {bearing_deg:.0f}°")
        else:
            # Random bearing (0-360 degrees) if no heading provided
            bearing_deg = random.uniform(0, 360)
            print(f"🎲 Generated random destination bearing: {bearing_deg:.0f}°")
        
        # Calculate destination coordinates
        dest_lat, dest_lon = self._calculate_destination_coords(start_lat, start_lon, distance_nm, bearing_deg)
        
        # Generate destination name
        dest_name = f"DEST{random.randint(1000, 9999)}"
        
        return Destination(
            name=dest_name,
            lat=dest_lat,
            lon=dest_lon,
            alt=35000  # Standard cruise altitude
        )
    
    def set_aircraft_destination(self, acid: str, destination: Destination) -> bool:
        """Set fixed destination for aircraft"""
        try:
            # Store destination
            self.destinations[acid] = destination
            
            # Method 1: Add waypoint first, then direct to it
            # Add destination waypoint to BlueSky route
            cmd = f"ADDWPT {acid},{destination.name},{destination.lat:.6f},{destination.lon:.6f}"
            print(f"📍 Adding waypoint: {cmd}")
            result1 = bs.stack.stack(cmd)
            
            # Small delay to ensure waypoint is added
            import time
            time.sleep(0.1)
            
            # Direct aircraft to destination waypoint
            cmd = f"DIRECT {acid},{destination.name}"
            print(f"🎯 Directing to waypoint: {cmd}")
            result2 = bs.stack.stack(cmd)
            
            # Additional verification - try to create a route to the destination
            try:
                cmd = f"DEST {acid},{destination.lat:.6f},{destination.lon:.6f}"
                print(f"🛣️ Setting destination route: {cmd}")
                bs.stack.stack(cmd)
            except:
                pass  # This is optional, don't fail if it doesn't work
            
            print(f"✅ Destination set for {acid}: {destination.name} at {destination.lat:.4f}, {destination.lon:.4f}")
            return True
            
        except Exception as e:
            print(f"❌ Error setting destination for {acid}: {e}")
            # Fallback: Try direct coordinates if waypoint method fails
            try:
                cmd = f"DIRECT {acid},{destination.lat:.6f},{destination.lon:.6f}"
                print(f"🔄 Fallback - Direct to coordinates: {cmd}")
                bs.stack.stack(cmd)
                return True
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                return False
    
    def get_aircraft_destination(self, acid: str) -> Optional[Destination]:
        """Get aircraft's fixed destination"""
        return self.destinations.get(acid)
    
    def _calculate_destination_coords(self, start_lat: float, start_lon: float, 
                                    distance_nm: float, bearing_deg: float) -> Tuple[float, float]:
        """Calculate destination coordinates from start point, distance and bearing"""
        # Convert to radians
        lat1_rad = math.radians(start_lat)
        lon1_rad = math.radians(start_lon)
        bearing_rad = math.radians(bearing_deg)
        
        # Earth radius in nautical miles
        earth_radius_nm = 3440.065
        
        # Angular distance
        angular_distance = distance_nm / earth_radius_nm
        
        # Calculate destination latitude
        lat2_rad = math.asin(
            math.sin(lat1_rad) * math.cos(angular_distance) +
            math.cos(lat1_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
        )
        
        # Calculate destination longitude
        lon2_rad = lon1_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat1_rad),
            math.cos(angular_distance) - math.sin(lat1_rad) * math.sin(lat2_rad)
        )
        
        # Convert back to degrees
        dest_lat = math.degrees(lat2_rad)
        dest_lon = math.degrees(lon2_rad)
        
        return dest_lat, dest_lon
    
    def change_heading(self, acid: str, new_heading: float) -> bool:
        """Change aircraft heading with enhanced command approach"""
        try:
            # Ensure heading is in valid range [0, 360)
            new_heading = new_heading % 360
            
            # Check if aircraft exists before command
            if acid not in bs.traf.id:
                print(f"❌ Aircraft {acid} not found in traffic. Available: {list(bs.traf.id)}")
                return False
            
            # Get current state for debugging
            idx = bs.traf.id.index(acid)
            current_heading = bs.traf.hdg[idx]
            print(f"🧭 {acid}: Current heading {current_heading:.1f}° → Commanding {new_heading:.1f}°")
            
            # First disable all autopilot modes and clear any route
            print(f"🤖 Clearing route and disabling autopilot for {acid}")
            bs.stack.stack(f"DELRTE {acid}")  # Delete route
            bs.sim.step(0.1)
            bs.stack.stack(f"LNAV {acid},OFF")  # Disable lateral navigation
            bs.sim.step(0.1)
            bs.stack.stack(f"VNAV {acid},OFF")  # Disable vertical navigation
            bs.sim.step(0.1)
            
            # Try multiple command formats for better compatibility
            # According to BlueSky command table: HDG acid,hdg (deg,True)
            commands = [
                f"HDG {acid},{int(new_heading)}",  # Correct BlueSky syntax
                f"HDG {acid} {int(new_heading)}",   # Alternative without comma
                f"HEADING {acid},{int(new_heading)}",  # Synonym
                f"{acid} HDG {int(new_heading)}",   # Legacy format
            ]
            
            success = False
            for cmd in commands:
                try:
                    print(f"🔧 Trying command: {cmd}")
                    result = bs.stack.stack(cmd)
                    # Force immediate command processing
                    bs.sim.step(0.1)
                    
                    # Check if heading started changing
                    check_idx = bs.traf.id.index(acid) if acid in bs.traf.id else -1
                    if check_idx >= 0:
                        check_heading = bs.traf.hdg[check_idx]
                        heading_diff = abs(check_heading - current_heading)
                        if heading_diff > 0.2:  # More lenient - any noticeable change
                            print(f"✅ Heading change initiated: {current_heading:.1f}° → {check_heading:.1f}°")
                            success = True
                            break
                        else:
                            print(f"⏳ Heading change in progress: {current_heading:.1f}° → {check_heading:.1f}°")
                except Exception as e:
                    print(f"⚠️ Command failed: {cmd} - {e}")
                    continue
            
            if success:
                # Give more time for the heading change to take effect
                bs.sim.step(3.0)  # Longer time step for gradual heading change
                
                # Verify the heading change is progressing
                final_idx = bs.traf.id.index(acid) if acid in bs.traf.id else -1
                if final_idx >= 0:
                    actual_heading = bs.traf.hdg[final_idx]
                    
                    print(f"📊 {acid}: Commanded {new_heading:.1f}°, Final {actual_heading:.1f}°")
                    
                    # Consider success if heading is changing in the right direction
                    total_change = abs(actual_heading - current_heading)
                    
                    # Calculate expected direction of change
                    heading_diff = (new_heading - current_heading + 180) % 360 - 180
                    actual_diff = (actual_heading - current_heading + 180) % 360 - 180
                    
                    # Check if moving in the right direction
                    same_direction = (heading_diff * actual_diff) > 0
                    
                    if total_change > 0.5 and same_direction:
                        print(f"✅ Heading change successful: moving towards target (Δ{actual_diff:+.1f}°)")
                        return True
                    elif total_change > 0.2:
                        print(f"⏳ Heading change in progress: slow change (Δ{actual_diff:+.1f}°)")
                        return True  # Accept gradual changes
                    else:
                        print(f"⚠️ Minimal heading change: {total_change:.1f}°")
                        return False
            
            return success
                
        except Exception as e:
            print(f"❌ Command failed: {e}")
            return False
    
    def change_altitude(self, acid: str, new_altitude: float) -> bool:
        """Change aircraft altitude"""
        try:
            # Correct BlueSky syntax: ALT acid, alt, [vspd]
            cmd = f"ALT {acid},{int(new_altitude)}"
            
            if acid not in bs.traf.id:
                return False
                
            bs.stack.stack(cmd)
            # BlueSky commands often return None, assume success if no exception
            return True
            
        except Exception as e:
            return False
    
    def change_speed(self, acid: str, new_speed: float) -> bool:
        """Change aircraft speed"""
        try:
            # Correct BlueSky syntax: SPD acid,spd (CAS-kts/Mach)
            cmd = f"SPD {acid},{int(new_speed)}"
            
            if acid not in bs.traf.id:
                return False
                
            bs.stack.stack(cmd)
            # BlueSky commands often return None, assume success if no exception
            return True
            
        except Exception as e:
            return False
    
    def change_vertical_speed(self, acid: str, vertical_speed_fpm: float) -> bool:
        """Change aircraft vertical speed"""
        try:
            # Correct BlueSky syntax: VS acid,vspd (ft/min)
            cmd = f"VS {acid},{int(vertical_speed_fpm)}"
            
            if acid not in bs.traf.id:
                return False
                
            bs.stack.stack(cmd)
            # BlueSky commands often return None, assume success if no exception
            return True
            
        except Exception as e:
            return False
    
    def step_simulation(self, dt: float = 1.0):
        """Step simulation forward"""
        bs.sim.step(dt)
    
    def get_aircraft_state(self, acid: str) -> Optional[AircraftState]:
        """Get aircraft state using direct attribute access"""
        try:
            # Find aircraft index
            if acid not in bs.traf.id:
                return None
                
            idx = bs.traf.id.index(acid)
            
            return AircraftState(
                id=acid,
                lat=bs.traf.lat[idx],
                lon=bs.traf.lon[idx], 
                alt=bs.traf.alt[idx],
                hdg=bs.traf.hdg[idx],
                tas=bs.traf.tas[idx],
                vs=bs.traf.vs[idx],
                x=bs.traf.x[idx] if hasattr(bs.traf, 'x') else 0.0,
                y=bs.traf.y[idx] if hasattr(bs.traf, 'y') else 0.0
            )
        except (ValueError, IndexError):
            return None
    
    def get_all_aircraft_states(self) -> List[AircraftState]:
        """Get states for all aircraft"""
        states = []
        for acid in self.aircraft_ids:
            state = self.get_aircraft_state(acid)
            if state:
                states.append(state)
        return states
    
    def delete_aircraft(self, acid: str) -> bool:
        """Delete aircraft"""
        cmd = f"DEL {acid}"
        success = bs.stack.stack(cmd)
        if success and acid in self.aircraft_ids:
            self.aircraft_ids.remove(acid)
        return success
    
    def get_simulation_time(self) -> float:
        """Get current simulation time"""
        return bs.sim.simt
    
    def set_simulation_speed(self, speed: float = 1.0):
        """Set simulation speed multiplier using FF (fast forward)"""
        cmd = f"FF {speed}"
        bs.stack.stack(cmd)
    
    def set_time_multiplier(self, dtmult: float = 1.0):
        """Set simulation time step multiplier using DTMULT
        DTMULT affects the actual simulation speed, not just display
        Values > 1.0 speed up simulation, < 1.0 slow it down
        """
        cmd = f"DTMULT {dtmult}"
        bs.stack.stack(cmd)
        
    def set_fast_simulation(self, speed_factor: float = 4.0):
        """Set both FF and DTMULT for maximum simulation speed"""
        self.set_simulation_speed(speed_factor)
        self.set_time_multiplier(speed_factor)
        print(f"🚀 Simulation accelerated: FF={speed_factor}x, DTMULT={speed_factor}x")
    
    def pause_simulation(self):
        """Pause simulation"""
        bs.stack.stack("PAUSE")
    
    def resume_simulation(self):
        """Resume simulation"""
        bs.stack.stack("RESUME")
    
    def op(self):
        """Start/operate simulation (alias for resume)"""
        bs.stack.stack("OP")
    
    def hold(self):
        """Hold/pause simulation (alias for pause)"""
        bs.stack.stack("HOLD")
    
    def configure_conflict_detection(self, pz_radius_nm: float = 5.0, 
                                   pz_height_ft: float = 1000, 
                                   lookahead_sec: float = 300) -> bool:
        """Configure BlueSky's native conflict detection parameters
        Default parameters match BlueSky settings.cfg:
        - asas_pzr = 5.0 NM
        - asas_pzh = 1000.0 ft  
        - asas_dtlookahead = 300.0 s (5 min)
        """
        try:
            bs.stack.stack("CDMETHOD ON")
            bs.stack.stack(f"ZONER {pz_radius_nm}")
            bs.stack.stack(f"ZONEDH {pz_height_ft}")
            bs.stack.stack(f"DTLOOK {lookahead_sec}")
            return True
        except Exception as e:
            print(f"Error configuring conflict detection: {e}")
            return False
    
    def get_bluesky_conflicts(self) -> Optional[Dict]:
        """Get conflicts from BlueSky's native ConflictDetection system"""
        try:
            if hasattr(bs.traf, 'cd'):
                cd = bs.traf.cd
                return {
                    'confpairs': getattr(cd, 'confpairs', []),
                    'lospairs': getattr(cd, 'lospairs', []),
                    'confpairs_unique': list(getattr(cd, 'confpairs_unique', set())),
                    'lospairs_unique': list(getattr(cd, 'lospairs_unique', set())),
                    'confpairs_all': getattr(cd, 'confpairs_all', []),
                    'lospairs_all': getattr(cd, 'lospairs_all', []),
                    'inconf': getattr(cd, 'inconf', np.array([])),
                    'tcpamax': getattr(cd, 'tcpamax', np.array([])),
                    'qdr': getattr(cd, 'qdr', np.array([])),
                    'dist': getattr(cd, 'dist', np.array([])),
                    'dcpa': getattr(cd, 'dcpa', np.array([])),
                    'tcpa': getattr(cd, 'tcpa', np.array([])),
                    'tLOS': getattr(cd, 'tLOS', np.array([]))
                }
        except Exception as e:
            print(f"Error accessing BlueSky conflicts: {e}")
        return None
    
    def get_conflict_summary(self) -> Dict:
        """Get a summary of current conflict status"""
        conflicts = self.get_bluesky_conflicts()
        if conflicts:
            return {
                'active_conflicts': len(conflicts['confpairs']) // 2,  # Divide by 2 as each conflict appears twice
                'active_los': len(conflicts['lospairs']) // 2,
                'unique_conflicts': len(conflicts['confpairs_unique']),
                'unique_los': len(conflicts['lospairs_unique']),
                'total_conflicts_detected': len(conflicts['confpairs_all']),
                'total_los_detected': len(conflicts['lospairs_all']),
                'aircraft_in_conflict': conflicts['inconf'].tolist() if len(conflicts['inconf']) > 0 else []
            }
        return {
            'active_conflicts': 0,
            'active_los': 0,
            'unique_conflicts': 0,
            'unique_los': 0,
            'total_conflicts_detected': 0,
            'total_los_detected': 0,
            'aircraft_in_conflict': []
        }



# Alias for compatibility
BlueSkyClient = SimpleBlueSkyClient

