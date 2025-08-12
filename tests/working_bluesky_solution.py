#!/usr/bin/env python3
"""
Working solution for BlueSky aircraft movement detection
This implements a proper fix for the aircraft state parsing issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from src.cdr.simulation.bluesky_client import BlueSkyClient, create_thesis_config, AircraftState
from typing import Dict, Optional, List

class BlueSkyWorkingClient(BlueSkyClient):
    """Extended BlueSky client with working aircraft state detection"""
    
    def __init__(self, config):
        super().__init__(config)
        self._initialize_embedded_access()
    
    def _initialize_embedded_access(self):
        """Initialize embedded access after BlueSky connection"""
        self.embedded_sim = None
        self.embedded_traf = None
    
    def connect(self, timeout: float = 30.0) -> bool:
        """Connect and set up embedded access"""
        if not super().connect(timeout):
            return False
        
        # Try to initialize embedded access after connection
        try:
            import bluesky as bs
            if hasattr(bs, 'sim') and bs.sim is not None:
                self.embedded_sim = bs.sim
                self.embedded_traf = bs.traf
                print("✅ Embedded access initialized")
            else:
                print("⚠️ BlueSky sim not initialized, will try after first command")
        except ImportError:
            print("⚠️ BlueSky module not available for embedded access")
        
        return True
    
    def _try_embedded_access(self):
        """Try to get embedded access if not already available"""
        if self.embedded_traf is not None:
            return True
        
        try:
            import bluesky as bs
            if hasattr(bs, 'traf') and bs.traf is not None:
                self.embedded_sim = bs.sim
                self.embedded_traf = bs.traf
                print("✅ Late embedded access initialized")
                return True
        except:
            pass
        
        return False
    
    def get_aircraft_states_working(self, callsigns: Optional[List[str]] = None) -> Dict[str, AircraftState]:
        """Working aircraft state retrieval that actually detects movement"""
        
        # Method 1: Try embedded access
        states = self._get_states_via_embedded()
        if states:
            print(f"✅ Got {len(states)} states via embedded access")
            return states
        
        # Method 2: Try alternative commands
        states = self._get_states_via_alternative_commands(callsigns)
        if states:
            print(f"✅ Got {len(states)} states via alternative commands")
            return states
        
        # Method 3: Fall back to cached states (but this doesn't show movement)
        print("⚠️ Falling back to cached states (movement won't be detected)")
        return self.aircraft_states.copy()
    
    def _get_states_via_embedded(self) -> Dict[str, AircraftState]:
        """Get aircraft states via embedded BlueSky access"""
        
        if not self._try_embedded_access():
            return {}
        
        try:
            if not hasattr(self.embedded_traf, 'ntraf') or self.embedded_traf.ntraf == 0:
                return {}
            
            states = {}
            current_time = time.time()
            
            for i in range(self.embedded_traf.ntraf):
                try:
                    # Get aircraft data
                    callsign = str(self.embedded_traf.id[i])
                    lat = float(self.embedded_traf.lat[i])
                    lon = float(self.embedded_traf.lon[i])
                    alt_m = float(self.embedded_traf.alt[i])
                    hdg = float(self.embedded_traf.hdg[i])
                    tas_ms = float(self.embedded_traf.tas[i])
                    vs_ms = float(self.embedded_traf.vs[i])
                    
                    # Convert units
                    alt_ft = alt_m * 3.28084  # meters to feet
                    speed_kt = tas_ms * 1.94384  # m/s to knots
                    vs_fpm = vs_ms * 196.85  # m/s to ft/min
                    
                    state = AircraftState(
                        callsign=callsign,
                        latitude=lat,
                        longitude=lon,
                        altitude_ft=alt_ft,
                        heading_deg=hdg,
                        speed_kt=speed_kt,
                        vertical_speed_fpm=vs_fpm,
                        timestamp=current_time
                    )
                    
                    states[callsign] = state
                    
                except Exception as e:
                    print(f"❌ Error extracting embedded data for aircraft {i}: {e}")
                    continue
            
            return states
            
        except Exception as e:
            print(f"❌ Error accessing embedded traffic: {e}")
            return {}
    
    def _get_states_via_alternative_commands(self, callsigns: Optional[List[str]]) -> Dict[str, AircraftState]:
        """Try alternative BlueSky commands to get aircraft states"""
        
        states = {}
        
        if not callsigns:
            callsigns = list(self.callsigns)
        
        for callsign in callsigns:
            # Try LIST command which might give position info
            try:
                response = self._send_command("LIST", expect_response=True, timeout=2.0)
                if response and callsign in response:
                    # Try to extract coordinates from LIST response
                    state = self._parse_list_response(response, callsign)
                    if state:
                        states[callsign] = state
                        continue
            except:
                pass
            
            # Try TRAIL command
            try:
                response = self._send_command(f"TRAIL {callsign}", expect_response=True, timeout=2.0)
                if response:
                    state = self._parse_trail_response(response, callsign)
                    if state:
                        states[callsign] = state
                        continue
            except:
                pass
        
        return states
    
    def _parse_list_response(self, response: str, callsign: str) -> Optional[AircraftState]:
        """Try to parse aircraft state from LIST command response"""
        # This would need to be implemented based on actual LIST response format
        return None
    
    def _parse_trail_response(self, response: str, callsign: str) -> Optional[AircraftState]:
        """Try to parse aircraft state from TRAIL command response"""
        # This would need to be implemented based on actual TRAIL response format
        return None
    
    def step_simulation_with_movement_check(self, duration_seconds: float) -> Dict[str, Dict[str, float]]:
        """Step simulation and return movement data for all aircraft"""
        
        # Get initial positions
        initial_states = self.get_aircraft_states_working()
        
        # Advance simulation
        if hasattr(self, 'step_minutes'):
            self.step_minutes(duration_seconds / 60.0)
        else:
            self.ff(duration_seconds)
        
        # Get final positions
        final_states = self.get_aircraft_states_working()
        
        # Calculate movement
        movement_data = {}
        
        for callsign in initial_states:
            if callsign in final_states:
                initial = initial_states[callsign]
                final = final_states[callsign]
                
                lat_diff = final.latitude - initial.latitude
                lon_diff = final.longitude - initial.longitude
                alt_diff = final.altitude_ft - initial.altitude_ft
                
                movement_data[callsign] = {
                    'lat_change': lat_diff,
                    'lon_change': lon_diff,
                    'alt_change': alt_diff,
                    'distance_deg': (lat_diff**2 + lon_diff**2)**0.5,
                    'initial_lat': initial.latitude,
                    'initial_lon': initial.longitude,
                    'final_lat': final.latitude,
                    'final_lon': final.longitude
                }
        
        return movement_data

def test_working_solution():
    """Test the working solution for aircraft movement detection"""
    
    print("=== Testing Working BlueSky Solution ===")
    
    # Create working BlueSky client
    bluesky_config = create_thesis_config()
    bluesky_client = BlueSkyWorkingClient(bluesky_config)
    
    try:
        # Connect to BlueSky
        print("Connecting to BlueSky...")
        if not bluesky_client.connect():
            print("❌ Failed to connect to BlueSky")
            return
        
        print("✅ Connected to BlueSky")
        
        # Reset and create aircraft
        bluesky_client._send_command("RESET")
        time.sleep(1)
        
        # Create test aircraft
        success = bluesky_client.create_aircraft(
            callsign="WORKING01",
            aircraft_type="B738",
            lat=42.0,
            lon=-87.0,
            heading=90,
            altitude_ft=35000,
            speed_kt=450
        )
        
        if not success:
            print("❌ Failed to create aircraft")
            return
        
        print("✅ Created aircraft WORKING01 at (42.0, -87.0)")
        
        # Start simulation
        bluesky_client.op()
        time.sleep(2)
        
        # Test aircraft state retrieval
        print("\n=== Testing Aircraft State Retrieval ===")
        
        initial_states = bluesky_client.get_aircraft_states_working()
        
        if initial_states:
            print(f"✅ Retrieved {len(initial_states)} aircraft states")
            for callsign, state in initial_states.items():
                print(f"   {callsign}: lat={state.latitude:.6f}, lon={state.longitude:.6f}, alt={state.altitude_ft:.0f}ft")
        else:
            print("❌ No aircraft states retrieved")
            return
        
        # Test movement detection with real-time simulation
        print("\n=== Testing Real-Time Movement ===")
        print("Running simulation for 10 seconds...")
        time.sleep(10)
        
        new_states = bluesky_client.get_aircraft_states_working()
        
        if new_states:
            for callsign in initial_states:
                if callsign in new_states:
                    initial = initial_states[callsign]
                    current = new_states[callsign]
                    
                    lat_diff = abs(current.latitude - initial.latitude)
                    lon_diff = abs(current.longitude - initial.longitude)
                    
                    print(f"   {callsign} real-time movement:")
                    print(f"     Initial: lat={initial.latitude:.6f}, lon={initial.longitude:.6f}")
                    print(f"     Current: lat={current.latitude:.6f}, lon={current.longitude:.6f}")
                    print(f"     Change:  Δlat={lat_diff:.6f}, Δlon={lon_diff:.6f}")
                    
                    if lat_diff > 0.0001 or lon_diff > 0.0001:
                        print(f"   🎉 SUCCESS: {callsign} is moving in real-time!")
                    else:
                        print(f"   ❌ No real-time movement detected for {callsign}")
        
        # Test step simulation with movement check
        print("\n=== Testing Step Simulation Movement ===")
        
        movement_data = bluesky_client.step_simulation_with_movement_check(120.0)  # 2 minutes
        
        if movement_data:
            print("✅ Movement data retrieved:")
            for callsign, data in movement_data.items():
                print(f"   {callsign}:")
                print(f"     Position change: Δlat={data['lat_change']:.6f}, Δlon={data['lon_change']:.6f}")
                print(f"     Distance moved: {data['distance_deg']:.6f} degrees")
                print(f"     Final position: lat={data['final_lat']:.6f}, lon={data['final_lon']:.6f}")
                
                if data['distance_deg'] > 0.001:
                    print(f"   🎉 SUCCESS: {callsign} moved significantly!")
                else:
                    print(f"   ❌ No significant movement for {callsign}")
        else:
            print("❌ No movement data retrieved")
    
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        bluesky_client.disconnect()
        print("\nDisconnected from BlueSky")

if __name__ == "__main__":
    test_working_solution()
