#!/usr/bin/env python3
"""
Fixed BlueSky direct access that handles timing and initialization properly
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.cdr.simulation.bluesky_client import BlueSkyClient, create_thesis_config
from typing import Dict, Optional, List
from dataclasses import dataclass

@dataclass
class AircraftState:
    callsign: str
    latitude: float
    longitude: float
    altitude_ft: float
    heading_deg: float
    speed_kt: float
    vertical_speed_fpm: float
    timestamp: float

class FixedBlueSkyAccess:
    """Fixed BlueSky access that properly handles initialization timing"""
    
    def __init__(self, bluesky_client):
        self.client = bluesky_client
        self.sim = None
        self.traf = None
        self.bs_module = None
        
    def _get_traffic_object(self):
        """Get traffic object with proper timing and fallback methods"""
        
        # Method 1: Try to get fresh BlueSky imports each time
        try:
            import bluesky as bs
            self.bs_module = bs
            
            # Debug what's available
            print(f"🔍 BlueSky module: {bs}")
            print(f"🔍 Has sim attr: {hasattr(bs, 'sim')}")
            print(f"🔍 Has traf attr: {hasattr(bs, 'traf')}")
            
            if hasattr(bs, 'sim'):
                print(f"🔍 bs.sim: {bs.sim}")
                print(f"🔍 bs.sim type: {type(bs.sim)}")
                
            if hasattr(bs, 'traf'):
                print(f"🔍 bs.traf: {bs.traf}")
                print(f"🔍 bs.traf type: {type(bs.traf)}")
                
                # Check if traf has the attributes we need
                if bs.traf is not None:
                    print(f"🔍 bs.traf attributes: {dir(bs.traf)[:10]}...")
                    if hasattr(bs.traf, 'ntraf'):
                        print(f"🔍 bs.traf.ntraf: {bs.traf.ntraf}")
                        return bs.traf
                    else:
                        print("⚠️ bs.traf missing ntraf attribute")
                else:
                    print("⚠️ bs.traf is None")
                    
        except ImportError as e:
            print(f"❌ Cannot import bluesky: {e}")
        except Exception as e:
            print(f"❌ Error accessing BlueSky modules: {e}")
        
        # Method 2: Try accessing through client's attributes
        try:
            if hasattr(self.client, 'sim') and self.client.sim:
                print(f"🔍 Client sim: {self.client.sim}")
                if hasattr(self.client, 'traf') and self.client.traf:
                    print(f"🔍 Client traf: {self.client.traf}")
                    return self.client.traf
        except Exception as e:
            print(f"❌ Error accessing client traffic: {e}")
            
        # Method 3: Try to initialize BlueSky objects manually
        try:
            print("🔄 Attempting to initialize BlueSky objects...")
            
            # Send a command to ensure BlueSky is ready
            self.client._send_command("IC", expect_response=True)
            time.sleep(1)
            
            # Try importing again after initialization
            import bluesky as bs
            if hasattr(bs, 'traf') and bs.traf is not None:
                print("✅ BlueSky traf available after initialization")
                return bs.traf
                
        except Exception as e:
            print(f"❌ Manual initialization failed: {e}")
        
        return None

    def wait_for_aircraft_creation(self, expected_callsign: str, timeout: float = 10.0) -> bool:
        """Wait for aircraft to be actually created in the traffic system"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            traf = self._get_traffic_object()
            
            if traf is not None:
                try:
                    if hasattr(traf, 'ntraf') and hasattr(traf, 'id'):
                        n_aircraft = traf.ntraf if traf.ntraf else 0
                        
                        if n_aircraft > 0:
                            # Check if our aircraft is in the list
                            if hasattr(traf.id, '__iter__'):
                                aircraft_ids = list(traf.id) if traf.id else []
                                print(f"🔍 Found {n_aircraft} aircraft: {aircraft_ids}")
                                
                                if expected_callsign in aircraft_ids:
                                    print(f"✅ Aircraft {expected_callsign} found in traffic system")
                                    return True
                            else:
                                print(f"⚠️ traf.id is not iterable: {type(traf.id)}")
                        else:
                            print(f"⚠️ No aircraft in traffic system (ntraf = {n_aircraft})")
                    else:
                        print("⚠️ Traffic object missing required attributes")
                        
                except Exception as e:
                    print(f"❌ Error checking aircraft: {e}")
            
            # Wait a bit before retrying
            time.sleep(0.5)
            
        print(f"❌ Timeout waiting for aircraft {expected_callsign}")
        return False

    def get_aircraft_states(self) -> Dict[str, AircraftState]:
        """Get aircraft states with better error handling and debugging"""
        states = {}
        
        print("🔍 Getting aircraft states...")
        
        # Get fresh traffic object
        traf = self._get_traffic_object()
        
        if traf is None:
            print("❌ No traffic object available")
            return states
            
        try:
            # Check if traffic has aircraft
            if not hasattr(traf, 'ntraf'):
                print("❌ Traffic object missing ntraf attribute")
                return states
                
            n_aircraft = traf.ntraf if traf.ntraf else 0
            print(f"📊 Traffic object has {n_aircraft} aircraft")
            
            if n_aircraft == 0:
                print("⚠️ No aircraft in simulation")
                return states
            
            # Check required attributes
            required_attrs = ['id', 'lat', 'lon', 'alt', 'hdg', 'tas', 'vs']
            missing_attrs = []
            
            for attr in required_attrs:
                if not hasattr(traf, attr):
                    missing_attrs.append(attr)
                else:
                    attr_value = getattr(traf, attr)
                    print(f"🔍 traf.{attr}: {type(attr_value)}, length: {len(attr_value) if hasattr(attr_value, '__len__') else 'N/A'}")
            
            if missing_attrs:
                print(f"❌ Missing traffic attributes: {missing_attrs}")
                return states
            
            # Extract data for each aircraft
            current_time = time.time()
            
            print(f"🔄 Processing {n_aircraft} aircraft...")
            
            for i in range(n_aircraft):
                try:
                    # Get aircraft ID
                    if hasattr(traf.id, '__getitem__'):
                        if i < len(traf.id):
                            callsign = str(traf.id[i])
                        else:
                            print(f"⚠️ Index {i} out of range for aircraft IDs")
                            continue
                    else:
                        callsign = f"AC{i}"
                    
                    print(f"🔄 Processing aircraft {i}: {callsign}")
                    
                    # Extract position and state data with bounds checking
                    def safe_get(attr_name, index, default=0.0):
                        try:
                            attr = getattr(traf, attr_name)
                            if hasattr(attr, '__getitem__') and index < len(attr):
                                return float(attr[index])
                            else:
                                print(f"⚠️ Index {index} out of range for {attr_name}")
                                return default
                        except Exception as e:
                            print(f"❌ Error getting {attr_name}[{index}]: {e}")
                            return default
                    
                    lat = safe_get('lat', i, 0.0)
                    lon = safe_get('lon', i, 0.0)
                    alt_m = safe_get('alt', i, 0.0)
                    hdg = safe_get('hdg', i, 0.0)
                    tas_ms = safe_get('tas', i, 0.0)
                    vs_ms = safe_get('vs', i, 0.0)
                    
                    # Convert units
                    alt_ft = alt_m * 3.28084  # meters to feet
                    speed_kt = tas_ms * 1.94384  # m/s to knots
                    vs_fpm = vs_ms * 196.85  # m/s to ft/min
                    
                    # Create aircraft state
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
                    print(f"✅ {callsign}: lat={lat:.6f}, lon={lon:.6f}, alt={alt_ft:.0f}ft, hdg={hdg:.1f}°, spd={speed_kt:.0f}kt")
                    
                except Exception as e:
                    print(f"❌ Error processing aircraft {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            return states
            
        except Exception as e:
            print(f"❌ Error in get_aircraft_states: {e}")
            import traceback
            traceback.print_exc()
            return states

def enhanced_test_with_proper_timing():
    """Enhanced test that properly handles BlueSky timing"""
    
    print("=== Enhanced BlueSky Access with Proper Timing ===")
    
    # Create BlueSky client
    bluesky_config = create_thesis_config()
    bluesky_client = BlueSkyClient(bluesky_config)
    
    try:
        # Connect to BlueSky
        print("Connecting to BlueSky...")
        if not bluesky_client.connect():
            print("❌ Failed to connect to BlueSky")
            return
        
        print("✅ Connected to BlueSky")
        
        # Initialize fixed access
        fixed_access = FixedBlueSkyAccess(bluesky_client)
        
        # Give BlueSky time to fully initialize
        print("⏳ Waiting for BlueSky to fully initialize...")
        time.sleep(3)
        
        # Test traffic access before creating aircraft
        print("\n=== Testing Traffic Access (Before Aircraft) ===")
        traf = fixed_access._get_traffic_object()
        if traf:
            print(f"✅ Traffic object accessible: {type(traf)}")
        else:
            print("❌ Cannot access traffic object")
            
        # Reset and create aircraft
        print("\n=== Creating Aircraft ===")
        bluesky_client._send_command("RESET")
        time.sleep(2)  # Give more time for reset
        
        # Create test aircraft
        success = bluesky_client.create_aircraft(
            callsign="TEST01",
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
        
        print("✅ Aircraft creation command sent")
        
        # Start simulation
        bluesky_client.op()
        time.sleep(1)
        
        # Wait for aircraft to actually appear in the system
        print("\n=== Waiting for Aircraft in System ===")
        if fixed_access.wait_for_aircraft_creation("TEST01", timeout=15.0):
            print("✅ Aircraft confirmed in traffic system")
        else:
            print("❌ Aircraft not found in traffic system")
            # Let's debug what's in the system
            print("\n=== Debugging Traffic System ===")
            traf = fixed_access._get_traffic_object()
            if traf:
                try:
                    n_aircraft = getattr(traf, 'ntraf', 0)
                    print(f"🔍 Total aircraft in system: {n_aircraft}")
                    
                    if hasattr(traf, 'id') and traf.id is not None:
                        if hasattr(traf.id, '__iter__'):
                            ids = list(traf.id)
                            print(f"🔍 Aircraft IDs: {ids}")
                        else:
                            print(f"🔍 traf.id type: {type(traf.id)}")
                    else:
                        print("🔍 No traf.id available")
                        
                except Exception as e:
                    print(f"❌ Error debugging traffic: {e}")
        
        # Try to get aircraft states
        print("\n=== Getting Aircraft States ===")
        initial_states = fixed_access.get_aircraft_states()
        
        if initial_states:
            print(f"🎉 SUCCESS! Retrieved {len(initial_states)} aircraft states")
            
            for callsign, state in initial_states.items():
                print(f"   {callsign}:")
                print(f"     lat={state.latitude:.6f}, lon={state.longitude:.6f}")
                print(f"     alt={state.altitude_ft:.0f}ft, hdg={state.heading_deg:.1f}°")
                print(f"     spd={state.speed_kt:.0f}kt")
                
            # Test movement
            print("\n=== Movement Test ===")
            print("Running simulation for 10 seconds...")
            time.sleep(10)
            
            new_states = fixed_access.get_aircraft_states()
            
            if new_states and initial_states:
                for callsign in initial_states:
                    if callsign in new_states:
                        initial = initial_states[callsign]
                        current = new_states[callsign]
                        
                        lat_diff = abs(current.latitude - initial.latitude)
                        lon_diff = abs(current.longitude - initial.longitude)
                        
                        print(f"   {callsign} movement:")
                        print(f"     Initial: lat={initial.latitude:.6f}, lon={initial.longitude:.6f}")
                        print(f"     Current: lat={current.latitude:.6f}, lon={current.longitude:.6f}")
                        print(f"     Change:  Δlat={lat_diff:.6f}, Δlon={lon_diff:.6f}")
                        
                        if lat_diff > 0.0001 or lon_diff > 0.0001:
                            print(f"   🎉 SUCCESS: {callsign} is moving!")
                        else:
                            print(f"   ❌ No movement detected for {callsign}")
        else:
            print("❌ No aircraft states retrieved")
            
            # Final debugging attempt
            print("\n=== Final Debug Attempt ===")
            
            # Try using alternative commands
            alt_commands = ["LISTAC", "POS *", "TRAIL *", "INFO"]
            
            for cmd in alt_commands:
                try:
                    print(f"\n--- Trying {cmd} ---")
                    response = bluesky_client._send_command(cmd, expect_response=True, timeout=3.0)
                    if response and len(response) > 10:
                        print(f"✅ {cmd} response: {response[:200]}...")
                    else:
                        print(f"❌ {cmd} no response")
                except Exception as e:
                    print(f"❌ {cmd} error: {e}")
    
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        bluesky_client.disconnect()
        print("\n✅ Disconnected from BlueSky")

if __name__ == "__main__":
    enhanced_test_with_proper_timing()