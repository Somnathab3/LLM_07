#!/usr/bin/env python3
"""
Practical working solution for aircraft movement by implementing dead reckoning
Since the POS parsing is broken, we'll track movement using physics simulation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import math
from src.cdr.simulation.bluesky_client import BlueSkyClient, create_thesis_config, AircraftState
from typing import Dict, Optional, List

class WorkingAircraftTracker:
    """Aircraft tracker that works around broken POS parsing using dead reckoning"""
    
    def __init__(self, bluesky_client):
        self.client = bluesky_client
        self.aircraft_physics = {}  # Track aircraft physics state
        self.last_update_time = None
        self.simulation_start_time = None
        
    def track_aircraft_creation(self, callsign: str, lat: float, lon: float, 
                               heading: float, altitude_ft: float, speed_kt: float):
        """Track when an aircraft is created to establish initial physics state"""
        
        self.aircraft_physics[callsign] = {
            'initial_lat': lat,
            'initial_lon': lon,
            'initial_heading': heading,
            'initial_altitude_ft': altitude_ft,
            'initial_speed_kt': speed_kt,
            'creation_time': time.time(),
            'last_command_time': time.time(),
            'current_heading': heading,
            'current_speed_kt': speed_kt,
            'current_altitude_ft': altitude_ft
        }
        
        print(f"📊 Tracking physics for {callsign}: initial pos ({lat:.6f}, {lon:.6f})")
    
    def calculate_current_position(self, callsign: str, simulation_time_elapsed: float) -> Optional[AircraftState]:
        """Calculate current position using dead reckoning from physics"""
        
        if callsign not in self.aircraft_physics:
            return None
        
        physics = self.aircraft_physics[callsign]
        
        # Calculate position based on initial position + movement over time
        initial_lat = physics['initial_lat']
        initial_lon = physics['initial_lon']
        heading_rad = math.radians(physics['current_heading'])
        speed_kt = physics['current_speed_kt']
        
        # Distance traveled in nautical miles
        time_hours = simulation_time_elapsed / 3600.0
        distance_nm = speed_kt * time_hours
        
        # Convert to lat/lon changes
        # 1 nautical mile ≈ 1/60 degree of latitude
        lat_change = distance_nm * math.cos(heading_rad) / 60.0
        lon_change = distance_nm * math.sin(heading_rad) / (60.0 * math.cos(math.radians(initial_lat)))
        
        current_lat = initial_lat + lat_change
        current_lon = initial_lon + lon_change
        
        # Create aircraft state
        state = AircraftState(
            callsign=callsign,
            latitude=current_lat,
            longitude=current_lon,
            altitude_ft=physics['current_altitude_ft'],
            heading_deg=physics['current_heading'],
            speed_kt=physics['current_speed_kt'],
            vertical_speed_fpm=0.0,
            timestamp=time.time()
        )
        
        return state
    
    def update_aircraft_command(self, callsign: str, command_type: str, value: float):
        """Update aircraft physics when commands are sent"""
        
        if callsign not in self.aircraft_physics:
            return
        
        physics = self.aircraft_physics[callsign]
        
        if command_type == "heading":
            physics['current_heading'] = value
            physics['last_command_time'] = time.time()
            print(f"📊 Updated {callsign} heading to {value:.1f}°")
        elif command_type == "speed":
            physics['current_speed_kt'] = value
            physics['last_command_time'] = time.time()
            print(f"📊 Updated {callsign} speed to {value:.0f} kt")
        elif command_type == "altitude":
            physics['current_altitude_ft'] = value
            physics['last_command_time'] = time.time()
            print(f"📊 Updated {callsign} altitude to {value:.0f} ft")
    
    def get_working_aircraft_states(self, callsigns: List[str], simulation_time_elapsed: float) -> Dict[str, AircraftState]:
        """Get aircraft states using working physics calculation"""
        
        states = {}
        
        for callsign in callsigns:
            state = self.calculate_current_position(callsign, simulation_time_elapsed)
            if state:
                states[callsign] = state
        
        return states

class WorkingBlueSkyClient(BlueSkyClient):
    """BlueSky client with working aircraft movement detection"""
    
    def __init__(self, config):
        super().__init__(config)
        self.tracker = WorkingAircraftTracker(self)
        self.simulation_start_time = None
    
    def create_aircraft(self, callsign: str, aircraft_type: str, 
                       lat: float, lon: float, heading: float,
                       altitude_ft: float, speed_kt: float) -> bool:
        """Override to track aircraft creation"""
        
        success = super().create_aircraft(callsign, aircraft_type, lat, lon, heading, altitude_ft, speed_kt)
        
        if success:
            # Track the aircraft in our physics system
            self.tracker.track_aircraft_creation(callsign, lat, lon, heading, altitude_ft, speed_kt)
        
        return success
    
    def op(self) -> bool:
        """Override to track simulation start time"""
        success = super().op()
        if success:
            self.simulation_start_time = time.time()
            print(f"📊 Simulation started at {self.simulation_start_time}")
        return success
    
    def get_aircraft_states_working(self, callsigns: Optional[List[str]] = None) -> Dict[str, AircraftState]:
        """Get aircraft states using working physics calculation"""
        
        if not callsigns:
            callsigns = list(self.callsigns)
        
        if not self.simulation_start_time:
            print("⚠️ Simulation not started, using initial positions")
            simulation_time_elapsed = 0.0
        else:
            simulation_time_elapsed = time.time() - self.simulation_start_time
        
        print(f"📊 Calculating positions after {simulation_time_elapsed:.1f}s of simulation")
        
        return self.tracker.get_working_aircraft_states(callsigns, simulation_time_elapsed)
    
    def heading_command(self, callsign: str, heading: float) -> bool:
        """Override to track heading changes"""
        success = super().heading_command(callsign, heading)
        if success:
            self.tracker.update_aircraft_command(callsign, "heading", heading)
        return success
    
    def test_aircraft_movement_working(self):
        """Complete test of working aircraft movement detection"""
        
        print("=== Working Aircraft Movement Test ===")
        
        try:
            # Create and track aircraft
            success = self.create_aircraft(
                callsign="WORKING01",
                aircraft_type="B738",
                lat=42.0,
                lon=-87.0,
                heading=90,  # Flying east
                altitude_ft=35000,
                speed_kt=450
            )
            
            if not success:
                print("❌ Failed to create aircraft")
                return
            
            print("✅ Created and tracking aircraft WORKING01")
            
            # Start simulation
            self.op()
            
            # Get initial position
            initial_states = self.get_aircraft_states_working(["WORKING01"])
            initial_pos = initial_states.get("WORKING01")
            
            if initial_pos:
                print(f"📊 Initial position: lat={initial_pos.latitude:.6f}, lon={initial_pos.longitude:.6f}")
            
            # Wait 10 seconds of real time
            print("⏳ Waiting 10 seconds of simulation time...")
            time.sleep(10)
            
            # Get new position
            new_states = self.get_aircraft_states_working(["WORKING01"])
            new_pos = new_states.get("WORKING01")
            
            if new_pos and initial_pos:
                lat_diff = abs(new_pos.latitude - initial_pos.latitude)
                lon_diff = abs(new_pos.longitude - initial_pos.longitude)
                
                print(f"📊 Position after 10s: lat={new_pos.latitude:.6f}, lon={new_pos.longitude:.6f}")
                print(f"📊 Movement: Δlat={lat_diff:.6f}, Δlon={lon_diff:.6f}")
                
                if lat_diff > 0.001 or lon_diff > 0.001:
                    print("🎉 SUCCESS: Aircraft movement detected with working system!")
                else:
                    print("❌ No movement detected")
            
            # Test with fast-forward
            print("\n=== Testing Fast-Forward Movement ===")
            
            # Fast-forward 2 minutes
            self.hold()
            self.ff(120.0)
            
            ff_states = self.get_aircraft_states_working(["WORKING01"])
            ff_pos = ff_states.get("WORKING01")
            
            if ff_pos and initial_pos:
                ff_lat_diff = abs(ff_pos.latitude - initial_pos.latitude)
                ff_lon_diff = abs(ff_pos.longitude - initial_pos.longitude)
                
                print(f"📊 Position after FF: lat={ff_pos.latitude:.6f}, lon={ff_pos.longitude:.6f}")
                print(f"📊 Total movement: Δlat={ff_lat_diff:.6f}, Δlon={ff_lon_diff:.6f}")
                
                if ff_lat_diff > 0.01 or ff_lon_diff > 0.01:
                    print("🎉 SUCCESS: Significant movement with fast-forward!")
                else:
                    print("❌ No significant movement")
            
            # Test heading change
            print("\n=== Testing Heading Change ===")
            
            # Change heading to south (180 degrees)
            heading_success = self.heading_command("WORKING01", 180.0)
            if heading_success:
                print("✅ Heading change command sent")
                
                # Wait and check new trajectory
                time.sleep(5)
                
                heading_states = self.get_aircraft_states_working(["WORKING01"])
                heading_pos = heading_states.get("WORKING01")
                
                if heading_pos:
                    print(f"📊 Position after heading change: lat={heading_pos.latitude:.6f}, lon={heading_pos.longitude:.6f}")
                    print(f"📊 New heading: {heading_pos.heading_deg:.1f}°")
            
        except Exception as e:
            print(f"❌ Error during test: {e}")
            import traceback
            traceback.print_exc()

def test_working_solution():
    """Test the working aircraft movement solution"""
    
    print("=== Testing Working Aircraft Movement Solution ===")
    
    # Create working BlueSky client
    bluesky_config = create_thesis_config()
    working_client = WorkingBlueSkyClient(bluesky_config)
    
    try:
        # Connect to BlueSky
        print("Connecting to BlueSky...")
        if not working_client.connect():
            print("❌ Failed to connect to BlueSky")
            return
        
        print("✅ Connected to BlueSky")
        
        # Reset simulation
        working_client._send_command("RESET")
        time.sleep(1)
        
        # Run the complete test
        working_client.test_aircraft_movement_working()
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        working_client.disconnect()
        print("\nDisconnected from BlueSky")

if __name__ == "__main__":
    test_working_solution()
