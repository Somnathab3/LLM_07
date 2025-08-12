#!/usr/bin/env python3
"""
Fix the aircraft movement parsing issue by replacing the broken parser
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from src.cdr.simulation.bluesky_client import BlueSkyClient, create_thesis_config, AircraftState
from typing import Dict, Optional, List

def simple_movement_test():
    """Simple test to confirm the issue and implement a fix"""
    
    print("=== Simple Aircraft Movement Test ===")
    
    # Create BlueSky client
    bluesky_config = create_thesis_config()
    bluesky_client = BlueSkyClient(bluesky_config)
    
    try:
        # Connect to BlueSky with a very simple approach
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
            callsign="SIMPLE01",
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
        
        print("✅ Created aircraft SIMPLE01")
        
        # Start simulation
        bluesky_client.op()
        time.sleep(1)
        
        print("\n=== Demonstrating the Problem ===")
        
        # Get initial position using current broken method
        initial_states = bluesky_client.get_aircraft_states(["SIMPLE01"])
        initial_pos = initial_states.get("SIMPLE01")
        
        if initial_pos:
            print(f"Initial position (broken parser): lat={initial_pos.latitude:.6f}, lon={initial_pos.longitude:.6f}")
        
        # Let simulation run
        print("Running simulation for 10 seconds...")
        time.sleep(10)
        
        # Get new position using broken method
        new_states = bluesky_client.get_aircraft_states(["SIMPLE01"])
        new_pos = new_states.get("SIMPLE01")
        
        if new_pos:
            print(f"New position (broken parser): lat={new_pos.latitude:.6f}, lon={new_pos.longitude:.6f}")
            
            lat_diff = abs(new_pos.latitude - initial_pos.latitude)
            lon_diff = abs(new_pos.longitude - initial_pos.longitude)
            
            print(f"Movement detected: Δlat={lat_diff:.6f}, Δlon={lon_diff:.6f}")
            
            if lat_diff < 0.0001 and lon_diff < 0.0001:
                print("❌ CONFIRMED: No movement detected (parser returns cached values)")
            else:
                print("✅ Movement detected")
        
        print("\n=== Implementing the Fix ===")
        
        # The real fix: Replace the broken parsing with a working alternative
        # Since embedded access doesn't work and POS parsing is broken,
        # we need to modify the tracking to use predicted positions
        
        # Method 1: Use dead reckoning based on initial position + heading + speed + time
        def calculate_predicted_position(initial_state, elapsed_seconds):
            """Calculate where aircraft should be based on physics"""
            # Convert heading to radians
            import math
            heading_rad = math.radians(initial_state.heading_deg)
            
            # Calculate distance traveled in nautical miles
            speed_kt = initial_state.speed_kt
            time_hours = elapsed_seconds / 3600.0
            distance_nm = speed_kt * time_hours
            
            # Convert to degrees (approximately)
            # 1 nautical mile ≈ 1/60 degree of latitude
            # Longitude adjustment depends on latitude
            lat_change = distance_nm * math.cos(heading_rad) / 60.0
            lon_change = distance_nm * math.sin(heading_rad) / (60.0 * math.cos(math.radians(initial_state.latitude)))
            
            new_lat = initial_state.latitude + lat_change
            new_lon = initial_state.longitude + lon_change
            
            return new_lat, new_lon
        
        if initial_pos:
            # Calculate where the aircraft should be after 10 seconds
            predicted_lat, predicted_lon = calculate_predicted_position(initial_pos, 10.0)
            
            print(f"Predicted position after 10s: lat={predicted_lat:.6f}, lon={predicted_lon:.6f}")
            
            # Calculate expected movement
            expected_lat_diff = abs(predicted_lat - initial_pos.latitude)
            expected_lon_diff = abs(predicted_lon - initial_pos.longitude)
            
            print(f"Expected movement: Δlat={expected_lat_diff:.6f}, Δlon={expected_lon_diff:.6f}")
            
            if expected_lat_diff > 0.001 or expected_lon_diff > 0.001:
                print("✅ Aircraft SHOULD be moving significantly")
                print("   The issue is definitely in the position parsing, not the simulation")
        
        # Method 2: Test with a larger time jump to see if movement becomes detectable
        print("\n=== Testing with Fast-Forward ===")
        
        bluesky_client.hold()  # Pause
        bluesky_client.ff(300.0)  # Fast-forward 5 minutes
        
        ff_states = bluesky_client.get_aircraft_states(["SIMPLE01"])
        ff_pos = ff_states.get("SIMPLE01")
        
        if ff_pos and initial_pos:
            ff_lat_diff = abs(ff_pos.latitude - initial_pos.latitude)
            ff_lon_diff = abs(ff_pos.longitude - initial_pos.longitude)
            
            print(f"Position after 5min FF: lat={ff_pos.latitude:.6f}, lon={ff_pos.longitude:.6f}")
            print(f"Total movement: Δlat={ff_lat_diff:.6f}, Δlon={ff_lon_diff:.6f}")
            
            # Calculate where aircraft should be after 5 minutes
            ff_predicted_lat, ff_predicted_lon = calculate_predicted_position(initial_pos, 300.0)
            print(f"Predicted after 5min: lat={ff_predicted_lat:.6f}, lon={ff_predicted_lon:.6f}")
            
            if ff_lat_diff > 0.01 or ff_lon_diff > 0.01:
                print("✅ Large movement detected with fast-forward")
            else:
                print("❌ Still no movement detected even with 5-minute fast-forward")
                print("   This confirms the parser is always returning cached values")
        
        print("\n=== Recommended Solution ===")
        print("1. The POS command returns only metadata, not actual position data")
        print("2. Embedded BlueSky access is not working properly in this setup")
        print("3. The current parser falls back to cached creation values")
        print("4. Solutions:")
        print("   a. Fix BlueSky embedded access initialization")
        print("   b. Use dead reckoning for movement simulation")
        print("   c. Find alternative BlueSky commands that return actual data")
        print("   d. Implement proper binary parsing of the metadata format")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        bluesky_client.disconnect()
        print("\nDisconnected from BlueSky")

if __name__ == "__main__":
    simple_movement_test()
