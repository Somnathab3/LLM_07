#!/usr/bin/env python3
"""
Debug aircraft movement in BlueSky
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.cdr.simulation.bluesky_client import BlueSkyClient, create_thesis_config
import time

def debug_aircraft_movement():
    """Debug aircraft movement by creating aircraft and testing if they move"""
    
    print("=== Debugging Aircraft Movement ===")
    
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
        
        # Reset simulation first
        print("Resetting simulation...")
        bluesky_client._send_command("RESET")
        time.sleep(1)
        
        # Create a test aircraft
        print("Creating test aircraft...")
        success = bluesky_client.create_aircraft(
            callsign="OWNSHIP",
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
        
        print("✅ Created aircraft OWNSHIP")
        
        # Start simulation and get initial position
        print("Starting simulation...")
        bluesky_client.op()
        time.sleep(1)  # Let simulation initialize
        
        print("Getting initial position...")
        initial_states = bluesky_client.get_aircraft_states(["OWNSHIP"])
        initial_pos = initial_states.get("OWNSHIP")
        
        if not initial_pos:
            print("❌ Failed to get initial position")
            return
            
        print(f"✅ Initial position: lat={initial_pos.latitude:.6f}, lon={initial_pos.longitude:.6f}")
        print(f"   Altitude: {initial_pos.altitude_ft:.0f} ft")
        print(f"   Heading: {initial_pos.heading_deg:.1f}°")
        print(f"   Speed: {initial_pos.speed_kt:.0f} kt")
        
        # Let simulation run for a few seconds
        print("Letting simulation run for 5 seconds...")
        time.sleep(5)
        
        # Get position after 5 seconds
        print("Getting position after 5 seconds...")
        new_states = bluesky_client.get_aircraft_states(["OWNSHIP"])
        new_pos = new_states.get("OWNSHIP")
        
        if not new_pos:
            print("❌ Failed to get new position")
            return
            
        print(f"✅ Position after 5s: lat={new_pos.latitude:.6f}, lon={new_pos.longitude:.6f}")
        print(f"   Altitude: {new_pos.altitude_ft:.0f} ft")
        print(f"   Heading: {new_pos.heading_deg:.1f}°")
        print(f"   Speed: {new_pos.speed_kt:.0f} kt")
        
        # Calculate movement
        lat_diff = abs(new_pos.latitude - initial_pos.latitude)
        lon_diff = abs(new_pos.longitude - initial_pos.longitude)
        
        print(f"\nMovement analysis:")
        print(f"   Δlat = {lat_diff:.6f} degrees")
        print(f"   Δlon = {lon_diff:.6f} degrees")
        
        if lat_diff > 0.001 or lon_diff > 0.001:
            print("✅ SUCCESS: Aircraft moved significantly!")
        else:
            print("❌ PROBLEM: Aircraft did not move")
            print("   Possible causes:")
            print("   1. Simulation is paused")
            print("   2. Aircraft is not receiving movement commands")
            print("   3. Time factor is set to 0")
            print("   4. Aircraft state parsing is returning cached values")
        
        # Test with fast-forward
        print("\n=== Testing Fast-Forward ===")
        print("Pausing simulation...")
        bluesky_client.hold()
        time.sleep(1)
        
        print("Fast-forwarding 60 seconds...")
        ff_success = bluesky_client.ff(60.0)
        
        if ff_success:
            print("✅ Fast-forward command successful")
            
            # Get position after fast-forward
            ff_states = bluesky_client.get_aircraft_states(["OWNSHIP"])
            ff_pos = ff_states.get("OWNSHIP")
            
            if ff_pos:
                print(f"Position after FF: lat={ff_pos.latitude:.6f}, lon={ff_pos.longitude:.6f}")
                
                ff_lat_diff = abs(ff_pos.latitude - initial_pos.latitude)
                ff_lon_diff = abs(ff_pos.longitude - initial_pos.longitude)
                
                print(f"Total movement: Δlat={ff_lat_diff:.6f}, Δlon={ff_lon_diff:.6f}")
                
                if ff_lat_diff > 0.01 or ff_lon_diff > 0.01:
                    print("✅ SUCCESS: Fast-forward caused significant movement!")
                else:
                    print("❌ PROBLEM: Even fast-forward didn't cause movement")
        else:
            print("❌ Fast-forward command failed")
        
        # Test direct simulation step
        print("\n=== Testing Direct Simulation Step ===")
        if hasattr(bluesky_client, 'step_minutes'):
            print("Testing step_minutes method...")
            step_success = bluesky_client.step_minutes(1.0)  # 1 minute
            
            if step_success:
                step_states = bluesky_client.get_aircraft_states(["OWNSHIP"])
                step_pos = step_states.get("OWNSHIP")
                
                if step_pos:
                    step_lat_diff = abs(step_pos.latitude - initial_pos.latitude)
                    step_lon_diff = abs(step_pos.longitude - initial_pos.longitude)
                    
                    print(f"Position after step: lat={step_pos.latitude:.6f}, lon={step_pos.longitude:.6f}")
                    print(f"Step movement: Δlat={step_lat_diff:.6f}, Δlon={step_lon_diff:.6f}")
                    
                    if step_lat_diff > 0.01 or step_lon_diff > 0.01:
                        print("✅ SUCCESS: Direct step caused movement!")
                    else:
                        print("❌ PROBLEM: Direct step didn't cause movement")
        
        # Check simulation state
        print("\n=== Checking Simulation State ===")
        
        # Check if simulation is actually running
        sim_time_response = bluesky_client._send_command("ECHO SIM TIME", expect_response=True)
        print(f"Simulation time response: {sim_time_response}")
        
        # Check time factor
        dtmult_response = bluesky_client._send_command("ECHO DTMULT", expect_response=True)
        print(f"Time factor (DTMULT) response: {dtmult_response}")
        
        # Try to set time factor explicitly
        print("Setting time factor to 1.0...")
        bluesky_client.set_fast_time_factor(1.0)
        
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        bluesky_client.disconnect()
        print("\nDisconnected from BlueSky")

if __name__ == "__main__":
    debug_aircraft_movement()
