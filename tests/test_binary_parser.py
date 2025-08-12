#!/usr/bin/env python3
"""
Test the improved binary parser for BlueSky POS responses
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from src.cdr.simulation.bluesky_client import BlueSkyClient, create_thesis_config
from bluesky_binary_parser import BlueSkyBinaryParser, fix_pos_command_method, improved_get_aircraft_states

def test_binary_parser():
    """Test the improved binary parser approach"""
    
    print("=== Testing Improved Binary Parser ===")
    
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
        
        # Reset and create aircraft
        bluesky_client._send_command("RESET")
        time.sleep(1)
        
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
        
        print("✅ Created aircraft TEST01 at (42.0, -87.0)")
        
        # Start simulation
        bluesky_client.op()
        time.sleep(2)
        
        # Test improved POS parsing
        print("\n=== Testing Improved POS Parsing ===")
        
        aircraft_state = fix_pos_command_method(bluesky_client, "TEST01")
        
        if aircraft_state:
            print("✅ Successfully parsed aircraft state:")
            print(f"   Callsign: {aircraft_state['callsign']}")
            print(f"   Latitude: {aircraft_state['latitude']:.6f}")
            print(f"   Longitude: {aircraft_state['longitude']:.6f}")
            print(f"   Altitude: {aircraft_state['altitude_ft']:.0f} ft")
            print(f"   Heading: {aircraft_state['heading_deg']:.1f}°")
            print(f"   Speed: {aircraft_state['speed_kt']:.0f} kt")
            
            initial_state = aircraft_state.copy()
        else:
            print("❌ Failed to parse aircraft state")
            return
        
        # Test movement detection
        print("\n=== Testing Movement Detection ===")
        print("Letting simulation run for 10 seconds...")
        time.sleep(10)
        
        new_aircraft_state = fix_pos_command_method(bluesky_client, "TEST01")
        
        if new_aircraft_state:
            print("✅ Successfully parsed new aircraft state:")
            print(f"   Latitude: {new_aircraft_state['latitude']:.6f}")
            print(f"   Longitude: {new_aircraft_state['longitude']:.6f}")
            
            # Calculate movement
            lat_diff = abs(new_aircraft_state['latitude'] - initial_state['latitude'])
            lon_diff = abs(new_aircraft_state['longitude'] - initial_state['longitude'])
            
            print(f"\n📊 Movement Analysis:")
            print(f"   Initial: lat={initial_state['latitude']:.6f}, lon={initial_state['longitude']:.6f}")
            print(f"   After 10s: lat={new_aircraft_state['latitude']:.6f}, lon={new_aircraft_state['longitude']:.6f}")
            print(f"   Change: Δlat={lat_diff:.6f}, Δlon={lon_diff:.6f}")
            
            if lat_diff > 0.0001 or lon_diff > 0.0001:
                print("🎉 SUCCESS: Aircraft is moving!")
            else:
                print("❌ No movement detected")
        else:
            print("❌ Failed to parse new aircraft state")
        
        # Test fast-forward
        print("\n=== Testing Fast-Forward Movement ===")
        bluesky_client.hold()
        print("Fast-forwarding 120 seconds...")
        bluesky_client.ff(120.0)
        
        ff_aircraft_state = fix_pos_command_method(bluesky_client, "TEST01")
        
        if ff_aircraft_state:
            lat_diff_ff = abs(ff_aircraft_state['latitude'] - initial_state['latitude'])
            lon_diff_ff = abs(ff_aircraft_state['longitude'] - initial_state['longitude'])
            
            print(f"📊 Fast-Forward Movement:")
            print(f"   Initial: lat={initial_state['latitude']:.6f}, lon={initial_state['longitude']:.6f}")
            print(f"   After FF: lat={ff_aircraft_state['latitude']:.6f}, lon={ff_aircraft_state['longitude']:.6f}")
            print(f"   Change: Δlat={lat_diff_ff:.6f}, Δlon={lon_diff_ff:.6f}")
            
            if lat_diff_ff > 0.001 or lon_diff_ff > 0.001:
                print("🎉 SUCCESS: Significant movement after fast-forward!")
            else:
                print("❌ Still no significant movement")
        
        # Test improved get_aircraft_states function
        print("\n=== Testing Improved get_aircraft_states ===")
        
        states = improved_get_aircraft_states(bluesky_client, ["TEST01"])
        
        if states:
            print(f"✅ Retrieved states for {len(states)} aircraft")
            for callsign, state in states.items():
                print(f"   {callsign}: lat={state['latitude']:.6f}, lon={state['longitude']:.6f}")
        else:
            print("❌ No states retrieved")
    
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        bluesky_client.disconnect()
        print("\nDisconnected from BlueSky")

if __name__ == "__main__":
    test_binary_parser()
