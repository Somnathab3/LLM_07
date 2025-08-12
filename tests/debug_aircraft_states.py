#!/usr/bin/env python3
"""Debug script to check aircraft states and heading updates"""

import sys
sys.path.append('.')

from src.cdr.simulation.bluesky_client import BlueSkyClient, BlueSkyConfig
import time

def debug_aircraft_states():
    """Check what aircraft states BlueSky is returning"""
    
    # Connect to running BlueSky instance (don't launch new one)
    config = BlueSkyConfig()
    client = BlueSkyClient(config)
    
    try:
        # Try to connect to existing instance without launching
        print("Attempting to connect to existing BlueSky instance...")
        if client._connect_to_bluesky():
            print("=== Connected to existing BlueSky ===")
        else:
            print("No existing BlueSky instance found")
            return
        
        # Get current aircraft states
        states = client.get_aircraft_states()
        
        print(f"\n=== Current Aircraft States ({len(states)} aircraft) ===")
        for callsign, state in states.items():
            print(f"Aircraft: {callsign}")
            print(f"  Position: {state.latitude:.4f}, {state.longitude:.4f}")
            print(f"  Altitude: {state.altitude_ft:.0f} ft")
            print(f"  Heading: {state.heading_deg:.1f}°")
            print(f"  Speed: {state.speed_kt:.1f} kt")
            print(f"  Timestamp: {state.timestamp}")
            print()
        
        # Test heading command
        if 'OWNSHIP' in states:
            print("=== Testing Heading Command ===")
            current_heading = states['OWNSHIP'].heading_deg
            test_heading = (current_heading + 45) % 360
            
            print(f"Current heading: {current_heading:.1f}°")
            print(f"Commanding new heading: {test_heading:.1f}°")
            
            success = client.heading_command('OWNSHIP', test_heading)
            print(f"Command success: {success}")
            
            # Wait and check if heading changed
            time.sleep(2)
            new_states = client.get_aircraft_states()
            if 'OWNSHIP' in new_states:
                new_heading = new_states['OWNSHIP'].heading_deg
                print(f"New heading after 2s: {new_heading:.1f}°")
                print(f"Heading changed: {abs(new_heading - current_heading) > 1}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.disconnect()

if __name__ == "__main__":
    debug_aircraft_states()
