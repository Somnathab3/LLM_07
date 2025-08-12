#!/usr/bin/env python3
"""
Debug BlueSky POS command response parsing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.cdr.simulation.bluesky_client import BlueSkyClient, create_thesis_config
import time

def debug_pos_response():
    """Debug what the POS command is actually returning"""
    
    print("=== Debugging BlueSky POS Response ===")
    
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
            callsign="TEST",
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
        
        print("✅ Created aircraft TEST")
        
        # Start simulation
        bluesky_client.op()
        time.sleep(2)
        
        # Get raw POS response
        print("\n=== Raw POS Response ===")
        raw_response = bluesky_client._send_command("POS TEST", expect_response=True)
        
        print(f"Response type: {type(raw_response)}")
        print(f"Response length: {len(raw_response) if raw_response else 'None'}")
        print(f"Raw response (first 200 chars): {repr(raw_response[:200]) if raw_response else 'None'}")
        
        # Try to decode different ways
        if raw_response:
            print("\n=== Response Analysis ===")
            
            # Try to extract aircraft name
            if "TEST" in raw_response:
                print("✅ Aircraft name found in response")
            else:
                print("❌ Aircraft name not found in response")
            
            # Look for numeric patterns
            import re
            numbers = re.findall(r'-?\d+\.?\d*', raw_response)
            print(f"Found {len(numbers)} numeric values: {numbers[:20]}...")
            
            # Look for specific patterns that might indicate coordinates
            lat_patterns = [r'lat[:\s]*(-?\d+\.?\d*)', r'latitude[:\s]*(-?\d+\.?\d*)']
            lon_patterns = [r'lon[:\s]*(-?\d+\.?\d*)', r'longitude[:\s]*(-?\d+\.?\d*)']
            
            for pattern in lat_patterns:
                matches = re.findall(pattern, raw_response, re.IGNORECASE)
                if matches:
                    print(f"Found latitude pattern: {matches}")
            
            for pattern in lon_patterns:
                matches = re.findall(pattern, raw_response, re.IGNORECASE)
                if matches:
                    print(f"Found longitude pattern: {matches}")
            
            # Try alternative approaches
            print("\n=== Alternative Approaches ===")
            
            # Try different BlueSky commands
            print("1. Testing individual property commands...")
            
            # Try getting latitude specifically
            lat_response = bluesky_client._send_command("ECHO lat TEST", expect_response=True)
            print(f"LAT response: {repr(lat_response[:100]) if lat_response else 'None'}")
            
            # Try different format
            pos_alt_response = bluesky_client._send_command("POS TEST lat lon alt", expect_response=True)
            print(f"POS with fields response: {repr(pos_alt_response[:100]) if pos_alt_response else 'None'}")
            
            # Try TRAIL command which might give position
            trail_response = bluesky_client._send_command("TRAIL TEST", expect_response=True)
            print(f"TRAIL response: {repr(trail_response[:100]) if trail_response else 'None'}")
            
            # Try direct property access if embedded mode works
            print("\n2. Testing embedded mode property access...")
            
            try:
                # Check if we have embedded access
                if hasattr(bluesky_client, 'embedded_sim') and bluesky_client.embedded_sim:
                    traffic = bluesky_client.embedded_sim.traffic
                    
                    if hasattr(traffic, 'id') and "TEST" in traffic.id:
                        idx = list(traffic.id).index("TEST")
                        print(f"Found aircraft at index {idx}")
                        
                        if hasattr(traffic, 'lat') and hasattr(traffic, 'lon'):
                            lat_val = traffic.lat[idx] if idx < len(traffic.lat) else None
                            lon_val = traffic.lon[idx] if idx < len(traffic.lon) else None
                            print(f"Direct access - Lat: {lat_val}, Lon: {lon_val}")
                        else:
                            print("Traffic object doesn't have lat/lon attributes")
                            print(f"Available attributes: {dir(traffic)[:10]}...")
                    else:
                        print(f"Aircraft not found in traffic.id: {getattr(traffic, 'id', 'No id attr')}")
                else:
                    print("Embedded simulation not available")
                    
            except Exception as e:
                print(f"Embedded access failed: {e}")
        
        # Test movement with a manual position command
        print("\n=== Testing Manual Position Setting ===")
        
        # Try to set position manually and see if that works
        move_success = bluesky_client.move_aircraft("TEST", 42.1, -86.9)
        if move_success:
            print("✅ Manual move command successful")
            
            # Check response again
            new_response = bluesky_client._send_command("POS TEST", expect_response=True)
            print(f"New POS response: {repr(new_response[:200]) if new_response else 'None'}")
        
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()
    finally:
        bluesky_client.disconnect()
        print("\nDisconnected from BlueSky")

if __name__ == "__main__":
    debug_pos_response()
