#!/usr/bin/env python3
"""
Try alternative BlueSky commands to get aircraft positions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.cdr.simulation.bluesky_client import BlueSkyClient, create_thesis_config
import time

def test_alternative_commands():
    """Test different BlueSky commands to get aircraft position data"""
    
    print("=== Testing Alternative BlueSky Commands ===")
    
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
        
        print("✅ Created aircraft TEST at (42.0, -87.0)")
        
        # Start simulation
        bluesky_client.op()
        time.sleep(2)
        
        # Test different commands
        commands_to_test = [
            "LIST",  # List all aircraft
            "LISTAC",  # List aircraft
            "POS",  # Position of all aircraft
            "POS TEST",  # Position of specific aircraft
            "POSALL",  # All positions
            "STATE",  # Aircraft state
            "STATE TEST",  # Specific aircraft state
            "ECHO lat",  # Try to echo latitude values
            "ECHO lon",  # Try to echo longitude values
            "ECHO alt",  # Try to echo altitude values
            "ECHO TEST.lat",  # Specific aircraft lat
            "ECHO TEST.lon",  # Specific aircraft lon
            "ECHO TEST.alt",  # Specific aircraft alt
            "TRAIL TEST",  # Aircraft trail info
            "INFO TEST",  # Aircraft info
            "DIST TEST",  # Distance from reference
        ]
        
        print("\n=== Testing Commands ===")
        for cmd in commands_to_test:
            print(f"\n--- Command: {cmd} ---")
            try:
                response = bluesky_client._send_command(cmd, expect_response=True, timeout=3.0)
                if response:
                    # Show first 200 chars to see format
                    print(f"Response ({len(response)} chars): {repr(response[:200])}...")
                    
                    # Look for our aircraft name and coordinates
                    if "TEST" in response:
                        print("✅ Contains aircraft name")
                    
                    # Look for coordinates around our expected values
                    import re
                    numbers = re.findall(r'-?\d+\.?\d+', response)
                    
                    # Look for values that might be our coordinates
                    for num_str in numbers:
                        try:
                            val = float(num_str)
                            if 41 <= val <= 43:  # Latitude range
                                print(f"🎯 Potential latitude: {val}")
                            elif -89 <= val <= -85:  # Longitude range
                                print(f"🎯 Potential longitude: {val}")
                            elif 34000 <= val <= 36000:  # Altitude range
                                print(f"🎯 Potential altitude: {val}")
                        except ValueError:
                            continue
                else:
                    print("❌ No response")
            except Exception as e:
                print(f"❌ Command failed: {e}")
        
        # Test specific property access
        print("\n=== Testing Property Access ===")
        
        # Try to access individual properties
        property_commands = [
            "LAT TEST",
            "LON TEST", 
            "ALT TEST",
            "HDG TEST",
            "SPD TEST",
            "VS TEST",
            "CAS TEST",
            "TAS TEST",
            "GS TEST"
        ]
        
        for cmd in property_commands:
            print(f"\n--- Property: {cmd} ---")
            try:
                response = bluesky_client._send_command(cmd, expect_response=True, timeout=2.0)
                if response:
                    print(f"Response: {repr(response[:100])}")
                    
                    # Try to extract numeric value
                    import re
                    numbers = re.findall(r'-?\d+\.?\d+', response)
                    if numbers:
                        print(f"Numeric values: {numbers}")
                else:
                    print("❌ No response")
            except Exception as e:
                print(f"❌ Property access failed: {e}")
        
        # Let aircraft move and test again
        print("\n=== Testing After Movement ===")
        print("Letting simulation run for 10 seconds...")
        time.sleep(10)
        
        # Test POS command again
        print("\nChecking POS TEST after movement:")
        response = bluesky_client._send_command("POS TEST", expect_response=True)
        if response:
            print(f"Response length: {len(response)}")
            
            # Try a different parsing approach for this specific response
            # Look for structured patterns
            if "lat" in response.lower() or "lon" in response.lower():
                print("✅ Response contains lat/lon keywords")
            
            # Try to find patterns like "lat:value" or "lat=value"
            import re
            lat_patterns = [
                r'lat[:\s=]*([0-9.-]+)',
                r'latitude[:\s=]*([0-9.-]+)',
                r'TEST.*?([0-9.-]+)[^0-9.-]*([0-9.-]+)'  # TEST followed by two numbers
            ]
            
            for pattern in lat_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                if matches:
                    print(f"Pattern matches for '{pattern}': {matches}")
        
        # Test fast-forward to see if movement is detectable
        print("\n=== Testing Fast-Forward Movement ===")
        bluesky_client.hold()
        print("Fast-forwarding 120 seconds...")
        bluesky_client.ff(120.0)
        
        # Check LIST command to see if aircraft is still there
        list_response = bluesky_client._send_command("LIST", expect_response=True)
        if list_response and "TEST" in list_response:
            print("✅ Aircraft still exists after fast-forward")
        
        # Check POS again
        pos_response_after = bluesky_client._send_command("POS TEST", expect_response=True)
        if pos_response_after:
            print(f"POS response after FF: {len(pos_response_after)} chars")
            # Compare with before
            if pos_response_after != response:
                print("✅ POS response changed after fast-forward!")
                print("   This confirms the aircraft is moving, just need better parsing")
            else:
                print("❌ POS response identical - either not moving or cached")
    
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        bluesky_client.disconnect()
        print("\nDisconnected from BlueSky")

if __name__ == "__main__":
    test_alternative_commands()
