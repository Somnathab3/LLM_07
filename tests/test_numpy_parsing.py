#!/usr/bin/env python3
"""
Fix BlueSky POS response parsing to handle numpy binary format properly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.cdr.simulation.bluesky_client import BlueSkyClient, create_thesis_config
import time
import struct
import numpy as np

def parse_bluesky_numpy_response(response: str, target_callsign: str = None):
    """
    Parse BlueSky's numpy binary response format
    
    The format appears to be:
    *****ACDATAS + binary numpy structured array data
    """
    
    if not response or not response.startswith("*****ACDATAS"):
        print("❌ Response doesn't have expected BlueSky format")
        return None
    
    try:
        # Remove the header and get the binary data
        binary_data = response[12:]  # Skip "*****ACDATAS"
        
        print(f"🔍 Parsing binary data, length: {len(binary_data)}")
        print(f"🔍 First 100 chars of binary: {repr(binary_data[:100])}")
        
        # The format seems to contain field descriptors followed by data
        # Let's try to extract the structured format information
        
        # Look for patterns that indicate field names and types
        import re
        
        # Extract field definitions (field name + numpy type info)
        field_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)numpy(?:type<([^>]+)>)?(?:shape)?(?:data)?'
        field_matches = re.findall(field_pattern, binary_data)
        
        print(f"🔍 Found {len(field_matches)} potential fields:")
        for field_name, field_type in field_matches:
            print(f"   {field_name}: {field_type}")
        
        # Look for specific fields we need
        needed_fields = ['lat', 'lon', 'alt', 'tas', 'cas', 'gs', 'hdg', 'vs']
        found_fields = {}
        
        for field_name, field_type in field_matches:
            if field_name in needed_fields:
                found_fields[field_name] = field_type
                print(f"✅ Found needed field: {field_name} ({field_type})")
        
        # Try to extract actual numeric data
        # The binary data likely contains the actual float64 values after the structure definition
        
        # Convert to bytes and look for float64 patterns
        binary_bytes = binary_data.encode('latin1') if isinstance(binary_data, str) else binary_data
        
        # Try to find float64 values (8 bytes each)
        float_values = []
        for i in range(0, len(binary_bytes) - 7, 1):  # Step by 1 to catch misaligned data
            try:
                # Try both little and big endian
                for endian in ['<', '>']:
                    try:
                        value = struct.unpack(f'{endian}d', binary_bytes[i:i+8])[0]
                        # Check if this looks like a reasonable coordinate/aviation value
                        if -180 <= value <= 180 or 0 <= value <= 100000:  # lat/lon range or altitude range
                            float_values.append((i, value, endian))
                    except:
                        continue
            except:
                continue
        
        print(f"🔍 Found {len(float_values)} potential float values:")
        for i, (offset, value, endian) in enumerate(float_values[:20]):  # Show first 20
            print(f"   {i}: {value:.6f} (offset {offset}, {endian})")
        
        # Try to identify which values are lat/lon based on reasonable ranges
        potential_coords = []
        for offset, value, endian in float_values:
            if -90 <= value <= 90:  # Latitude range
                potential_coords.append(('lat', value, offset, endian))
            elif -180 <= value <= 180 and abs(value) > 90:  # Longitude range (excluding lat overlap)
                potential_coords.append(('lon', value, offset, endian))
            elif 0 <= value <= 50000:  # Altitude range
                potential_coords.append(('alt', value, offset, endian))
            elif 0 <= value <= 1000:  # Speed range
                potential_coords.append(('speed', value, offset, endian))
        
        print(f"🔍 Potential coordinates/aviation values:")
        for coord_type, value, offset, endian in potential_coords:
            print(f"   {coord_type}: {value:.6f} (offset {offset})")
        
        # Group by type and pick most likely values
        result = {}
        for coord_type in ['lat', 'lon', 'alt', 'speed']:
            candidates = [v for t, v, o, e in potential_coords if t == coord_type]
            if candidates:
                # For now, take the first reasonable value
                # In a more sophisticated parser, we'd consider proximity and structure
                result[coord_type] = candidates[0]
                print(f"✅ Selected {coord_type}: {candidates[0]:.6f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error parsing numpy response: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_improved_parsing():
    """Test the improved parsing with a real BlueSky instance"""
    
    print("=== Testing Improved BlueSky Parsing ===")
    
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
        
        # Start simulation and let it run
        bluesky_client.op()
        print("⏳ Letting simulation run for 10 seconds...")
        time.sleep(10)
        
        # Get POS response and parse it
        print("\n=== Parsing Initial Position ===")
        raw_response = bluesky_client._send_command("POS TEST", expect_response=True)
        
        if raw_response:
            parsed_data = parse_bluesky_numpy_response(raw_response, "TEST")
            
            if parsed_data:
                print("✅ Successfully parsed initial position:")
                print(f"   Latitude: {parsed_data.get('lat', 'N/A')}")
                print(f"   Longitude: {parsed_data.get('lon', 'N/A')}")
                print(f"   Altitude: {parsed_data.get('alt', 'N/A')}")
                print(f"   Speed: {parsed_data.get('speed', 'N/A')}")
                initial_lat = parsed_data.get('lat')
                initial_lon = parsed_data.get('lon')
            else:
                print("❌ Failed to parse initial position")
                return
        else:
            print("❌ No response from POS command")
            return
        
        # Fast forward and check again
        print("\n=== Fast Forward Test ===")
        bluesky_client.hold()
        bluesky_client.ff(120.0)  # 2 minutes
        
        print("Getting position after fast-forward...")
        new_response = bluesky_client._send_command("POS TEST", expect_response=True)
        
        if new_response:
            new_parsed_data = parse_bluesky_numpy_response(new_response, "TEST")
            
            if new_parsed_data:
                print("✅ Successfully parsed new position:")
                print(f"   Latitude: {new_parsed_data.get('lat', 'N/A')}")
                print(f"   Longitude: {new_parsed_data.get('lon', 'N/A')}")
                print(f"   Altitude: {new_parsed_data.get('alt', 'N/A')}")
                print(f"   Speed: {new_parsed_data.get('speed', 'N/A')}")
                
                # Calculate movement
                new_lat = new_parsed_data.get('lat')
                new_lon = new_parsed_data.get('lon')
                
                if initial_lat is not None and new_lat is not None and initial_lon is not None and new_lon is not None:
                    lat_diff = abs(new_lat - initial_lat)
                    lon_diff = abs(new_lon - initial_lon)
                    
                    print(f"\n📊 Movement Analysis:")
                    print(f"   Initial: ({initial_lat:.6f}, {initial_lon:.6f})")
                    print(f"   Final:   ({new_lat:.6f}, {new_lon:.6f})")
                    print(f"   Change:  Δlat={lat_diff:.6f}, Δlon={lon_diff:.6f}")
                    
                    if lat_diff > 0.001 or lon_diff > 0.001:
                        print("✅ SUCCESS: Aircraft moved significantly!")
                        print("   The parsing issue has been identified and can be fixed!")
                    else:
                        print("❌ Still no movement detected")
            else:
                print("❌ Failed to parse new position")
        else:
            print("❌ No response from second POS command")
    
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        bluesky_client.disconnect()
        print("\nDisconnected from BlueSky")

if __name__ == "__main__":
    test_improved_parsing()
