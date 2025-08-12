#!/usr/bin/env python3
"""
More robust BlueSky response parser
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.cdr.simulation.bluesky_client import BlueSkyClient, create_thesis_config
import time
import struct

def analyze_response_format(response: str):
    """Analyze the actual format of BlueSky responses"""
    
    print(f"=== Response Format Analysis ===")
    print(f"Response type: {type(response)}")
    print(f"Response length: {len(response)}")
    print(f"First 50 chars: {repr(response[:50])}")
    print(f"Last 50 chars: {repr(response[-50:])}")
    
    # Check for different possible headers
    possible_headers = ["*****ACDATAS", "*****AC", "ACDATAS", "numpy", "data"]
    for header in possible_headers:
        if response.startswith(header):
            print(f"✅ Starts with: {header}")
            return header
        elif header in response[:20]:
            print(f"✅ Contains early: {header}")
    
    # Look for patterns
    import re
    
    # Look for field patterns
    field_patterns = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*(?:numpy|type)', response[:200])
    print(f"Field patterns found: {field_patterns}")
    
    # Look for numpy type patterns
    numpy_patterns = re.findall(r'numpy(?:type)?<[^>]*>', response[:500])
    print(f"Numpy type patterns: {numpy_patterns}")
    
    return None

def extract_float_data_simple(response: str):
    """Simple approach: try to extract float data directly from the binary response"""
    
    try:
        # Convert to bytes
        if isinstance(response, str):
            response_bytes = response.encode('latin1')
        else:
            response_bytes = response
        
        print(f"🔍 Analyzing {len(response_bytes)} bytes of data")
        
        # Try to find 8-byte float64 values at different alignments
        found_floats = []
        
        for offset in range(0, min(len(response_bytes) - 7, 1000), 1):  # Check first 1000 bytes
            for endian in ['<', '>']:
                try:
                    value = struct.unpack(f'{endian}d', response_bytes[offset:offset+8])[0]
                    
                    # Filter for reasonable aviation values
                    if (-90 <= value <= 90 or          # Latitude
                        -180 <= value <= 180 or        # Longitude  
                        0 <= value <= 50000 or         # Altitude
                        0 <= value <= 1000):           # Speed
                        
                        found_floats.append((offset, value, endian))
                except:
                    continue
        
        print(f"🔍 Found {len(found_floats)} potential float values:")
        
        # Group by value ranges
        latitudes = [(o, v, e) for o, v, e in found_floats if -90 <= v <= 90]
        longitudes = [(o, v, e) for o, v, e in found_floats if -180 <= v <= 180 and abs(v) > 90]
        altitudes = [(o, v, e) for o, v, e in found_floats if 1000 <= v <= 50000]
        speeds = [(o, v, e) for o, v, e in found_floats if 0 <= v <= 1000 and v != 0]
        
        print(f"   Potential latitudes: {len(latitudes)}")
        for o, v, e in latitudes[:5]:
            print(f"     {v:.6f} at offset {o}")
        
        print(f"   Potential longitudes: {len(longitudes)}")  
        for o, v, e in longitudes[:5]:
            print(f"     {v:.6f} at offset {o}")
            
        print(f"   Potential altitudes: {len(altitudes)}")
        for o, v, e in altitudes[:5]:
            print(f"     {v:.1f} at offset {o}")
            
        print(f"   Potential speeds: {len(speeds)}")
        for o, v, e in speeds[:5]:
            print(f"     {v:.1f} at offset {o}")
        
        # Try to pick the most reasonable values
        result = {}
        
        if latitudes:
            # Pick latitude closest to expected range (around 42)
            best_lat = min(latitudes, key=lambda x: abs(x[1] - 42))
            result['lat'] = best_lat[1]
            print(f"✅ Selected latitude: {best_lat[1]:.6f}")
        
        if longitudes:
            # Pick longitude closest to expected range (around -87)
            best_lon = min(longitudes, key=lambda x: abs(x[1] - (-87)))
            result['lon'] = best_lon[1]
            print(f"✅ Selected longitude: {best_lon[1]:.6f}")
        
        if altitudes:
            # Pick altitude closest to expected (35000)
            best_alt = min(altitudes, key=lambda x: abs(x[1] - 35000))
            result['alt'] = best_alt[1]
            print(f"✅ Selected altitude: {best_alt[1]:.1f}")
        
        if speeds:
            # Pick reasonable speed 
            best_speed = min(speeds, key=lambda x: abs(x[1] - 450))
            result['speed'] = best_speed[1]
            print(f"✅ Selected speed: {best_speed[1]:.1f}")
        
        return result if result else None
        
    except Exception as e:
        print(f"❌ Error in simple extraction: {e}")
        return None

def test_robust_parsing():
    """Test more robust parsing approach"""
    
    print("=== Testing Robust BlueSky Parsing ===")
    
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
        print("⏳ Letting simulation run for 5 seconds...")
        time.sleep(5)
        
        # Get POS response and analyze it
        print("\n=== Analyzing POS Response ===")
        raw_response = bluesky_client._send_command("POS TEST", expect_response=True)
        
        if raw_response:
            # Analyze the format first
            analyze_response_format(raw_response)
            
            # Try simple extraction
            print("\n=== Extracting Data ===")
            parsed_data = extract_float_data_simple(raw_response)
            
            if parsed_data:
                print("✅ Successfully extracted position data:")
                for key, value in parsed_data.items():
                    print(f"   {key}: {value}")
                initial_position = parsed_data.copy()
            else:
                print("❌ Failed to extract position data")
                return
        else:
            print("❌ No response from POS command")
            return
        
        # Test movement by fast-forwarding
        print("\n=== Testing Movement ===")
        bluesky_client.hold()
        print("Fast-forwarding 120 seconds...")
        bluesky_client.ff(120.0)
        
        # Get new position
        new_response = bluesky_client._send_command("POS TEST", expect_response=True)
        
        if new_response:
            new_parsed_data = extract_float_data_simple(new_response)
            
            if new_parsed_data:
                print("✅ Successfully extracted new position data:")
                for key, value in new_parsed_data.items():
                    print(f"   {key}: {value}")
                
                # Compare positions
                if 'lat' in initial_position and 'lat' in new_parsed_data:
                    lat_diff = abs(new_parsed_data['lat'] - initial_position['lat'])
                    lon_diff = abs(new_parsed_data['lon'] - initial_position['lon']) if 'lon' in both else 0
                    
                    print(f"\n📊 Movement Analysis:")
                    print(f"   Initial: lat={initial_position.get('lat', 'N/A'):.6f}, lon={initial_position.get('lon', 'N/A'):.6f}")
                    print(f"   Final:   lat={new_parsed_data.get('lat', 'N/A'):.6f}, lon={new_parsed_data.get('lon', 'N/A'):.6f}")
                    print(f"   Change:  Δlat={lat_diff:.6f}, Δlon={lon_diff:.6f}")
                    
                    if lat_diff > 0.001 or lon_diff > 0.001:
                        print("🎉 SUCCESS: Aircraft is actually moving!")
                        print("   The issue is definitely in the parsing, not the simulation")
                    else:
                        print("❌ Still no movement detected")
            else:
                print("❌ Failed to extract new position data")
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
    test_robust_parsing()
