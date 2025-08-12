#!/usr/bin/env python3
"""
Improved BlueSky binary protocol parser for POS command responses
"""

import struct
import re
import math
from typing import Optional, Dict, Any, List

class BlueSkyBinaryParser:
    """Parser for BlueSky's binary numpy data format"""
    
    def __init__(self):
        # Known BlueSky data structure based on the metadata
        self.field_mapping = {
            'lat': ('latitude', 'f8'),      # 64-bit float
            'lon': ('longitude', 'f8'),     # 64-bit float  
            'alt': ('altitude', 'f8'),      # 64-bit float (meters)
            'tas': ('true_airspeed', 'f8'), # 64-bit float (m/s)
            'cas': ('calibrated_airspeed', 'f8'), # 64-bit float (m/s)
            'gs': ('ground_speed', 'f8'),   # 64-bit float (m/s)
            'hdg': ('heading', 'f8'),       # 64-bit float (degrees)
            'vs': ('vertical_speed', 'f8'), # 64-bit float (m/s)
            'ingroup': ('in_group', 'i8'),  # 64-bit int
            'inconf': ('in_conflict', 'i8') # 64-bit int
        }
    
    def parse_acdatas_response(self, response: str, callsign: str) -> Optional[Dict[str, Any]]:
        """
        Parse BlueSky ACDATAS response format
        
        The response format appears to be:
        *****ACDATAS;R<timestamp>lat<metadata>lon<metadata>alt<metadata>...
        
        This is followed by the actual binary data, but you're only getting the metadata.
        """
        try:
            if not response or "*****ACDATAS" not in response:
                return None
            
            print(f"🔍 Parsing ACDATAS response for {callsign}")
            print(f"   Response length: {len(response)} characters")
            print(f"   First 200 chars: {response[:200]}")
            
            # Strategy 1: Check if this is just metadata (what you're seeing)
            if self._is_metadata_only(response):
                print("⚠️ Response contains only metadata, no actual data")
                print("   This means the POS command is returning schema info instead of values")
                return None
            
            # Strategy 2: Look for actual binary data after metadata
            data_section = self._extract_data_section(response)
            if data_section:
                return self._parse_binary_data_section(data_section, callsign)
            
            # Strategy 3: Try to find embedded values in the metadata
            embedded_values = self._extract_embedded_values(response, callsign)
            if embedded_values:
                return embedded_values
            
            print("❌ Could not extract aircraft data from response")
            return None
            
        except Exception as e:
            print(f"❌ Error parsing ACDATAS response: {e}")
            return None
    
    def _is_metadata_only(self, response: str) -> bool:
        """Check if response contains only metadata without actual data"""
        # Look for indicators that this is metadata only
        metadata_indicators = [
            'numpytype<f8>',
            'numpytype<i8>',
            'shapedatalon',
            'shapedataaltnumpy',
            'shapedataingroupnumpy'
        ]
        
        # If we have metadata indicators but no actual floating point values
        has_metadata = any(indicator in response for indicator in metadata_indicators)
        
        # Look for actual floating point patterns (not just integers)
        float_pattern = re.findall(r'\d+\.\d+', response)
        
        return has_metadata and len(float_pattern) < 2
    
    def _extract_data_section(self, response: str) -> Optional[bytes]:
        """Extract binary data section if it exists after metadata"""
        try:
            # Look for the end of metadata and start of binary data
            # This might be after patterns like 'shapedatainconfnum'
            metadata_end_patterns = [
                'shapedatainconfnum',
                'numpytype<i8>shape',
                'dataingroupnumpy'
            ]
            
            for pattern in metadata_end_patterns:
                if pattern in response:
                    start_idx = response.find(pattern) + len(pattern)
                    if start_idx < len(response):
                        # Try to extract binary data after metadata
                        binary_section = response[start_idx:]
                        if binary_section and len(binary_section) > 10:
                            return binary_section.encode('latin1')
            
            return None
            
        except Exception as e:
            print(f"❌ Error extracting data section: {e}")
            return None
    
    def _parse_binary_data_section(self, data_bytes: bytes, callsign: str) -> Optional[Dict[str, Any]]:
        """Parse binary data section containing actual aircraft values"""
        try:
            aircraft_data = {}
            
            # Try to extract 64-bit floats from binary data
            for i in range(0, len(data_bytes) - 7, 8):  # Step by 8 bytes for doubles
                try:
                    # Try both little and big endian
                    for endian in ['<', '>']:
                        try:
                            value = struct.unpack(f'{endian}d', data_bytes[i:i+8])[0]
                            
                            # Skip invalid values
                            if math.isnan(value) or math.isinf(value):
                                continue
                            
                            # Categorize values based on reasonable ranges
                            if -90 <= value <= 90 and 'latitude' not in aircraft_data:
                                aircraft_data['latitude'] = value
                            elif -180 <= value <= 180 and 'longitude' not in aircraft_data:
                                aircraft_data['longitude'] = value
                            elif 0 <= value <= 50000 and 'altitude_m' not in aircraft_data:
                                aircraft_data['altitude_m'] = value
                            elif 0 <= value <= 360 and 'heading' not in aircraft_data:
                                aircraft_data['heading'] = value
                            elif 0 <= value <= 300 and 'speed_ms' not in aircraft_data:
                                aircraft_data['speed_ms'] = value
                                
                        except struct.error:
                            continue
                except:
                    continue
            
            if len(aircraft_data) >= 2:  # At least lat/lon
                print(f"✅ Extracted binary data for {callsign}: {aircraft_data}")
                return aircraft_data
            
            return None
            
        except Exception as e:
            print(f"❌ Error parsing binary data: {e}")
            return None
    
    def _extract_embedded_values(self, response: str, callsign: str) -> Optional[Dict[str, Any]]:
        """Try to extract any embedded numeric values from metadata"""
        try:
            # Look for any floating point numbers in the response
            all_numbers = re.findall(r'-?\d+\.?\d*', response)
            
            if not all_numbers:
                return None
            
            print(f"🔍 Found numbers in metadata: {all_numbers[:20]}...")  # Show first 20
            
            # Try to identify meaningful values
            aircraft_data = {}
            
            for num_str in all_numbers:
                try:
                    value = float(num_str)
                    
                    # Skip obviously invalid values
                    if abs(value) > 1000000:
                        continue
                    
                    # Try to categorize
                    if -90 <= value <= 90 and 'latitude' not in aircraft_data:
                        aircraft_data['latitude'] = value
                    elif -180 <= value <= 180 and 'longitude' not in aircraft_data:
                        aircraft_data['longitude'] = value
                    elif 0 <= value <= 50000 and 'altitude_m' not in aircraft_data:
                        aircraft_data['altitude_m'] = value
                    elif 0 <= value <= 360 and 'heading' not in aircraft_data:
                        aircraft_data['heading'] = value
                    
                except ValueError:
                    continue
            
            return aircraft_data if aircraft_data else None
            
        except Exception as e:
            print(f"❌ Error extracting embedded values: {e}")
            return None

def fix_pos_command_method(bluesky_client, callsign: str) -> Optional[Dict[str, Any]]:
    """
    Fixed version of POS command parsing for your BlueSkyClient
    """
    try:
        parser = BlueSkyBinaryParser()
        
        # Send POS command
        response = bluesky_client._send_command(f"POS {callsign}", expect_response=True, timeout=5.0)
        
        if not response or "ERROR" in response.upper():
            print(f"❌ POS command failed for {callsign}: {response}")
            return None
        
        # Parse the response
        parsed_data = parser.parse_acdatas_response(response, callsign)
        
        if parsed_data:
            # Convert units if needed
            aircraft_state = {
                'callsign': callsign,
                'latitude': parsed_data.get('latitude', 0.0),
                'longitude': parsed_data.get('longitude', 0.0),
                'altitude_ft': parsed_data.get('altitude_m', 0.0) * 3.28084,  # m to ft
                'heading_deg': parsed_data.get('heading', 0.0),
                'speed_kt': parsed_data.get('speed_ms', 0.0) * 1.94384,  # m/s to kt
                'timestamp': time.time()
            }
            return aircraft_state
        
        return None
        
    except Exception as e:
        print(f"❌ Error in fixed POS parsing: {e}")
        return None

def alternative_commands_test(bluesky_client):
    """Test alternative BlueSky commands that might work better than POS"""
    
    print("=== Testing Alternative BlueSky Commands ===")
    
    commands_to_try = [
        "TRAIL",      # Aircraft trail info
        "SSD",        # Screen data
        "LISTTRAIL",  # List trail data
        "INFO",       # General info
        "STATE",      # State information
        "GETACDATA",  # Get aircraft data (if available)
    ]
    
    for cmd in commands_to_try:
        try:
            print(f"\n--- Testing {cmd} command ---")
            response = bluesky_client._send_command(cmd, expect_response=True, timeout=3.0)
            
            if response and len(response) > 10:
                print(f"✅ {cmd} response ({len(response)} chars):")
                print(f"   {response[:200]}...")
                
                # Look for numeric data
                numbers = re.findall(r'-?\d+\.?\d+', response)
                if numbers:
                    print(f"   Found {len(numbers)} numbers: {numbers[:10]}...")
            else:
                print(f"❌ {cmd} no useful response")
                
        except Exception as e:
            print(f"❌ {cmd} failed: {e}")

# Example usage to replace your current parsing method
def improved_get_aircraft_states(bluesky_client, callsigns: List[str]) -> Dict[str, Dict[str, Any]]:
    """Improved aircraft state retrieval with multiple fallback methods"""
    
    states = {}
    
    print(f"🔍 Getting aircraft states for: {callsigns}")
    
    for callsign in callsigns:
        print(f"\n--- Processing {callsign} ---")
        
        # Method 1: Try improved POS parsing
        state = fix_pos_command_method(bluesky_client, callsign)
        if state:
            states[callsign] = state
            print(f"✅ Got {callsign} via improved POS parsing")
            continue
        
        # Method 2: Try alternative commands
        for alt_cmd in ["TRAIL", "SSD", "INFO"]:
            try:
                response = bluesky_client._send_command(f"{alt_cmd} {callsign}", expect_response=True, timeout=2.0)
                if response and any(char.isdigit() for char in response):
                    # Try to extract position info
                    numbers = re.findall(r'-?\d+\.?\d+', response)
                    if len(numbers) >= 2:
                        print(f"✅ {callsign} found numbers via {alt_cmd}: {numbers[:5]}...")
                        # You could add parsing logic here
                        break
            except:
                continue
        
        # Method 3: Fallback to cached state
        if callsign in bluesky_client.aircraft_states:
            cached = bluesky_client.aircraft_states[callsign]
            states[callsign] = {
                'callsign': callsign,
                'latitude': cached.latitude,
                'longitude': cached.longitude,
                'altitude_ft': cached.altitude_ft,
                'heading_deg': cached.heading_deg,
                'speed_kt': cached.speed_kt,
                'timestamp': time.time()
            }
            print(f"📦 Using cached state for {callsign}")
    
    return states
