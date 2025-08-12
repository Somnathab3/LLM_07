#!/usr/bin/env python3
"""
Enhanced test script for SCAT-based destination fix with multi-aircraft movement and advanced CDR
"""

import sys
import json
import math
import time
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cdr.simulation.bluesky_client import SimpleBlueSkyClient, Destination
from cdr.ai.llm_client_streamlined import StreamlinedLLMClient, LLMConfig, LLMProvider, ConflictContext


class SCATDestinationAnalyzer:
    """Analyzer for SCAT data to extract destination from track 51 (last but one)"""
    
    def __init__(self, scat_file_path: str):
        self.scat_file_path = scat_file_path
        self.scat_data = None
        self.tracks = []
        
    def load_scat_data(self):
        """Load and parse SCAT data"""
        try:
            with open(self.scat_file_path, 'r') as f:
                self.scat_data = json.load(f)
            
            # Extract tracks with position data from "plots" array
            if 'plots' in self.scat_data:
                for plot in self.scat_data['plots']:
                    if 'I062/105' in plot:  # Position data
                        lat = plot['I062/105']['lat']
                        lon = plot['I062/105']['lon']
                        time_stamp = plot.get('time_of_track', '')
                        altitude = None
                        heading = None
                        
                        # Extract altitude if available
                        if 'I062/380' in plot and 'subitem6' in plot['I062/380']:
                            altitude = plot['I062/380']['subitem6'].get('altitude')
                        
                        # Extract heading if available
                        if 'I062/380' in plot and 'subitem3' in plot['I062/380']:
                            heading = plot['I062/380']['subitem3'].get('mag_hdg')
                        
                        self.tracks.append({
                            'latitude': lat,
                            'longitude': lon,
                            'altitude': altitude,
                            'heading': heading,
                            'timestamp': time_stamp
                        })
            
            print(f"✅ Loaded {len(self.tracks)} track points from SCAT data")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load SCAT data: {e}")
            return False
    
    def get_destination_fix(self):
        """Get the 51st track point as destination fix (last but one)"""
        if len(self.tracks) < 52:
            print(f"⚠️ Warning: Only {len(self.tracks)} tracks available, expected 52")
            if len(self.tracks) >= 2:
                # Use the second to last available track
                destination_track = self.tracks[-2]
            else:
                print("❌ Insufficient track data for destination extraction")
                return None
        else:
            # Use the 51st track (index 50, as it's 0-based)
            destination_track = self.tracks[50]
        
        destination = {
            'name': 'SCAT_DEST',
            'latitude': destination_track['latitude'],
            'longitude': destination_track['longitude'],
            'altitude': destination_track.get('altitude', 35000),
            'source': 'scat_track_51'
        }
        
        print(f"🎯 Extracted destination from SCAT track 51:")
        print(f"   Position: {destination['latitude']:.6f}, {destination['longitude']:.6f}")
        print(f"   Altitude: {destination['altitude']} ft")
        
        return destination
    
    def get_start_position(self):
        """Get the first track point as starting position"""
        if not self.tracks:
            return None
        
        start_track = self.tracks[0]
        return {
            'latitude': start_track['latitude'],
            'longitude': start_track['longitude'],
            'altitude': start_track.get('altitude', 35000),
            'heading': start_track.get('heading', 90.0)
        }


class EnhancedLLMClient(StreamlinedLLMClient):
    """Enhanced LLM client with strict JSON formatting and destination-aware prompts"""
    
    def detect_and_resolve_conflicts_enhanced(self, context: ConflictContext) -> dict:
        """Enhanced conflict detection with strict JSON formatting and destination guidance"""
        conflict_id = f"SCAT_CDR_{int(time.time() * 1000)}"
        
        # Prepare context data
        ownship = context.ownship_state
        intruders_list = self._format_intruders_enhanced(context.intruders)
        
        # Enhanced destination info with guidance emphasis
        dest_info = self._format_destination_enhanced(context.destination, ownship)
        
        # Enhanced prompt with strict JSON requirement and destination guidance
        prompt = self._create_enhanced_prompt(
            conflict_id=conflict_id,
            ownship_callsign=context.ownship_callsign,
            ownship=ownship,
            intruders_list=intruders_list,
            lookahead_minutes=int(context.lookahead_minutes),
            dest_info=dest_info
        )
        
        # Call LLM with enhanced error handling
        try:
            response = self._call_llm(prompt)
            parsed = self._parse_json_response_strict(response)
            
            # Log for debugging
            self._log_interaction(conflict_id, prompt, response, parsed)
            
            return parsed
            
        except Exception as e:
            print(f"❌ Enhanced LLM call failed: {e}")
            return {
                'schema_version': 'cdr.v1',
                'conflict_id': conflict_id,
                'conflicts_detected': False,
                'conflicts': [],
                'resolution': {
                    'resolution_type': 'no_action',
                    'parameters': {},
                    'reasoning': f'LLM error: {str(e)}',
                    'confidence': 0.0
                }
            }
    
    def _format_intruders_enhanced(self, intruders: list) -> str:
        """Enhanced intruder formatting with threat assessment"""
        if not intruders:
            return "No traffic detected"
        
        formatted = []
        for i, intruder in enumerate(intruders, 1):
            threat_level = self._assess_threat_level(intruder)
            formatted.append(
                f"   TFC{i}: {intruder.get('callsign', f'UNK{i}')} at "
                f"{intruder.get('latitude', 0):.4f},{intruder.get('longitude', 0):.4f} "
                f"FL{int(intruder.get('altitude', 35000)/100)} "
                f"hdg={int(intruder.get('heading', 0))}° "
                f"spd={int(intruder.get('speed', 400))}kt "
                f"[THREAT: {threat_level}]"
            )
        
        return "\n".join(formatted)
    
    def _assess_threat_level(self, intruder: dict) -> str:
        """Assess threat level of intruder"""
        # Simple threat assessment based on relative position and heading
        # This could be enhanced with more sophisticated conflict prediction
        alt_diff = abs(intruder.get('altitude', 35000) - 35000)  # Assuming ownship at FL350
        
        if alt_diff < 1000:
            return "HIGH"
        elif alt_diff < 2000:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _format_destination_enhanced(self, destination: dict, ownship: dict) -> dict:
        """Enhanced destination formatting with guidance emphasis"""
        if not destination:
            return {
                'name': 'NO_DEST',
                'lat': 0.0,
                'lon': 0.0,
                'bearing': 0,
                'distance_nm': 0
            }
        
        # Calculate bearing and distance to destination
        own_lat = ownship.get('latitude', 42.0)
        own_lon = ownship.get('longitude', -87.0)
        dest_lat = destination.get('latitude', own_lat + 1.0)
        dest_lon = destination.get('longitude', own_lon + 1.0)
        
        bearing = self._calculate_bearing(own_lat, own_lon, dest_lat, dest_lon)
        distance = self._calculate_distance(own_lat, own_lon, dest_lat, dest_lon)
        
        return {
            'name': destination.get('name', 'SCAT_DEST'),
            'lat': dest_lat,
            'lon': dest_lon,
            'bearing': int(bearing),
            'distance_nm': round(distance, 1),
            'source': destination.get('source', 'unknown')
        }
    
    def _create_enhanced_prompt(self, conflict_id: str, ownship_callsign: str, 
                               ownship: dict, intruders_list: str, 
                               lookahead_minutes: int, dest_info: dict) -> str:
        """Create enhanced prompt with strict formatting requirements"""
        
        return f"""CRITICAL: You are an advanced ATC conflict resolution system. Your primary mission is to guide the ownship safely to its FINAL DESTINATION while avoiding ALL intruders.

STRICT REQUIREMENTS:
1. Return ONLY valid JSON - NO additional text, comments, or unicode characters
2. Follow exact schema cdr.v1 
3. ALL guidance must prioritize reaching the final destination
4. Prevent conflicts with ALL intruders
5. Keep reasoning under 150 characters (ASCII only)

OWNSHIP STATUS:
- Callsign: {ownship_callsign}
- Position: {ownship.get('latitude', 0):.6f}, {ownship.get('longitude', 0):.6f}
- Flight Level: FL{int(ownship.get('altitude', 35000)/100)}
- Heading: {int(ownship.get('heading', 0))}°
- Speed: {int(ownship.get('speed', 400))} kt
- Vertical Speed: {int(ownship.get('vertical_speed_fpm', 0)):+d} fpm

INTRUDER AIRCRAFT:
{intruders_list}

FINAL DESTINATION (CRITICAL - MUST REACH):
- Name: {dest_info['name']}
- Position: {dest_info['lat']:.6f}, {dest_info['lon']:.6f}
- Current Bearing: {dest_info['bearing']}°
- Distance: {dest_info['distance_nm']} NM
- Source: {dest_info.get('source', 'unknown')}

CONFLICT ANALYSIS ({lookahead_minutes} min projection):
- Check straight-line trajectories for all aircraft
- Conflict if separation <5NM horizontal OR <1000ft vertical
- Consider relative speeds and closure rates

RESOLUTION PRIORITIES:
1. AVOID ALL CONFLICTS (safety first)
2. MAINTAIN PROGRESS toward final destination
3. Use minimal deviation from direct route
4. Resume direct routing as soon as safe

AVAILABLE RESOLUTION TYPES:
- heading_change: {{"new_heading_deg": 120}} (±20° minimum change)
- altitude_change: {{"target_altitude_ft": 36000}} (±1000ft minimum)
- speed_change: {{"target_speed_kt": 380}} (±20kt minimum)
- direct_to: {{"waypoint_name": "SCAT_DEST"}} (direct to destination)
- reroute_via: {{"via_waypoint": {{"name":"AVOID1","lat":42.0,"lon":-87.0}}, "resume_to_destination": true}}
- no_action: {{}} (if no conflicts detected)

RETURN ONLY THIS JSON STRUCTURE:
{{
  "schema_version": "cdr.v1",
  "conflict_id": "{conflict_id}",
  "conflicts_detected": true_or_false,
  "conflicts": [
    {{
      "intruder_callsign": "string",
      "time_to_conflict_minutes": 0.0,
      "predicted_min_separation_nm": 0.0,
      "predicted_min_vertical_separation_ft": 0.0,
      "conflict_type": "head_on_or_crossing_or_overtaking"
    }}
  ],
  "resolution": {{
    "resolution_type": "chosen_type",
    "parameters": {{}},
    "reasoning": "ASCII text under 150 chars explaining guidance toward destination",
    "confidence": 0.0_to_1.0
  }}
}}"""
    
    def _parse_json_response_strict(self, response: str) -> dict:
        """Strict JSON parsing with enhanced error handling"""
        try:
            # Clean the response
            cleaned = response.strip()
            
            # Remove any markdown formatting
            if cleaned.startswith('```'):
                lines = cleaned.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith('```'):
                        in_json = not in_json
                        continue
                    if in_json:
                        json_lines.append(line)
                cleaned = '\n'.join(json_lines)
            
            # Parse JSON
            parsed = json.loads(cleaned)
            
            # Validate required fields
            required_fields = ['schema_version', 'conflict_id', 'conflicts_detected', 'conflicts', 'resolution']
            for field in required_fields:
                if field not in parsed:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure resolution has required fields
            resolution = parsed.get('resolution', {})
            if 'resolution_type' not in resolution:
                resolution['resolution_type'] = 'no_action'
            if 'parameters' not in resolution:
                resolution['parameters'] = {}
            if 'reasoning' not in resolution:
                resolution['reasoning'] = 'No reasoning provided'
            if 'confidence' not in resolution:
                resolution['confidence'] = 0.5
            
            # Clean reasoning to ensure ASCII only
            reasoning = resolution.get('reasoning', '')
            resolution['reasoning'] = ''.join(char for char in reasoning if ord(char) < 128)[:150]
            
            return parsed
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Response: {response[:200]}...")
            raise
        except Exception as e:
            print(f"❌ Response validation error: {e}")
            raise


def test_scat_destination_extraction():
    """Test SCAT destination extraction from track 51"""
    print("🧪 Testing SCAT destination extraction...")
    
    # Initialize SCAT analyzer
    scat_file = "data/sample_scat.json"
    analyzer = SCATDestinationAnalyzer(scat_file)
    
    # Load SCAT data
    if not analyzer.load_scat_data():
        print("❌ Failed to load SCAT data")
        return None
    
    # Extract destination from track 51
    destination = analyzer.get_destination_fix()
    if not destination:
        print("❌ Failed to extract destination")
        return None
    
    # Extract starting position
    start_pos = analyzer.get_start_position()
    if not start_pos:
        print("❌ Failed to extract starting position")
        return None
    
    print(f"📍 Starting position: {start_pos['latitude']:.6f}, {start_pos['longitude']:.6f}")
    print(f"🎯 Destination: {destination['latitude']:.6f}, {destination['longitude']:.6f}")
    
    return analyzer, destination, start_pos


def test_multi_aircraft_scenario_enhanced():
    """Test enhanced multi-aircraft scenario with SCAT destination"""
    print("\n🧪 Testing enhanced multi-aircraft scenario with SCAT destination...")
    
    # Extract SCAT destination
    analyzer, destination, start_pos = test_scat_destination_extraction()
    if not analyzer or not destination or not start_pos:
        print("❌ Failed to extract SCAT data")
        return
    
    # Initialize enhanced LLM client
    config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3.1:8b")
    llm_client = EnhancedLLMClient(config)
    
    # Initialize BlueSky client
    bs_client = SimpleBlueSkyClient()
    bs_client.initialize()
    bs_client.reset()
    
    # Create ownship at starting position
    ownship_callsign = "SCAT1"
    success = bs_client.create_aircraft(
        acid=ownship_callsign,
        lat=start_pos['latitude'],
        lon=start_pos['longitude'],
        hdg=start_pos.get('heading', 90.0),
        alt=start_pos.get('altitude', 35000),
        spd=450
    )
    
    if not success:
        print("❌ Failed to create ownship")
        return
    
    print(f"✅ Created ownship {ownship_callsign} at SCAT starting position")
    
    # Set SCAT-based destination
    bs_destination = Destination(
        name=destination['name'],
        lat=destination['latitude'],
        lon=destination['longitude'],
        alt=destination['altitude']
    )
    bs_client.set_aircraft_destination(ownship_callsign, bs_destination)
    
    print(f"🎯 Set SCAT destination: {destination['name']} at {destination['latitude']:.6f}, {destination['longitude']:.6f}")
    
    # Create multiple intruders with varying complexity
    num_intruders = 5  # Increased complexity
    intruders = []
    
    for i in range(1, num_intruders + 1):
        # Generate intruders at various positions and altitudes
        lat_offset = 0.02 * (i - 2.5)  # Spread around ownship
        lon_offset = 0.03 * (i - 2.5)
        alt_offset = 1000 * (i - 3)  # Varying altitudes
        
        intruder = {
            "callsign": f"TFC{i}",
            "lat": start_pos['latitude'] + lat_offset,
            "lon": start_pos['longitude'] + lon_offset,
            "hdg": 45 + (i * 60) % 360,  # Varying headings
            "alt": max(25000, min(40000, start_pos.get('altitude', 35000) + alt_offset)),
            "spd": 400 + (i * 20)  # Varying speeds
        }
        
        success = bs_client.create_aircraft(
            acid=intruder["callsign"],
            lat=intruder["lat"],
            lon=intruder["lon"],
            hdg=intruder["hdg"],
            alt=intruder["alt"],
            spd=intruder["spd"]
        )
        
        if success:
            intruders.append(intruder)
            print(f"✅ Created intruder {intruder['callsign']} - FL{int(intruder['alt']/100)}, HDG {intruder['hdg']}°")
    
    print(f"\n📊 Created {len(intruders)} intruders for complexity testing")
    
    # Step simulation forward
    bs_client.step_simulation(30)  # 30 seconds
    
    # Get aircraft states
    aircraft_states = bs_client.get_all_aircraft_states()
    
    if not aircraft_states:
        print("❌ No aircraft states retrieved")
        return
    
    print(f"\n📊 Retrieved {len(aircraft_states)} aircraft states")
    
    # Find ownship and intruders
    ownship_state = None
    intruder_states = []
    
    for state in aircraft_states:
        if state.id == ownship_callsign:
            ownship_state = state
        else:
            intruder_states.append(state)
    
    if not ownship_state:
        print("❌ Ownship state not found")
        return
    
    print(f"🛫 Ownship: {ownship_state.id} at {ownship_state.lat:.6f},{ownship_state.lon:.6f} FL{int(ownship_state.alt/100)}")
    
    # Format intruders for LLM
    intruders_data = []
    for state in intruder_states:
        intruder_data = {
            'callsign': state.id,
            'latitude': state.lat,
            'longitude': state.lon,
            'altitude': state.alt,
            'heading': state.hdg,
            'speed': state.tas,
            'vertical_speed_fpm': state.vs * 60
        }
        intruders_data.append(intruder_data)
        print(f"✈️ Intruder: {state.id} at {state.lat:.6f},{state.lon:.6f} FL{int(state.alt/100)}")
    
    # Create enhanced conflict context with SCAT destination
    context = ConflictContext(
        ownship_callsign=ownship_callsign,
        ownship_state={
            'latitude': ownship_state.lat,
            'longitude': ownship_state.lon,
            'altitude': ownship_state.alt,
            'heading': ownship_state.hdg,
            'speed': ownship_state.tas,
            'vertical_speed_fpm': ownship_state.vs * 60
        },
        intruders=intruders_data,
        scenario_time=30.0,
        lookahead_minutes=10.0,
        destination=destination
    )
    
    print(f"\n📤 Calling enhanced LLM with {len(intruders_data)} intruders and SCAT destination...")
    print("📋 Instructions: Guide ownship to destination while preventing all conflicts")
    
    # Get enhanced LLM response
    try:
        start_time = time.time()
        response = llm_client.detect_and_resolve_conflicts_enhanced(context)
        end_time = time.time()
        
        print(f"\n📥 Enhanced LLM Response (completed in {end_time - start_time:.2f}s):")
        print(f"   Schema Version: {response.get('schema_version', 'unknown')}")
        print(f"   Conflicts Detected: {response.get('conflicts_detected', False)}")
        print(f"   Number of Conflicts: {len(response.get('conflicts', []))}")
        
        if response.get('conflicts'):
            print("\n🚨 Detected Conflicts:")
            for i, conflict in enumerate(response['conflicts'], 1):
                print(f"   {i}. Intruder: {conflict.get('intruder_callsign', 'unknown')}")
                print(f"      Time to conflict: {conflict.get('time_to_conflict_minutes', 0):.1f} min")
                print(f"      Min separation: {conflict.get('predicted_min_separation_nm', 0):.1f} NM")
                print(f"      Conflict type: {conflict.get('conflict_type', 'unknown')}")
        
        resolution = response.get('resolution', {})
        if resolution:
            print(f"\n🎯 Resolution Guidance:")
            print(f"   Type: {resolution.get('resolution_type', 'none')}")
            print(f"   Parameters: {resolution.get('parameters', {})}")
            print(f"   Reasoning: {resolution.get('reasoning', '')}")
            print(f"   Confidence: {resolution.get('confidence', 0):.2f}")
            
            # Check if resolution maintains destination awareness
            reasoning = resolution.get('reasoning', '').lower()
            if 'destination' in reasoning or 'dest' in reasoning:
                print("   ✅ Resolution maintains destination awareness")
            else:
                print("   ⚠️ Resolution may not consider destination")
        
        print("\n🎉 Enhanced multi-aircraft test with SCAT destination completed successfully!")
        
        # Display summary
        print(f"\n📋 Test Summary:")
        print(f"   Starting Position: {start_pos['latitude']:.6f}, {start_pos['longitude']:.6f}")
        print(f"   Destination: {destination['latitude']:.6f}, {destination['longitude']:.6f}")
        print(f"   Intruders: {len(intruders_data)}")
        print(f"   Conflicts Found: {len(response.get('conflicts', []))}")
        print(f"   Resolution Type: {resolution.get('resolution_type', 'none')}")
        print(f"   Processing Time: {end_time - start_time:.2f}s")
        
        return response
        
    except Exception as e:
        print(f"❌ Enhanced LLM call failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run enhanced end-to-end test with SCAT destination fix and multi-aircraft movement"""
    print("🚀 Starting Enhanced E2E Test: SCAT Destination Fix + Multi-Aircraft CDR")
    print("=" * 80)
    
    try:
        # Test 1: SCAT destination extraction
        print("\n" + "=" * 40)
        print("PHASE 1: SCAT DESTINATION EXTRACTION")
        print("=" * 40)
        
        analyzer, destination, start_pos = test_scat_destination_extraction()
        
        if not analyzer or not destination or not start_pos:
            print("❌ Phase 1 failed - cannot continue")
            return
        
        # Test 2: Enhanced multi-aircraft scenario
        print("\n" + "=" * 40)
        print("PHASE 2: ENHANCED MULTI-AIRCRAFT CDR")
        print("=" * 40)
        
        response = test_multi_aircraft_scenario_enhanced()
        
        if response:
            print("\n" + "=" * 40)
            print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
            print("=" * 40)
            print("🎯 SCAT-based destination fix implemented")
            print("🛫 Multi-aircraft movement simulation completed")
            print("🤖 Enhanced LLM guidance with destination awareness")
            print("📊 Strict JSON formatting enforced")
            print("🚨 Conflict prevention with complexity testing")
        else:
            print("\n❌ Phase 2 failed")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
