#!/usr/bin/env python3
"""
Token-optimized test for SCAT destination fix with debug logging
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


class TokenOptimizedLLMClient(StreamlinedLLMClient):
    """Token-optimized LLM client with detailed debug logging"""
    
    def detect_conflicts_compact(self, context: ConflictContext) -> dict:
        """Ultra-compact conflict detection with detailed logging"""
        conflict_id = f"CDR_{int(time.time() * 1000)}"
        
        # Create minimal prompt
        prompt = self._create_minimal_prompt(conflict_id, context)
        
        # Debug logging
        self._debug_prompt_analysis(prompt, context)
        
        try:
            print(f"🤖 Calling LLM...")
            start_time = time.time()
            
            response = self._call_llm(prompt)
            
            end_time = time.time()
            
            # Debug response
            self._debug_response_analysis(response, end_time - start_time)
            
            parsed = self._parse_json_response(response)
            
            return parsed
            
        except Exception as e:
            print(f"❌ LLM failed: {e}")
            return self._create_minimal_fallback(conflict_id, str(e))
    
    def _create_minimal_prompt(self, conflict_id: str, context: ConflictContext) -> str:
        """Create ultra-minimal prompt to reduce tokens"""
        own = context.ownship_state
        dest = context.destination
        
        # Super compact intruder format
        tfc = []
        for i, intruder in enumerate(context.intruders[:3], 1):  # Max 3 intruders
            tfc.append(f"T{i}:{intruder.get('callsign', f'U{i}')} "
                      f"{intruder.get('latitude', 0):.2f},{intruder.get('longitude', 0):.2f} "
                      f"FL{int(intruder.get('altitude', 35000)/100)}")
        
        tfc_str = " | ".join(tfc) if tfc else "None"
        
        # Minimal destination info
        if dest:
            dest_str = f"{dest.get('name', 'DEST')} {dest.get('latitude', 0):.2f},{dest.get('longitude', 0):.2f}"
        else:
            dest_str = "None"
        
        return f"""ATC CDR: Guide {context.ownship_callsign} to dest avoiding conflicts.

OWN: {own.get('latitude', 0):.2f},{own.get('longitude', 0):.2f} FL{int(own.get('altitude', 35000)/100)} H{int(own.get('heading', 0))} S{int(own.get('speed', 400))}
TFC: {tfc_str}
DEST: {dest_str}

Rule: <5NM/<1000ft = conflict. Guide to dest safely.

JSON (cdr.v1):
{{
  "schema_version": "cdr.v1",
  "conflict_id": "{conflict_id}",
  "conflicts_detected": bool,
  "conflicts": [{{"intruder_callsign": "str", "time_to_conflict_minutes": 0.0, "predicted_min_separation_nm": 0.0, "predicted_min_vertical_separation_ft": 0.0, "conflict_type": "head_on"}}],
  "resolution": {{"resolution_type": "heading_change|altitude_change|speed_change|direct_to|no_action", "parameters": {{}}, "reasoning": "brief", "confidence": 0.8}}
}}"""
    
    def _debug_prompt_analysis(self, prompt: str, context: ConflictContext):
        """Analyze prompt for token optimization"""
        lines = prompt.split('\n')
        words = prompt.split()
        token_estimate = len(words) * 1.3  # Rough estimate
        
        print(f"📊 PROMPT ANALYSIS:")
        print(f"   📏 Length: {len(prompt)} chars, {len(lines)} lines")
        print(f"   🔤 Words: {len(words)}, Est. Tokens: {token_estimate:.0f}")
        print(f"   ✈️ Intruders: {len(context.intruders)}")
        print(f"   🎯 Destination: {'Yes' if context.destination else 'No'}")
        print(f"   📝 Preview: {prompt[:150].replace(chr(10), ' ')}...")
    
    def _debug_response_analysis(self, response: str, duration: float):
        """Analyze response for debugging"""
        words = response.split()
        token_estimate = len(words) * 1.3
        
        print(f"📊 RESPONSE ANALYSIS:")
        print(f"   ⏱️ Duration: {duration:.2f}s")
        print(f"   📏 Length: {len(response)} chars")
        print(f"   🔤 Words: {len(words)}, Est. Tokens: {token_estimate:.0f}")
        print(f"   📝 Preview: {response[:150].replace(chr(10), ' ')}...")
        
        # Check for JSON structure
        try:
            json.loads(response)
            print(f"   ✅ Valid JSON structure")
        except:
            print(f"   ❌ Invalid JSON structure")
    
    def _create_minimal_fallback(self, conflict_id: str, error: str) -> dict:
        """Create minimal fallback response"""
        return {
            'schema_version': 'cdr.v1',
            'conflict_id': conflict_id,
            'conflicts_detected': False,
            'conflicts': [],
            'resolution': {
                'resolution_type': 'no_action',
                'parameters': {},
                'reasoning': f'Error: {error[:50]}',
                'confidence': 0.0
            }
        }


def test_token_optimized_scenario():
    """Test token-optimized multi-aircraft scenario"""
    print("🧪 Testing Token-Optimized SCAT Destination CDR")
    print("=" * 60)
    
    try:
        # Load SCAT destination
        with open("data/sample_scat.json", 'r') as f:
            scat_data = json.load(f)
        
        tracks = []
        if 'plots' in scat_data:
            for plot in scat_data['plots']:
                if 'I062/105' in plot:
                    tracks.append({
                        'latitude': plot['I062/105']['lat'],
                        'longitude': plot['I062/105']['lon'],
                        'altitude': plot['I062/380']['subitem6'].get('altitude', 35000) if 'I062/380' in plot and 'subitem6' in plot['I062/380'] else 35000,
                        'heading': plot['I062/380']['subitem3'].get('mag_hdg', 90.0) if 'I062/380' in plot and 'subitem3' in plot['I062/380'] else 90.0
                    })
        
        if len(tracks) < 52:
            print(f"❌ Insufficient tracks: {len(tracks)}")
            return
        
        start_pos = tracks[0]
        destination = tracks[50]  # Track 51
        
        print(f"📍 SCAT Route: {len(tracks)} tracks")
        print(f"   Start: {start_pos['latitude']:.4f}, {start_pos['longitude']:.4f}")
        print(f"   Dest:  {destination['latitude']:.4f}, {destination['longitude']:.4f}")
        
        # Initialize token-optimized LLM
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3.1:8b")
        llm_client = TokenOptimizedLLMClient(config)
        
        # Initialize BlueSky
        bs_client = SimpleBlueSkyClient()
        bs_client.initialize()
        bs_client.reset()
        
        # Create ownship
        ownship_callsign = "SCAT1"
        success = bs_client.create_aircraft(
            acid=ownship_callsign,
            lat=start_pos['latitude'],
            lon=start_pos['longitude'],
            hdg=start_pos['heading'],
            alt=start_pos['altitude'],
            spd=450
        )
        
        if not success:
            print("❌ Failed to create ownship")
            return
        
        # Set destination
        bs_destination = Destination(
            name="SCAT_DEST",
            lat=destination['latitude'],
            lon=destination['longitude'],
            alt=destination['altitude']
        )
        bs_client.set_aircraft_destination(ownship_callsign, bs_destination)
        
        print(f"✅ Created ownship with SCAT destination")
        
        # Create LIMITED intruders for token optimization (only 2)
        intruders_created = 0
        for i in range(1, 3):  # Only 2 intruders to reduce tokens
            lat_offset = 0.02 * (i - 1.5)
            lon_offset = 0.025 * (i - 1.5)
            
            success = bs_client.create_aircraft(
                acid=f"TFC{i:02d}",
                lat=start_pos['latitude'] + lat_offset,
                lon=start_pos['longitude'] + lon_offset,
                hdg=(90 * i) % 360,
                alt=start_pos['altitude'] + (1000 * i),
                spd=400 + (i * 20)
            )
            
            if success:
                intruders_created += 1
        
        print(f"✅ Created {intruders_created} intruders (limited for token optimization)")
        
        # Step simulation
        bs_client.step_simulation(30)
        
        # Get aircraft states
        aircraft_states = bs_client.get_all_aircraft_states()
        
        # Prepare context
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
        
        # Format for LLM
        intruders_data = []
        for state in intruder_states:
            intruders_data.append({
                'callsign': state.id,
                'latitude': state.lat,
                'longitude': state.lon,
                'altitude': state.alt,
                'heading': state.hdg,
                'speed': state.tas
            })
        
        # Create context
        context = ConflictContext(
            ownship_callsign=ownship_callsign,
            ownship_state={
                'latitude': ownship_state.lat,
                'longitude': ownship_state.lon,
                'altitude': ownship_state.alt,
                'heading': ownship_state.hdg,
                'speed': ownship_state.tas
            },
            intruders=intruders_data,
            scenario_time=30.0,
            lookahead_minutes=5.0,
            destination={
                'name': 'SCAT_DEST',
                'latitude': destination['latitude'],
                'longitude': destination['longitude'],
                'altitude': destination['altitude']
            }
        )
        
        print(f"\n🤖 Testing Token-Optimized LLM Call...")
        
        # Call LLM with token optimization
        response = llm_client.detect_conflicts_compact(context)
        
        print(f"\n📥 LLM RESPONSE:")
        print(f"   Conflicts: {response.get('conflicts_detected', False)}")
        print(f"   Resolution: {response.get('resolution', {}).get('resolution_type', 'none')}")
        print(f"   Reasoning: {response.get('resolution', {}).get('reasoning', '')}")
        
        print(f"\n🎉 Token-optimized test completed successfully!")
        
        return response
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run token-optimized test"""
    print("🚀 Token-Optimized SCAT CDR Test")
    print("=" * 50)
    
    test_token_optimized_scenario()


if __name__ == "__main__":
    main()
