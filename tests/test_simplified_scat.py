#!/usr/bin/env python3
"""
Simplified SCAT destination test without token optimization
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


class SimplifiedLLMClient(StreamlinedLLMClient):
    """Simplified LLM client without token optimization"""
    
    def detect_conflicts_simple(self, context: ConflictContext) -> dict:
        """Simple conflict detection"""
        conflict_id = f"CDR_{int(time.time() * 1000)}"
        
        # Create simple prompt
        prompt = self._create_simple_prompt(conflict_id, context)
        
        try:
            print(f"🤖 Calling LLM...")
            start_time = time.time()
            
            response = self._call_llm(prompt)
            
            end_time = time.time()
            print(f"✅ LLM response received in {end_time - start_time:.2f}s")
            
            parsed = self._parse_json_response(response)
            return parsed
            
        except Exception as e:
            print(f"❌ LLM failed: {e}")
            return self._create_fallback(conflict_id, str(e))
    
    def _create_simple_prompt(self, conflict_id: str, context: ConflictContext) -> str:
        """Create simple prompt"""
        ownship = context.ownship_state
        
        # Simple destination
        dest_text = "No destination"
        if context.destination:
            dest_text = f"Destination: {context.destination.get('latitude', 0):.4f}, {context.destination.get('longitude', 0):.4f}"
        
        # Simple intruders
        intruders_text = "No traffic"
        if context.intruders:
            lines = []
            for i, intruder in enumerate(context.intruders[:3], 1):  # Limit to 3
                lines.append(f"Traffic {i}: {intruder.get('callsign', f'TFC{i}')} at {intruder.get('latitude', 0):.4f}, {intruder.get('longitude', 0):.4f}")
            intruders_text = "\n".join(lines)
        
        return f"""Air traffic controller task: Guide {context.ownship_callsign} safely.

Ownship: {context.ownship_callsign}
Position: {ownship.get('latitude', 0):.4f}, {ownship.get('longitude', 0):.4f}
Altitude: {ownship.get('altitude', 35000)} ft

{dest_text}

{intruders_text}

Return JSON only:
{{
  "schema_version": "cdr.v1",
  "conflict_id": "{conflict_id}",
  "conflicts_detected": false,
  "conflicts": [],
  "resolution": {{
    "resolution_type": "no_action",
    "parameters": {{}},
    "reasoning": "No conflicts detected",
    "confidence": 0.8
  }}
}}"""
    
    def _create_fallback(self, conflict_id: str, error: str) -> dict:
        """Create fallback response"""
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


def test_simplified_scenario():
    """Test simplified scenario"""
    print("🧪 Testing Simplified SCAT Destination")
    print("=" * 50)
    
    try:
        # Load SCAT data
        with open("data/sample_scat.json", 'r') as f:
            scat_data = json.load(f)
        
        tracks = []
        if 'plots' in scat_data:
            for plot in scat_data['plots']:
                if 'I062/105' in plot:
                    tracks.append({
                        'latitude': plot['I062/105']['lat'],
                        'longitude': plot['I062/105']['lon'],
                        'altitude': plot['I062/380']['subitem6'].get('altitude', 35000) if 'I062/380' in plot and 'subitem6' in plot['I062/380'] else 35000
                    })
        
        if len(tracks) < 52:
            print(f"❌ Insufficient tracks: {len(tracks)}")
            return
        
        start_pos = tracks[0]
        destination = tracks[50]  # Track 51
        
        print(f"📍 SCAT Route: {len(tracks)} tracks")
        print(f"   Start: {start_pos['latitude']:.4f}, {start_pos['longitude']:.4f}")
        print(f"   Dest:  {destination['latitude']:.4f}, {destination['longitude']:.4f}")
        
        # Initialize simplified LLM
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3.1:8b")
        llm_client = SimplifiedLLMClient(config)
        
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
            hdg=90,
            alt=start_pos['altitude'],
            spd=450
        )
        
        if not success:
            print("❌ Failed to create ownship")
            return
        
        # Create 1 intruder only
        success = bs_client.create_aircraft(
            acid="TFC01",
            lat=start_pos['latitude'] + 0.01,
            lon=start_pos['longitude'] + 0.01,
            hdg=180,
            alt=start_pos['altitude'] + 1000,
            spd=400
        )
        
        if not success:
            print("❌ Failed to create intruder")
            return
        
        print(f"✅ Created ownship and 1 intruder")
        
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
                'altitude': state.alt
            })
        
        # Create context
        context = ConflictContext(
            ownship_callsign=ownship_callsign,
            ownship_state={
                'latitude': ownship_state.lat,
                'longitude': ownship_state.lon,
                'altitude': ownship_state.alt
            },
            intruders=intruders_data,
            scenario_time=30.0,
            lookahead_minutes=5.0,
            destination={
                'latitude': destination['latitude'],
                'longitude': destination['longitude']
            }
        )
        
        # Call LLM
        response = llm_client.detect_conflicts_simple(context)
        
        print(f"\n📥 LLM RESPONSE:")
        print(f"   Conflicts: {response.get('conflicts_detected', False)}")
        print(f"   Resolution: {response.get('resolution', {}).get('resolution_type', 'none')}")
        
        print(f"\n🎉 Simplified test completed!")
        
        return response
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run simplified test"""
    print("🚀 Simplified SCAT Test")
    print("=" * 30)
    
    test_simplified_scenario()


if __name__ == "__main__":
    main()
