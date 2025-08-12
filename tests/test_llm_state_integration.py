#!/usr/bin/env python3
"""
Test LLM integration with tracked aircraft states to verify 
the LLM receives updated aircraft data instead of stale data.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__) / "src"))

from src.cdr.simulation.bluesky_client import BlueSkyClient, BlueSkyConfig, AircraftState
from src.cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider
import time
import json

def test_llm_with_tracked_states():
    """Test that LLM receives updated aircraft states through state tracking"""
    
    print("🧪 Testing LLM integration with tracked aircraft states...")
    
    # Setup BlueSky client
    bs_config = BlueSkyConfig()
    bs_client = BlueSkyClient(bs_config)
    bs_client.connect()
    
    # Setup LLM client
    llm_config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama3.1:8b", 
        temperature=0.2,
        max_tokens=150,
        timeout=30.0
    )
    llm_client = LLMClient(llm_config)
    
    try:
        print("\n1. Creating test aircraft...")
        # Create aircraft with initial states
        bs_client._send_command('CRE AC001 A320 42.0 -87.9 0 10000 250')
        bs_client._send_command('CRE AC002 B737 42.1 -87.8 180 10000 240') 
        time.sleep(1)
        
        print("\n2. Getting initial aircraft states...")
        initial_states = bs_client.get_aircraft_states(['AC001', 'AC002'])
        
        print("Initial states:")
        for callsign, state in initial_states.items():
            print(f"  {callsign}: hdg={state.heading_deg}°, alt={state.altitude_ft}ft, spd={state.speed_kt}kt")
        
        print("\n3. Modifying aircraft states...")
        # Change aircraft parameters
        bs_client._send_command('HDG AC001 90')  # Change heading to 90°
        bs_client._send_command('ALT AC002 12000')  # Change altitude to 12000ft
        bs_client._send_command('SPD AC001 300')  # Change speed to 300kt
        time.sleep(1)
        
        print("\n4. Getting updated aircraft states...")
        updated_states = bs_client.get_aircraft_states(['AC001', 'AC002'])
        
        print("Updated states:")
        for callsign, state in updated_states.items():
            print(f"  {callsign}: hdg={state.heading_deg}°, alt={state.altitude_ft}ft, spd={state.speed_kt}kt")
        
        print("\n5. Testing LLM with updated states...")
        
        # Create a scenario prompt with current aircraft states
        aircraft_data = []
        for callsign, state in updated_states.items():
            aircraft_data.append({
                "callsign": callsign,
                "latitude": state.latitude,
                "longitude": state.longitude, 
                "altitude_ft": state.altitude_ft,
                "heading_deg": state.heading_deg,
                "speed_kt": state.speed_kt
            })
        
        scenario_prompt = f"""
AIRCRAFT STATE DATA:
{json.dumps(aircraft_data, indent=2)}

TASK: Analyze the current aircraft states above. What is the current heading of aircraft AC001?

Respond with a JSON object containing:
{{"analysis": "description of AC001's current heading", "ac001_heading": <heading_in_degrees>}}
"""
        
        print("Sending aircraft state data to LLM...")
        llm_response = llm_client.generate_response(scenario_prompt)
        
        print(f"\n6. LLM Response:")
        print(llm_response)
        
        # Parse LLM response to check if it correctly identified the updated heading
        try:
            if "{" in llm_response and "}" in llm_response:
                # Extract JSON from response
                start = llm_response.find("{")
                end = llm_response.rfind("}") + 1
                json_part = llm_response[start:end]
                parsed = json.loads(json_part)
                
                if "ac001_heading" in parsed:
                    llm_heading = float(parsed["ac001_heading"])
                    expected_heading = 90.0  # We set AC001 to 90°
                    
                    print(f"\n7. Verification:")
                    print(f"   Expected AC001 heading: {expected_heading}°")
                    print(f"   LLM reported heading: {llm_heading}°")
                    
                    if abs(llm_heading - expected_heading) < 1.0:
                        print("   ✅ SUCCESS: LLM correctly identified updated heading!")
                        print("   🎉 State tracking is working - LLM receives current data!")
                    else:
                        print("   ❌ FAILURE: LLM reported incorrect heading")
                        print("   🚨 LLM may still be receiving stale data")
                else:
                    print("   ⚠️ Could not extract heading from LLM response")
        except Exception as e:
            print(f"   ❌ Error parsing LLM response: {e}")
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        
    finally:
        print("\n8. Cleanup...")
        bs_client.disconnect()

if __name__ == "__main__":
    test_llm_with_tracked_states()
