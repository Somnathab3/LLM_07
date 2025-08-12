#!/usr/bin/env python3
"""Quick test for combined LLM functionality"""

import sys
sys.path.append('.')

from src.cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider, ConflictContext

def test_combined():
    # Setup LLM client
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama3.1:8b",
        enable_combined_mode=True,
        temperature=0.2,
        num_predict=160
    )
    
    client = LLMClient(config)
    
    # Create test context with closer aircraft (should cause conflict)
    context = ConflictContext(
        ownship_callsign="OWNSHIP",
        ownship_state={
            'latitude': 41.978,
            'longitude': -87.904,
            'altitude': 35000,
            'heading': 90,   # Eastbound
            'speed': 450
        },
        intruders=[{
            'callsign': 'INTRUDER1',
            'latitude': 41.978,        # Same latitude
            'longitude': -87.800,      # 0.1 degrees east = ~6 NM
            'altitude': 35000,         # Same altitude
            'heading': 270,            # Westbound - head-on!
            'speed': 450
        }],
        scenario_time=300.0,
        lookahead_minutes=10.0,
        constraints={},
        destination={
            "name": "DST",
            "lat": 41.9742,  # Chicago O'Hare coordinates  
            "lon": -87.9073
        }
    )
    
    # Test combined detection and resolution
    result = client.detect_and_resolve(context, "TEST_CONFLICT_001")
    
    print("=== COMBINED RESULT ===")
    print(result)
    
    return result

if __name__ == "__main__":
    test_combined()
