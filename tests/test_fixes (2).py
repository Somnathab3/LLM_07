#!/usr/bin/env python3
"""Test the fixed waypoint coordinate issue"""

import json
from src.cdr.pipeline.cdr_pipeline import CDRPipeline, PipelineConfig
from src.cdr.ai.llm_client import ConflictContext

def test_waypoint_extraction():
    """Test if waypoint coordinates are properly extracted"""
    
    # Create a pipeline instance
    config = PipelineConfig()
    pipeline = CDRPipeline(config, None, None)
    
    # Simulate the LLM response structure
    combined_response = {
        "schema_version": "cdr.v1",
        "conflict_id": "TEST_CONFLICT",
        "conflicts_detected": True,
        "conflicts": [
            {
                "intruder_callsign": "INTRUDER1",
                "time_to_conflict_minutes": 10,
                "conflict_type": "crossing"
            }
        ],
        "resolution": {
            "resolution_type": "reroute_via",
            "parameters": {
                "via_waypoint": {
                    "name": "AVOID1",
                    "lat": 42.32,
                    "lon": -87.35
                },
                "resume_to_destination": True
            },
            "reasoning": "Test waypoint extraction",
            "confidence": 0.8
        }
    }
    
    # Create a context
    context = ConflictContext(
        ownship_callsign='OWNSHIP',
        ownship_state={'latitude': 41.978, 'longitude': -87.904, 'altitude': 37000},
        intruders=[],
        scenario_time=300,
        lookahead_minutes=10,
        constraints={},
        destination={'name': 'DST', 'lat': 41.9742, 'lon': -87.9073}
    )
    
    # Simulate the conversion process
    resolution = combined_response.get('resolution', {})
    parameters = resolution.get('parameters', {})
    
    print("Original LLM parameters:")
    print(json.dumps(parameters, indent=2))
    
    # Test our conversion logic
    converted_params = {}
    
    # Handle route-aware parameters
    if 'waypoint_name' in parameters:
        converted_params['waypoint_name'] = parameters['waypoint_name']
        if 'lat' in parameters:
            converted_params['lat'] = parameters['lat']
        if 'lon' in parameters:
            converted_params['lon'] = parameters['lon']
    
    if 'via_waypoint' in parameters:
        converted_params['via_waypoint'] = parameters['via_waypoint']
        converted_params['resume_to_destination'] = parameters.get('resume_to_destination', True)
    
    print("\nConverted parameters:")
    print(json.dumps(converted_params, indent=2))
    
    # Test resolution structure
    result = {
        'aircraft_callsign': context.ownship_callsign,
        'resolution_type': resolution.get('resolution_type', 'no_action'),
        'reasoning': resolution.get('reasoning', 'Test resolution'),
        'confidence': resolution.get('confidence', 0.5),
        'parameters': parameters  # Add the original parameters
    }
    result.update(converted_params)
    
    print("\nFinal resolution structure:")
    print(json.dumps(result, indent=2))
    
    # Test extraction like in _apply_resolution_to_bluesky
    resolution_type = result.get('resolution_type', '')
    parameters = result.get('parameters', {})
    
    print(f"\nTesting extraction:")
    print(f"Resolution type: {resolution_type}")
    
    if resolution_type == "reroute_via":
        via = parameters.get('via_waypoint', {})
        waypoint_name = via.get('name', 'AVOID1')
        lat = via.get('lat', 0)
        lon = via.get('lon', 0)
        
        print(f"Via waypoint structure: {via}")
        print(f"Waypoint: {waypoint_name} at {lat:.4f},{lon:.4f}")
        
        if lat != 0 and lon != 0:
            print("✅ Waypoint coordinates extracted successfully!")
        else:
            print("❌ Waypoint coordinates still missing!")
    
    return result

if __name__ == "__main__":
    test_waypoint_extraction()
