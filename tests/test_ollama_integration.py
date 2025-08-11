#!/usr/bin/env python3
"""
Test script for Ollama LLM Client integration
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider, ConflictContext


def test_ollama_connection():
    """Test basic Ollama connection"""
    print("=" * 60)
    print("TESTING OLLAMA CONNECTION")
    print("=" * 60)
    
    try:
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model_name="llama3.1:8b",
            base_url="http://localhost:11434",
            temperature=0.1,
            max_tokens=300,  # Reduced for faster responses
            timeout=60.0     # Increased timeout
        )
        
        client = LLMClient(config)
        
        # Test connection
        test_result = client.test_connection()
        
        print(f"Server Connected: {test_result['server_connected']}")
        print(f"Model Available: {test_result['model_available']}")
        print(f"Generation Test: {test_result['generation_test']}")
        
        if test_result['error_messages']:
            print("Error Messages:")
            for error in test_result['error_messages']:
                print(f"  - {error}")
        
        # Get model info
        model_info = client.get_model_info()
        print(f"\nModel Info: {json.dumps(model_info, indent=2)}")
        
        return client if test_result['generation_test'] else None
        
    except Exception as e:
        print(f"Connection test failed: {e}")
        return None


def test_conflict_detection(client: LLMClient):
    """Test conflict detection functionality"""
    print("\n" + "=" * 60)
    print("TESTING CONFLICT DETECTION")
    print("=" * 60)
    
    try:
        # Create test scenario
        context = ConflictContext(
            ownship_callsign="AAL123",
            ownship_state={
                'latitude': 40.7128,
                'longitude': -74.0060,
                'altitude': 35000,
                'heading': 90,
                'speed': 450
            },
            intruders=[
                {
                    'callsign': 'UAL456',
                    'latitude': 40.7200,
                    'longitude': -74.0000,
                    'altitude': 35000,
                    'heading': 270,
                    'speed': 440,
                    'position': '40.7200°, -74.0000°'
                },
                {
                    'callsign': 'DAL789',
                    'latitude': 40.7000,
                    'longitude': -74.0120,
                    'altitude': 36000,
                    'heading': 180,
                    'speed': 460,
                    'position': '40.7000°, -74.0120°'
                }
            ],
            scenario_time=0.0,
            lookahead_minutes=5.0,  # Reduced for faster processing
            constraints={}
        )
        
        print("Scenario:")
        print(f"  Ownship: {context.ownship_callsign} at FL350, heading 90°")
        print(f"  Intruders: {len(context.intruders)} aircraft")
        
        # Run conflict detection with simplified prompts for faster testing
        print("  Using simplified prompts for faster testing...")
        result = client.detect_conflicts(context, use_simple_prompt=True)
        
        print(f"\nDetection Result:")
        print(json.dumps(result, indent=2))
        
        return result
        
    except Exception as e:
        print(f"Conflict detection test failed: {e}")
        return None


def test_resolution_generation(client: LLMClient, conflict_result):
    """Test resolution generation functionality"""
    print("\n" + "=" * 60)
    print("TESTING RESOLUTION GENERATION")
    print("=" * 60)
    
    try:
        # Create test scenario for resolution
        context = ConflictContext(
            ownship_callsign="AAL123",
            ownship_state={
                'latitude': 40.7128,
                'longitude': -74.0060,
                'altitude': 35000,
                'heading': 90,
                'speed': 450
            },
            intruders=[
                {
                    'callsign': 'UAL456',
                    'latitude': 40.7200,
                    'longitude': -74.0000,
                    'altitude': 35000,
                    'heading': 270,
                    'speed': 440
                }
            ],
            scenario_time=0.0,
            lookahead_minutes=5.0,  # Reduced for faster processing
            constraints={}
        )
        
        # If no conflicts from detection, create a mock conflict
        if not conflict_result or not conflict_result.get('conflicts_detected'):
            conflict_info = {
                'conflicts_detected': True,
                'conflicts': [
                    {
                        'intruder_callsign': 'UAL456',
                        'time_to_conflict_minutes': 5.2,
                        'predicted_min_separation_nm': 3.1,
                        'conflict_type': 'head_on'
                    }
                ]
            }
        else:
            conflict_info = conflict_result
        
        print("Generating resolution for conflict...")
        
        # Generate resolution with simplified prompts for faster testing
        resolution = client.generate_resolution(context, conflict_info, use_simple_prompt=True)
        
        print(f"\nResolution Result:")
        print(f"  Success: {resolution.success}")
        print(f"  Type: {resolution.resolution_type}")
        print(f"  Parameters: {resolution.parameters}")
        print(f"  Reasoning: {resolution.reasoning}")
        print(f"  Confidence: {resolution.confidence}")
        
        return resolution
        
    except Exception as e:
        print(f"Resolution generation test failed: {e}")
        return None


def test_error_handling(client: LLMClient):
    """Test error handling capabilities"""
    print("\n" + "=" * 60)
    print("TESTING ERROR HANDLING")
    print("=" * 60)
    
    try:
        # Test with invalid context
        invalid_context = ConflictContext(
            ownship_callsign="",
            ownship_state={},
            intruders=[],
            scenario_time=0.0,
            lookahead_minutes=10.0,
            constraints={}
        )
        
        print("Testing with invalid/empty context...")
        result = client.detect_conflicts(invalid_context)
        print(f"Result: {result}")
        
        # Test resolution with no conflicts
        print("\nTesting resolution with no conflicts...")
        resolution = client.generate_resolution(invalid_context, {"conflicts_detected": False, "conflicts": []})
        print(f"Resolution: {resolution.success}, {resolution.resolution_type}")
        
        print("Error handling tests completed successfully")
        
    except Exception as e:
        print(f"Error handling test failed: {e}")


def main():
    """Main test function"""
    print("OLLAMA LLM CLIENT INTEGRATION TEST")
    print("Please ensure Ollama is running with llama3.1:8b model")
    print("Start Ollama: ollama run llama3.1:8b")
    
    # Test connection
    client = test_ollama_connection()
    if not client:
        print("❌ Connection test failed. Please check Ollama setup.")
        return
    
    print("✅ Connection test passed!")
    
    # Test conflict detection
    conflict_result = test_conflict_detection(client)
    if conflict_result is not None:
        print("✅ Conflict detection test completed!")
    else:
        print("⚠️  Conflict detection test had issues")
    
    # Test resolution generation
    resolution_result = test_resolution_generation(client, conflict_result)
    if resolution_result:
        print("✅ Resolution generation test completed!")
    else:
        print("⚠️  Resolution generation test had issues")
    
    # Test error handling
    test_error_handling(client)
    print("✅ Error handling test completed!")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
