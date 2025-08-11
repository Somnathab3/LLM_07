#!/usr/bin/env python3
"""Test script for enhanced LLM client"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cdr.ai.llm_client import (
    LLMClient, LLMConfig, LLMProvider, ConflictContext, 
    ResolutionResponse, PromptTemplate
)


def test_llm_setup():
    """Test LLM client setup for different providers"""
    print("🧪 Testing LLM Client Setup")
    print("=" * 40)
    
    # Test Ollama setup
    print("\n1️⃣ Testing Ollama Setup...")
    try:
        ollama_config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model_name="llama3.1:8b",  # or whatever model you have
            base_url="http://localhost:11434",
            temperature=0.1,
            max_tokens=1000
        )
        
        ollama_client = LLMClient(ollama_config)
        print("✅ Ollama client created successfully")
        
        # Test connection
        test_results = ollama_client.test_connection()
        print(f"   Connection: {'✅' if test_results['connection_ok'] else '❌'}")
        print(f"   Model Available: {'✅' if test_results['model_available'] else '❌'}")
        print(f"   JSON Parsing: {'✅' if test_results['json_parsing'] else '❌'}")
        print(f"   Response Time: {test_results['response_time']:.2f}s")
        
        if test_results.get('error'):
            print(f"   Error: {test_results['error']}")
        
        return ollama_client
        
    except Exception as e:
        print(f"❌ Ollama setup failed: {e}")
        print("   Make sure Ollama is running and has a model installed")
        return None


def test_conflict_detection(llm_client):
    """Test LLM conflict detection"""
    if not llm_client:
        print("⏭️ Skipping conflict detection test (no LLM client)")
        return
    
    print("\n🧪 Testing Conflict Detection")
    print("=" * 40)
    
    # Create test scenario
    context = ConflictContext(
        ownship_callsign="AAL123",
        ownship_state={
            "latitude": 40.7128,
            "longitude": -74.0060,
            "altitude": 35000,
            "heading": 90,
            "speed": 450,
            "vertical_speed": 0,
            "aircraft_type": "B737"
        },
        intruders=[
            {
                "callsign": "UAL456",
                "latitude": 40.7200,
                "longitude": -74.0000,
                "altitude": 35000,
                "heading": 270,
                "speed": 480,
                "aircraft_type": "A320",
                "distance_nm": 8.5
            },
            {
                "callsign": "DAL789",
                "latitude": 40.7000,
                "longitude": -74.0200,
                "altitude": 37000,
                "heading": 45,
                "speed": 420,
                "aircraft_type": "B777",
                "distance_nm": 12.2
            }
        ],
        scenario_time=time.time(),
        lookahead_minutes=10,
        constraints={},
        weather_conditions={"conditions": "VMC", "visibility": "10+ miles", "wind": "270/15"},
        emergency_priority=False
    )
    
    try:
        print("🔍 Running conflict detection...")
        start_time = time.time()
        
        detection_result = llm_client.detect_conflicts(context)
        
        detection_time = time.time() - start_time
        print(f"⏱️ Detection completed in {detection_time:.2f}s")
        
        print("\n📊 Detection Results:")
        print(f"   Conflicts Detected: {detection_result.get('conflicts_detected', False)}")
        print(f"   Number of Conflicts: {len(detection_result.get('conflicts', []))}")
        print(f"   Analysis Confidence: {detection_result.get('analysis_confidence', 0):.2f}")
        
        for i, conflict in enumerate(detection_result.get('conflicts', [])):
            print(f"\n   Conflict {i+1}:")
            print(f"     Intruder: {conflict.get('intruder_callsign')}")
            print(f"     Time to Conflict: {conflict.get('time_to_conflict_seconds', 0):.0f}s")
            print(f"     Geometry: {conflict.get('conflict_geometry', 'unknown')}")
            print(f"     Severity: {conflict.get('severity', 'unknown')}")
            print(f"     Confidence: {conflict.get('confidence', 0):.2f}")
        
        if detection_result.get('overall_assessment'):
            print(f"\n   Assessment: {detection_result['overall_assessment']}")
        
        return detection_result
        
    except Exception as e:
        print(f"❌ Conflict detection failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_resolution_generation(llm_client, detection_result):
    """Test LLM resolution generation"""
    if not llm_client or not detection_result:
        print("⏭️ Skipping resolution generation test")
        return
    
    print("\n🧪 Testing Resolution Generation")
    print("=" * 40)
    
    # Use the same context as detection
    context = ConflictContext(
        ownship_callsign="AAL123",
        ownship_state={
            "latitude": 40.7128,
            "longitude": -74.0060,
            "altitude": 35000,
            "heading": 90,
            "speed": 450,
            "vertical_speed": 0,
            "aircraft_type": "B737"
        },
        intruders=[
            {
                "callsign": "UAL456",
                "latitude": 40.7200,
                "longitude": -74.0000,
                "altitude": 35000,
                "heading": 270,
                "speed": 480,
                "aircraft_type": "A320"
            }
        ],
        scenario_time=time.time(),
        lookahead_minutes=10,
        constraints={}
    )
    
    try:
        print("🎯 Generating resolution...")
        start_time = time.time()
        
        resolution = llm_client.generate_resolution(context, detection_result)
        
        resolution_time = time.time() - start_time
        print(f"⏱️ Resolution generated in {resolution_time:.2f}s")
        
        print("\n📋 Resolution Results:")
        print(f"   Success: {resolution.success}")
        print(f"   Type: {resolution.resolution_type}")
        print(f"   Confidence: {resolution.confidence:.2f}")
        print(f"   Safety Score: {resolution.safety_score:.2f}")
        print(f"   Priority: {resolution.execution_priority}")
        
        if resolution.parameters:
            print(f"\n   Parameters:")
            for key, value in resolution.parameters.items():
                print(f"     {key}: {value}")
        
        if resolution.reasoning:
            print(f"\n   Reasoning: {resolution.reasoning}")
        
        return resolution
        
    except Exception as e:
        print(f"❌ Resolution generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_json_parsing(llm_client):
    """Test JSON parsing capabilities"""
    if not llm_client:
        print("⏭️ Skipping JSON parsing test")
        return
    
    print("\n🧪 Testing JSON Parsing")
    print("=" * 30)
    
    # Test various JSON response formats
    test_responses = [
        # Valid JSON
        '{"test": true, "value": 42}',
        
        # JSON with markdown
        '''```json
{
  "conflicts_detected": true,
  "confidence": 0.85
}
```''',
        
        # JSON with extra text
        'Here is the analysis:\n\n{"conflicts_detected": false, "reason": "no conflicts"}\n\nEnd of analysis.',
        
        # Malformed JSON that needs manual extraction
        'conflicts_detected: true, confidence: 0.75, resolution_type: "heading_change"',
        
        # Complex nested JSON
        '''
{
  "conflicts_detected": true,
  "conflicts": [
    {
      "intruder_callsign": "TEST123",
      "time_to_conflict_seconds": 180,
      "severity": "medium"
    }
  ],
  "confidence": 0.92
}
'''
    ]
    
    success_count = 0
    for i, response in enumerate(test_responses):
        print(f"\n   Test {i+1}: ", end="")
        try:
            parsed = llm_client._parse_json_response(response)
            if parsed:
                print("✅ Parsed successfully")
                print(f"      Keys: {list(parsed.keys())}")
                success_count += 1
            else:
                print("❌ Parsing failed")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n   Success Rate: {success_count}/{len(test_responses)} ({success_count/len(test_responses)*100:.1f}%)")


def test_performance_metrics(llm_client):
    """Test performance metrics tracking"""
    if not llm_client:
        print("⏭️ Skipping performance metrics test")
        return
    
    print("\n🧪 Testing Performance Metrics")
    print("=" * 35)
    
    # Get initial metrics
    initial_metrics = llm_client.get_performance_metrics()
    print(f"   Initial metrics: {initial_metrics}")
    
    # Make a few test calls
    test_prompt = '{"test": "performance"}'
    for i in range(3):
        try:
            llm_client._call_llm(f"Test call {i+1}: {test_prompt}")
        except:
            pass
    
    # Get updated metrics
    final_metrics = llm_client.get_performance_metrics()
    print(f"\n   Final metrics:")
    print(f"     Total calls: {final_metrics['total_calls']}")
    print(f"     Success rate: {final_metrics['success_rate']:.2f}")
    print(f"     Avg response time: {final_metrics['avg_response_time']:.2f}s")
    print(f"     JSON parse rate: {final_metrics['json_parse_success_rate']:.2f}")


def main():
    """Main test function"""
    print("🚀 Enhanced LLM Client Test Suite")
    print("=" * 50)
    
    # Test 1: Setup
    llm_client = test_llm_setup()
    
    if llm_client:
        # Test 2: JSON Parsing
        test_json_parsing(llm_client)
        
        # Test 3: Conflict Detection
        detection_result = test_conflict_detection(llm_client)
        
        # Test 4: Resolution Generation
        test_resolution_generation(llm_client, detection_result)
        
        # Test 5: Performance Metrics
        test_performance_metrics(llm_client)
        
        print("\n" + "=" * 50)
        print("🎉 LLM Client tests completed!")
        
        # Show final performance summary
        final_metrics = llm_client.get_performance_metrics()
        print(f"\n📊 Final Performance Summary:")
        for key, value in final_metrics.items():
            print(f"   {key}: {value}")
    
    else:
        print("\n❌ LLM Client not available - tests skipped")
        print("   To run full tests:")
        print("   1. Install and start Ollama: https://ollama.ai/")
        print("   2. Pull a model: ollama pull llama3.1:8b")
        print("   3. Ensure Ollama is running on port 11434")


if __name__ == "__main__":
    main()
