#!/usr/bin/env python3
"""Test script for LLM enhancement fixes"""

import json
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider, ConflictContext

def test_enhanced_llm_resolution():
    """Test the enhanced LLM resolution with sanitization and verification"""
    print("🧪 Testing Enhanced LLM Resolution Pipeline")
    print("=" * 50)
    
    # Initialize enhanced LLM client
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama3.1:8b",
        base_url="http://localhost:11434",
        enable_verifier=True,
        enable_agree_on_two=True,
        enable_reprompt_on_failure=True,
        temperature=0.2,
        seed=1337
    )
    
    try:
        llm_client = LLMClient(config)
        print("✅ LLM client initialized with enhanced settings")
    except Exception as e:
        print(f"❌ Failed to initialize LLM client: {e}")
        return
    
    # Test conflict context
    context = ConflictContext(
        ownship_callsign="UAL123",
        ownship_state={
            "callsign": "UAL123",
            "latitude": 41.978,
            "longitude": -87.904,
            "altitude": 35000,  # FL350
            "heading": 90,      # East
            "speed": 450        # knots
        },
        intruders=[{
            "callsign": "AAL456",
            "latitude": 42.0,
            "longitude": -87.8,
            "altitude": 35000,  # Same level - conflict!
            "heading": 270,     # West (head-on)
            "speed": 420
        }],
        scenario_time=300.0,  # 5 minutes
        lookahead_minutes=10.0,
        constraints={"min_separation_nm": 5.0, "min_separation_ft": 1000}
    )
    
    # Test conflict info
    conflict_info = {
        "conflict_id": "TEST_CONFLICT_001",
        "conflicts": [{
            "intruder_callsign": "AAL456",
            "conflict_type": "head_on",
            "time_to_conflict": 8.5
        }]
    }
    
    print("\n📍 Test Scenario:")
    print(f"   Ownship: {context.ownship_callsign} at FL350, heading 90°, 450kt")
    print(f"   Intruder: AAL456 at FL350, heading 270°, 420kt")
    print(f"   Situation: Head-on conflict, ~8.5 min to impact")
    
    # Test resolution generation
    print("\n🤖 Generating enhanced resolution...")
    start_time = time.perf_counter()
    
    try:
        resolution_response = llm_client.generate_resolution(context, conflict_info)
        elapsed = time.perf_counter() - start_time
        
        print(f"⏱️  Resolution generated in {elapsed:.2f}s")
        
        if resolution_response.success:
            print("✅ Resolution Generation SUCCESSFUL")
            print(f"   Type: {resolution_response.resolution_type}")
            print(f"   Parameters: {resolution_response.parameters}")
            print(f"   Reasoning: {resolution_response.reasoning}")
            print(f"   Confidence: {resolution_response.confidence:.2f}")
            
            # Check if it's a proper resolution
            if resolution_response.resolution_type == "heading_change":
                new_heading = resolution_response.parameters.get("new_heading_deg")
                current_heading = context.ownship_state["heading"]
                heading_change = abs(new_heading - current_heading)
                
                print(f"   Heading Change: {current_heading}° → {new_heading}° (±{heading_change:.1f}°)")
                
                if heading_change >= 15:
                    print("✅ Heading change meets minimum 15° requirement")
                else:
                    print(f"❌ Heading change {heading_change:.1f}° below 15° minimum")
                    
        else:
            print("❌ Resolution Generation FAILED")
            print(f"   Fallback: {resolution_response.resolution_type}")
            
    except Exception as e:
        print(f"❌ Resolution generation error: {e}")
        return
    
    # Display telemetry
    print(f"\n📊 LLM Telemetry:")
    telemetry = llm_client.get_telemetry()
    print(f"   Total calls: {telemetry.get('total_calls', 0)}")
    print(f"   Schema violations: {telemetry.get('schema_violations', 0)}")
    print(f"   Verifier failures: {telemetry.get('verifier_failures', 0)}")
    print(f"   Agreement mismatches: {telemetry.get('agreement_mismatches', 0)}")
    print(f"   Average latency: {telemetry.get('average_latency', 0):.2f}s")
    
    print("\n🎯 Test completed!")

if __name__ == "__main__":
    test_enhanced_llm_resolution()
