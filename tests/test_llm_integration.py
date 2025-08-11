#!/usr/bin/env python3
"""
Integration test for enhanced LLM client with real Ollama model
"""

import sys
import json
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cdr.ai.llm_client import (
    LLMClient, LLMConfig, LLMProvider, ConflictContext
)

def test_end_to_end_resolution():
    """Test complete end-to-end resolution generation"""
    print("🧪 End-to-End LLM Resolution Test")
    print("=" * 50)
    
    # Use real Ollama model
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama3.1:8b",
        temperature=0.3,
        enable_verifier=False,  # Disable for faster testing
        enable_agree_on_two=False,
        max_intruders=2
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        client = LLMClient(config, log_dir=Path(temp_dir))
        
        # Create realistic conflict scenario
        context = ConflictContext(
            ownship_callsign="UAL123",
            ownship_state={
                "latitude": 40.7128,
                "longitude": -74.0060,
                "altitude": 35000,
                "speed": 450,
                "heading": 90
            },
            intruders=[
                {
                    "callsign": "DAL456",
                    "latitude": 40.7200,
                    "longitude": -74.0000,
                    "altitude": 35000,
                    "speed": 460,
                    "heading": 270
                }
            ],
            scenario_time=0.0,
            lookahead_minutes=10.0,
            constraints={}
        )
        
        conflict_info = {
            "conflict_id": "test_integration_001",
            "conflicts": [
                {
                    "intruder_callsign": "DAL456",
                    "conflict_type": "head_on",
                    "time_to_conflict_minutes": 5.2
                }
            ]
        }
        
        print("📋 Test Scenario:")
        print(f"   Ownship: {context.ownship_callsign} at FL350, hdg 090°")
        print(f"   Intruder: DAL456 at FL350, hdg 270°")
        print(f"   Conflict: Head-on, 5.2 min to impact")
        print()
        
        try:
            # Generate resolution
            print("🤖 Generating LLM resolution...")
            response = client.generate_resolution(context, conflict_info)
            
            if response.success:
                print("✅ Resolution generated successfully!")
                print(f"   Type: {response.resolution_type}")
                print(f"   Parameters: {response.parameters}")
                print(f"   Reasoning: {response.reasoning}")
                print(f"   Confidence: {response.confidence:.2f}")
                print()
                
                # Test TrafScript conversion
                if response.resolution_type != "no_action":
                    commands = LLMClient.to_trafscript(context.ownship_callsign, {
                        "resolution_type": response.resolution_type,
                        "parameters": response.parameters
                    })
                    print(f"🛫 TrafScript Command: {commands[0] if commands else 'None'}")
                    print()
                
                # Check telemetry
                telemetry = client.get_telemetry()
                print("📊 Telemetry:")
                print(f"   Total calls: {telemetry['total_calls']}")
                print(f"   Schema violations: {telemetry['schema_violations']}")
                print(f"   Average latency: {telemetry['average_latency']:.2f}s")
                print()
                
                # Verify audit files were created
                audit_files = list((Path(temp_dir) / "prompts").glob("*.json"))
                audit_files.extend(list((Path(temp_dir) / "responses").glob("*")))
                
                print(f"📁 Audit files created: {len(audit_files)}")
                for file in audit_files:
                    print(f"   {file.name}")
                print()
                
                return True
                
            else:
                print("❌ Resolution generation failed")
                return False
                
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return False


def test_schema_enforcement():
    """Test that schema enforcement actually works with real LLM"""
    print("🧪 Schema Enforcement Test")
    print("=" * 50)
    
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama3.1:8b",
        temperature=0.3,
        enable_verifier=False,
        enable_agree_on_two=False
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        client = LLMClient(config, log_dir=Path(temp_dir))
        
        # Test that the LLM actually produces schema-compliant output
        try:
            context = ConflictContext(
                ownship_callsign="TEST123",
                ownship_state={"latitude": 40.0, "longitude": -74.0, "altitude": 35000, "speed": 450, "heading": 90},
                intruders=[{"callsign": "TEST456", "latitude": 40.1, "longitude": -74.1, "altitude": 35000, "speed": 460, "heading": 270}],
                scenario_time=0.0,
                lookahead_minutes=10.0,
                constraints={}
            )
            
            conflict_info = {
                "conflict_id": "schema_test_001",
                "conflicts": [{"intruder_callsign": "TEST456", "conflict_type": "head_on"}]
            }
            
            resolution = client._generate_single_resolution(context, conflict_info)
            schema_valid, violations = client._validate_schema(resolution)
            
            print(f"📋 Generated Resolution:")
            print(json.dumps(resolution, indent=2))
            print()
            
            if schema_valid:
                print("✅ Schema validation passed!")
            else:
                print(f"❌ Schema validation failed: {violations}")
            
            return schema_valid
            
        except Exception as e:
            print(f"❌ Schema enforcement test failed: {e}")
            return False


def main():
    """Run integration tests"""
    print("🚀 LLM Integration Tests")
    print("=" * 60)
    print()
    
    # Check if Ollama is available
    try:
        from src.cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider
        test_config = LLMConfig(provider=LLMProvider.OLLAMA, model_name="llama3.1:8b")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_client = LLMClient(test_config, log_dir=Path(temp_dir))
            connection_test = test_client.test_connection()
            
            if not connection_test['server_connected']:
                print("❌ Ollama server not available - skipping integration tests")
                print("   Start Ollama server and ensure llama3.1:8b model is available")
                return False
                
            if not connection_test['model_available']:
                print("❌ llama3.1:8b model not available - skipping integration tests")
                print("   Run: ollama pull llama3.1:8b")
                return False
                
    except Exception as e:
        print(f"❌ Cannot initialize LLM client: {e}")
        return False
    
    # Run tests
    tests = [
        test_end_to_end_resolution,
        test_schema_enforcement
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print(f"📊 Integration Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All integration tests passed!")
        print()
        print("✨ Enhanced LLM client is ready for production use!")
        return True
    else:
        print("💥 Some integration tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
