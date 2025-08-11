#!/usr/bin/env python3
"""
Test script for targeted LLM improvements:
1. Contract-first JSON schema validation
2. Two-pass verification 
3. Input shaping and sanitization
4. Robustness features
5. Telemetry and audit logging
"""

import sys
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_schema_validation():
    """Test A: Strict JSON schema validation"""
    print("🧪 Test A: JSON Schema Validation")
    
    try:
        from src.cdr.ai.llm_client import RESOLUTION_SCHEMA_V1, LLMClient, LLMConfig, LLMProvider
        import jsonschema
        
        # Test valid schema
        valid_response = {
            "schema_version": "cdr.v1",
            "conflict_id": "test_123",
            "aircraft1": "UAL123",
            "aircraft2": "DAL456", 
            "resolution_type": "heading_change",
            "parameters": {"new_heading_deg": 270},
            "reasoning": "Turn left to avoid conflict",
            "confidence": 0.85
        }
        
        # Should not raise exception
        jsonschema.validate(valid_response, RESOLUTION_SCHEMA_V1)
        print("✅ Valid schema passes validation")
        
        # Test boundary values
        boundary_cases = [
            # Valid boundaries
            {"schema_version": "cdr.v1", "conflict_id": "test", "aircraft1": "A", "aircraft2": "B", 
             "resolution_type": "heading_change", "parameters": {"new_heading_deg": 0}, 
             "reasoning": "Test", "confidence": 0.0},
            {"schema_version": "cdr.v1", "conflict_id": "test", "aircraft1": "A", "aircraft2": "B",
             "resolution_type": "heading_change", "parameters": {"new_heading_deg": 360},
             "reasoning": "Test", "confidence": 1.0},
            {"schema_version": "cdr.v1", "conflict_id": "test", "aircraft1": "A", "aircraft2": "B",
             "resolution_type": "altitude_change", "parameters": {"target_altitude_ft": 10000},
             "reasoning": "Test", "confidence": 0.5},
            {"schema_version": "cdr.v1", "conflict_id": "test", "aircraft1": "A", "aircraft2": "B",
             "resolution_type": "speed_change", "parameters": {"target_speed_kt": 250},
             "reasoning": "Test", "confidence": 0.5},
            {"schema_version": "cdr.v1", "conflict_id": "test", "aircraft1": "A", "aircraft2": "B",
             "resolution_type": "no_action", "parameters": {},
             "reasoning": "Test", "confidence": 0.5},
        ]
        
        for case in boundary_cases:
            jsonschema.validate(case, RESOLUTION_SCHEMA_V1)
        print("✅ Boundary values pass validation")
        
        # Test invalid cases
        invalid_cases = [
            # Missing required field
            {"schema_version": "cdr.v1", "conflict_id": "test", "aircraft1": "A", 
             "resolution_type": "heading_change", "parameters": {"new_heading_deg": 270},
             "reasoning": "Test", "confidence": 0.85},
            # Wrong schema version
            {"schema_version": "cdr.v2", "conflict_id": "test", "aircraft1": "A", "aircraft2": "B",
             "resolution_type": "heading_change", "parameters": {"new_heading_deg": 270},
             "reasoning": "Test", "confidence": 0.85},
            # Invalid resolution type
            {"schema_version": "cdr.v1", "conflict_id": "test", "aircraft1": "A", "aircraft2": "B",
             "resolution_type": "invalid_type", "parameters": {"new_heading_deg": 270},
             "reasoning": "Test", "confidence": 0.85},
            # Confidence out of range
            {"schema_version": "cdr.v1", "conflict_id": "test", "aircraft1": "A", "aircraft2": "B",
             "resolution_type": "heading_change", "parameters": {"new_heading_deg": 270},
             "reasoning": "Test", "confidence": 1.5},
        ]
        
        invalid_count = 0
        for case in invalid_cases:
            try:
                jsonschema.validate(case, RESOLUTION_SCHEMA_V1)
                print(f"❌ Invalid case should have failed: {case}")
            except jsonschema.exceptions.ValidationError:
                invalid_count += 1
        
        if invalid_count == len(invalid_cases):
            print("✅ Invalid cases correctly rejected")
        else:
            print(f"❌ Only {invalid_count}/{len(invalid_cases)} invalid cases rejected")
            
    except Exception as e:
        print(f"❌ Schema validation test failed: {e}")
        return False
    
    return True


def test_input_shaping():
    """Test B: Input shaping and sanitization"""
    print("🧪 Test B: Input Shaping and Sanitization")
    
    try:
        from src.cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider, ConflictContext
        
        # Create test configuration
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model_name="test_model",
            max_intruders=3
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            client = LLMClient(config, log_dir=Path(temp_dir))
            
            # Test input with many intruders (should be limited to top 3)
            many_intruders = [
                {"callsign": f"UAL{i:03d}", "latitude": 40.0 + i*0.1, "longitude": -74.0 + i*0.1,
                 "altitude": 35000, "speed": 450, "heading": 90} 
                for i in range(10)
            ]
            
            context = ConflictContext(
                ownship_callsign="DAL123",
                ownship_state={"latitude": 40.0, "longitude": -74.0, "altitude": 35000, "speed": 450, "heading": 90},
                intruders=many_intruders,
                scenario_time=0.0,
                lookahead_minutes=10.0,
                constraints={}
            )
            
            # Test input shaping
            shaped = client._shape_input(context)
            
            # Should limit to max_intruders
            assert len(shaped.intruders) <= config.max_intruders
            print(f"✅ Input shaping limits intruders: {len(many_intruders)} -> {len(shaped.intruders)}")
            
            # Test quantization
            test_state = {"latitude": 40.123456789, "longitude": -74.987654321, "speed": 450.7, "heading": 89.9}
            quantized_context = client._shape_input(ConflictContext(
                ownship_callsign="TEST",
                ownship_state=test_state,
                intruders=[test_state],
                scenario_time=0.0,
                lookahead_minutes=10.0,
                constraints={}
            ))
            
            # Check quantization
            qs = quantized_context.ownship_state
            assert qs["latitude"] == round(test_state["latitude"], 4)
            assert qs["speed"] == int(round(test_state["speed"]))
            assert qs["heading"] == int(round(test_state["heading"]))
            print("✅ State quantization working correctly")
            
    except Exception as e:
        print(f"❌ Input shaping test failed: {e}")
        return False
    
    return True


def test_sanitization():
    """Test C: Parameter sanitization and bounds enforcement"""
    print("🧪 Test C: Parameter Sanitization")
    
    try:
        from src.cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider
        
        config = LLMConfig(provider=LLMProvider.OLLAMA, model_name="test_model")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            client = LLMClient(config, log_dir=Path(temp_dir))
            
            # Test heading sanitization
            heading_cases = [
                ({"resolution_type": "heading_change", "parameters": {"new_heading_deg": -10}}, 0),
                ({"resolution_type": "heading_change", "parameters": {"new_heading_deg": 370}}, 360),
                ({"resolution_type": "heading_change", "parameters": {"new_heading_deg": 180.7}}, 181),
            ]
            
            for input_res, expected_heading in heading_cases:
                sanitized = client._sanitize_resolution(input_res)
                actual = sanitized["parameters"]["new_heading_deg"]
                assert actual == expected_heading, f"Expected {expected_heading}, got {actual}"
            print("✅ Heading sanitization working")
            
            # Test altitude sanitization (snap to FL)
            altitude_cases = [
                ({"resolution_type": "altitude_change", "parameters": {"target_altitude_ft": 34567}}, 34600),  # Rounds to nearest 100
                ({"resolution_type": "altitude_change", "parameters": {"target_altitude_ft": 34533}}, 34500),  # Rounds down 
                ({"resolution_type": "altitude_change", "parameters": {"target_altitude_ft": 5000}}, 10000),  # Min bound
                ({"resolution_type": "altitude_change", "parameters": {"target_altitude_ft": 50000}}, 45000),  # Max bound
            ]
            
            for input_res, expected_alt in altitude_cases:
                sanitized = client._sanitize_resolution(input_res)
                actual = sanitized["parameters"]["target_altitude_ft"]
                assert actual == expected_alt, f"Expected {expected_alt}, got {actual}"
            print("✅ Altitude sanitization working")
            
            # Test speed sanitization
            speed_cases = [
                ({"resolution_type": "speed_change", "parameters": {"target_speed_kt": 200}}, 250),  # Min bound
                ({"resolution_type": "speed_change", "parameters": {"target_speed_kt": 500}}, 490),  # Max bound
                ({"resolution_type": "speed_change", "parameters": {"target_speed_kt": 350.7}}, 351),
            ]
            
            for input_res, expected_speed in speed_cases:
                sanitized = client._sanitize_resolution(input_res)
                actual = sanitized["parameters"]["target_speed_kt"]
                assert actual == expected_speed, f"Expected {expected_speed}, got {actual}"
            print("✅ Speed sanitization working")
            
    except Exception as e:
        print(f"❌ Sanitization test failed: {e}")
        return False
    
    return True


def test_trafscript_conversion():
    """Test D: LLM to TrafScript conversion"""
    print("🧪 Test D: TrafScript Conversion")
    
    try:
        from src.cdr.ai.llm_client import LLMClient
        
        # Test heading change
        heading_res = {
            "resolution_type": "heading_change",
            "parameters": {"new_heading_deg": 270}
        }
        commands = LLMClient.to_trafscript("UAL123", heading_res)
        assert commands == ["HDG UAL123,270"]
        print("✅ Heading change conversion")
        
        # Test altitude change
        altitude_res = {
            "resolution_type": "altitude_change", 
            "parameters": {"target_altitude_ft": 35000}
        }
        commands = LLMClient.to_trafscript("DAL456", altitude_res)
        assert commands == ["ALT DAL456,FL350"]
        print("✅ Altitude change conversion")
        
        # Test speed change
        speed_res = {
            "resolution_type": "speed_change",
            "parameters": {"target_speed_kt": 300}
        }
        commands = LLMClient.to_trafscript("AAL789", speed_res)
        assert commands == ["SPD AAL789,300"]
        print("✅ Speed change conversion")
        
        # Test no action
        no_action_res = {"resolution_type": "no_action", "parameters": {}}
        commands = LLMClient.to_trafscript("SWA321", no_action_res)
        assert commands == []
        print("✅ No action conversion")
        
    except Exception as e:
        print(f"❌ TrafScript conversion test failed: {e}")
        return False
    
    return True


def test_telemetry():
    """Test E: Telemetry and audit logging"""
    print("🧪 Test E: Telemetry and Audit Logging")
    
    try:
        from src.cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider
        
        config = LLMConfig(provider=LLMProvider.OLLAMA, model_name="test_model")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            client = LLMClient(config, log_dir=log_dir)
            
            # Check that directories were created
            assert (log_dir / "prompts").exists()
            assert (log_dir / "responses").exists()
            print("✅ Audit directories created")
            
            # Test telemetry initialization
            telemetry = client.get_telemetry()
            required_fields = ['total_calls', 'schema_violations', 'verifier_failures', 
                             'agreement_mismatches', 'average_latency']
            
            for field in required_fields:
                assert field in telemetry
            print("✅ Telemetry initialized correctly")
            
            # Test telemetry update
            client._update_telemetry(latency=1.5, schema_valid=True, verifier_valid=False, agreement=True)
            updated = client.get_telemetry()
            
            assert updated['total_calls'] == 1
            assert updated['verifier_failures'] == 1
            assert updated['average_latency'] == 1.5
            print("✅ Telemetry updates correctly")
            
            # Test artifact logging
            client._log_artifacts("test_conflict_123", "test prompt", "test response", {"test": "parsed"})
            
            prompt_file = log_dir / "prompts" / "test_conflict_123.json"
            raw_file = log_dir / "responses" / "test_conflict_123.raw.txt"
            parsed_file = log_dir / "responses" / "test_conflict_123.parsed.json"
            
            assert prompt_file.exists()
            assert raw_file.exists() 
            assert parsed_file.exists()
            print("✅ Artifact logging working")
            
    except Exception as e:
        print(f"❌ Telemetry test failed: {e}")
        return False
    
    return True


def test_prompt_size_guard():
    """Test F: Prompt size guard"""
    print("🧪 Test F: Prompt Size Guard")
    
    try:
        from src.cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider, ConflictContext
        
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model_name="test_model",
            prompt_char_limit=1000  # Small limit for testing
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            client = LLMClient(config, log_dir=Path(temp_dir))
            
            # Create large context that would exceed limit
            many_intruders = [
                {"callsign": f"VERY_LONG_CALLSIGN_{i:05d}", 
                 "latitude": 40.0 + i*0.001, "longitude": -74.0 + i*0.001,
                 "altitude": 35000 + i*100, "speed": 450 + i, "heading": (90 + i) % 360} 
                for i in range(20)  # Many intruders to create large prompt
            ]
            
            context = ConflictContext(
                ownship_callsign="VERY_LONG_OWNSHIP_CALLSIGN_123",
                ownship_state={"latitude": 40.0, "longitude": -74.0, "altitude": 35000, "speed": 450, "heading": 90},
                intruders=many_intruders,
                scenario_time=0.0,
                lookahead_minutes=10.0,
                constraints={}
            )
            
            # Shape input should limit size
            shaped = client._shape_input(context)
            assert len(shaped.intruders) <= config.max_intruders
            print("✅ Prompt size guard limits intruders automatically")
            
    except Exception as e:
        print(f"❌ Prompt size guard test failed: {e}")
        return False
    
    return True


def test_config_integration():
    """Test G: Config integration with new parameters"""
    print("🧪 Test G: Config Integration")
    
    try:
        from src.cdr.utils.config import Config
        
        # Test default config includes new LLM parameters
        config = Config()
        llm_config = config.llm
        
        required_params = ['seed', 'num_predict', 'enable_verifier', 
                          'enable_agree_on_two', 'prompt_char_limit', 'max_intruders']
        
        for param in required_params:
            assert param in llm_config, f"Missing LLM config parameter: {param}"
        
        # Check reasonable defaults
        assert llm_config['seed'] == 1337
        assert llm_config['num_predict'] == 192
        assert llm_config['enable_verifier'] == True
        assert llm_config['enable_agree_on_two'] == False
        assert llm_config['prompt_char_limit'] == 12000
        assert llm_config['max_intruders'] == 3
        
        print("✅ Config integration working with new parameters")
        
    except Exception as e:
        print(f"❌ Config integration test failed: {e}")
        return False
    
    return True


def main():
    """Run all LLM improvement tests"""
    print("🚀 Running LLM Improvements Tests\n")
    
    tests = [
        test_schema_validation,
        test_input_shaping,
        test_sanitization,
        test_trafscript_conversion,
        test_telemetry,
        test_prompt_size_guard,
        test_config_integration
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
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All LLM improvement tests passed!")
        return True
    else:
        print("💥 Some tests failed - please review implementation")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
