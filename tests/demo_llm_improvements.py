#!/usr/bin/env python3
"""
Demonstration of enhanced LLM capabilities with contract-first design
"""

import sys
import json
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cdr.ai.llm_client import (
    LLMClient, LLMConfig, LLMProvider, ConflictContext, 
    RESOLUTION_SCHEMA_V1
)

def demo_contract_first_design():
    """Demonstrate contract-first JSON schema validation"""
    print("🎯 Demo: Contract-First Design")
    print("=" * 50)
    
    # Show the strict schema
    print("📋 Resolution Schema v1:")
    print(json.dumps(RESOLUTION_SCHEMA_V1, indent=2))
    print()
    
    # Example valid resolution
    valid_resolution = {
        "schema_version": "cdr.v1",
        "conflict_id": "demo_001",
        "aircraft1": "UAL123",
        "aircraft2": "DAL456",
        "resolution_type": "heading_change",
        "parameters": {"new_heading_deg": 270},
        "reasoning": "Turn left 20 degrees to maintain 5+ NM separation",
        "confidence": 0.85
    }
    
    print("✅ Valid Resolution Example:")
    print(json.dumps(valid_resolution, indent=2))
    print()


def demo_input_shaping():
    """Demonstrate input shaping and quantization"""
    print("🔧 Demo: Input Shaping & Quantization")
    print("=" * 50)
    
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama3.1:8b",
        max_intruders=3
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        client = LLMClient(config, log_dir=Path(temp_dir))
        
        # Create context with many intruders and precise coordinates
        many_intruders = [
            {"callsign": f"UAL{i:03d}", "latitude": 40.123456789 + i*0.1, 
             "longitude": -74.987654321 + i*0.1, "altitude": 35000, 
             "speed": 450.7 + i, "heading": 89.9 + i}
            for i in range(8)
        ]
        
        original_context = ConflictContext(
            ownship_callsign="DAL123",
            ownship_state={"latitude": 40.123456789, "longitude": -74.987654321, 
                          "altitude": 35000, "speed": 450.7, "heading": 89.9},
            intruders=many_intruders,
            scenario_time=0.0,
            lookahead_minutes=10.0,
            constraints={}
        )
        
        print(f"📥 Original Input:")
        print(f"   Intruders: {len(original_context.intruders)}")
        print(f"   Ownship lat: {original_context.ownship_state['latitude']}")
        print(f"   Ownship speed: {original_context.ownship_state['speed']}")
        print()
        
        # Apply input shaping
        shaped_context = client._shape_input(original_context)
        
        print(f"🎯 After Input Shaping:")
        print(f"   Intruders: {len(shaped_context.intruders)} (limited to top-{config.max_intruders})")
        print(f"   Ownship lat: {shaped_context.ownship_state['latitude']} (quantized)")
        print(f"   Ownship speed: {shaped_context.ownship_state['speed']} (quantized)")
        print()


def demo_sanitization():
    """Demonstrate parameter sanitization"""
    print("🛡️ Demo: Parameter Sanitization")
    print("=" * 50)
    
    config = LLMConfig(provider=LLMProvider.OLLAMA, model_name="llama3.1:8b")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        client = LLMClient(config, log_dir=Path(temp_dir))
        
        # Test cases with out-of-bounds parameters
        test_cases = [
            {
                "name": "Heading out of bounds",
                "input": {"resolution_type": "heading_change", "parameters": {"new_heading_deg": 370}},
                "field": "new_heading_deg"
            },
            {
                "name": "Altitude below minimum",
                "input": {"resolution_type": "altitude_change", "parameters": {"target_altitude_ft": 5000}},
                "field": "target_altitude_ft"
            },
            {
                "name": "Speed above maximum", 
                "input": {"resolution_type": "speed_change", "parameters": {"target_speed_kt": 600}},
                "field": "target_speed_kt"
            },
            {
                "name": "Altitude rounding to FL",
                "input": {"resolution_type": "altitude_change", "parameters": {"target_altitude_ft": 34567}},
                "field": "target_altitude_ft"
            }
        ]
        
        for case in test_cases:
            print(f"🔧 {case['name']}:")
            original_value = case['input']['parameters'][case['field']]
            sanitized = client._sanitize_resolution(case['input'])
            sanitized_value = sanitized['parameters'][case['field']]
            
            print(f"   Input:  {original_value}")
            print(f"   Output: {sanitized_value}")
            print()


def demo_trafscript_conversion():
    """Demonstrate TrafScript command generation"""
    print("✈️ Demo: TrafScript Command Generation")
    print("=" * 50)
    
    resolutions = [
        {
            "callsign": "UAL123",
            "resolution": {
                "resolution_type": "heading_change",
                "parameters": {"new_heading_deg": 270}
            }
        },
        {
            "callsign": "DAL456", 
            "resolution": {
                "resolution_type": "altitude_change",
                "parameters": {"target_altitude_ft": 35000}
            }
        },
        {
            "callsign": "AAL789",
            "resolution": {
                "resolution_type": "speed_change",
                "parameters": {"target_speed_kt": 320}
            }
        },
        {
            "callsign": "SWA321",
            "resolution": {
                "resolution_type": "no_action",
                "parameters": {}
            }
        }
    ]
    
    for item in resolutions:
        commands = LLMClient.to_trafscript(item["callsign"], item["resolution"])
        res_type = item["resolution"]["resolution_type"]
        
        print(f"🎯 {res_type.replace('_', ' ').title()}:")
        print(f"   Aircraft: {item['callsign']}")
        if commands:
            print(f"   Command:  {commands[0]}")
        else:
            print(f"   Command:  (no action required)")
        print()


def demo_telemetry():
    """Demonstrate telemetry and audit capabilities"""
    print("📊 Demo: Telemetry & Audit Logging")
    print("=" * 50)
    
    config = LLMConfig(provider=LLMProvider.OLLAMA, model_name="llama3.1:8b")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir)
        client = LLMClient(config, log_dir=log_dir)
        
        print(f"📁 Audit Directories Created:")
        print(f"   Prompts:   {log_dir / 'prompts'}")
        print(f"   Responses: {log_dir / 'responses'}")
        print()
        
        # Simulate some telemetry updates
        client._update_telemetry(latency=1.2, schema_valid=True, verifier_valid=True, agreement=True)
        client._update_telemetry(latency=0.8, schema_valid=False, verifier_valid=True, agreement=True)  
        client._update_telemetry(latency=1.5, schema_valid=True, verifier_valid=False, agreement=False)
        
        telemetry = client.get_telemetry()
        
        print("📈 Telemetry Metrics:")
        print(f"   Total calls:         {telemetry['total_calls']}")
        print(f"   Schema violations:   {telemetry['schema_violations']}")
        print(f"   Verifier failures:   {telemetry['verifier_failures']}")
        print(f"   Agreement mismatches: {telemetry['agreement_mismatches']}")
        print(f"   Average latency:     {telemetry['average_latency']:.2f}s")
        print()
        
        # Demonstrate artifact logging
        client._log_artifacts(
            "demo_conflict_001",
            "Sample prompt for demo purposes",
            '{"resolution_type": "heading_change", "parameters": {"new_heading_deg": 270}}',
            {"resolution_type": "heading_change", "parameters": {"new_heading_deg": 270}}
        )
        
        print("📝 Audit Files Created:")
        for file_path in (log_dir / "prompts").glob("*.json"):
            print(f"   {file_path.name}")
        for file_path in (log_dir / "responses").glob("*"):
            print(f"   {file_path.name}")
        print()


def demo_config_enhancements():
    """Demonstrate enhanced configuration options"""
    print("⚙️ Demo: Enhanced Configuration")
    print("=" * 50)
    
    from src.cdr.utils.config import Config
    
    config = Config()
    llm_config = config.llm
    
    print("🔧 New LLM Configuration Parameters:")
    enhanced_params = [
        ('seed', 'Reproducibility seed'),
        ('num_predict', 'Token limit for faster responses'),
        ('enable_verifier', 'Two-pass verification'),
        ('enable_agree_on_two', 'Agreement-of-two sampling'),
        ('prompt_char_limit', 'Prompt size guard'),
        ('max_intruders', 'Input shaping limit')
    ]
    
    for param, description in enhanced_params:
        value = llm_config.get(param, 'Not set')
        print(f"   {param:20s}: {value:8} - {description}")
    print()


def main():
    """Run all demonstrations"""
    print("🚀 Enhanced LLM Capabilities Demonstration")
    print("=" * 60)
    print()
    
    demos = [
        demo_contract_first_design,
        demo_input_shaping,
        demo_sanitization,
        demo_trafscript_conversion,
        demo_telemetry,
        demo_config_enhancements
    ]
    
    for demo in demos:
        try:
            demo()
            print()
        except Exception as e:
            print(f"❌ Demo {demo.__name__} failed: {e}")
            print()
    
    print("✨ Demonstration complete!")
    print()
    print("🎯 Key Benefits:")
    print("  • Contract-first design with strict schema validation")
    print("  • Input shaping for performance and consistency")  
    print("  • Parameter sanitization and bounds enforcement")
    print("  • Two-pass verification for safety")
    print("  • Comprehensive telemetry and audit logging")
    print("  • Robust error handling and retry policies")
    print("  • Direct TrafScript command generation")


if __name__ == "__main__":
    main()
