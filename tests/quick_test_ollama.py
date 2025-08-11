#!/usr/bin/env python3
"""
Quick test for Ollama LLM Client with simplified prompts
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider, ConflictContext


def quick_test():
    """Run a quick test with simplified settings"""
    print("⚡ Quick Ollama Test with Simplified Prompts")
    print("=" * 50)
    
    # Optimized config for testing
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama3.1:8b",
        base_url="http://localhost:11434",
        temperature=0.0,  # Most deterministic
        max_tokens=100,   # Very limited for fast response
        timeout=30.0
    )
    
    try:
        # Initialize client
        print("🔧 Initializing client...")
        client = LLMClient(config)
        
        # Test connection
        print("🔌 Testing connection...")
        test_result = client.test_connection()
        
        if not test_result['generation_test']:
            print("❌ Connection failed")
            return
        
        print("✅ Connection OK!")
        
        # Simple test scenario
        print("\\n📊 Testing simple conflict scenario...")
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
                    'latitude': 40.7150,
                    'longitude': -73.9950,
                    'altitude': 35000,
                    'heading': 270,
                    'speed': 440,
                    'position': '40.7150°, -73.9950°'
                }
            ],
            scenario_time=0.0,
            lookahead_minutes=5.0,
            constraints={}
        )
        
        # Test with simple prompts
        print("🔍 Testing conflict detection (simple prompts)...")
        import time
        start_time = time.time()
        
        # Force use of simple prompts
        conflict_result = client.detect_conflicts(context, use_simple_prompt=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"⏱️  Detection took {duration:.1f} seconds")
        print(f"📊 Result: {conflict_result}")
        
        # Test resolution if conflicts detected
        if conflict_result.get('conflicts_detected'):
            print("\\n⚡ Testing resolution generation...")
            start_time = time.time()
            
            resolution = client.generate_resolution(context, conflict_result)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"⏱️  Resolution took {duration:.1f} seconds")
            print(f"📊 Resolution: {resolution.resolution_type}")
            print(f"📊 Parameters: {resolution.parameters}")
        
        print("\\n✅ Quick test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    quick_test()
