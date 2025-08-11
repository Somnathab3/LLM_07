#!/usr/bin/env python3
"""
Optimized test for Ollama LLM Client with better timeout handling
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider, ConflictContext


def test_with_shorter_timeout():
    """Test with optimized settings for faster responses"""
    print("🚀 Optimized Ollama LLM Client Test")
    print("=" * 50)
    
    # Use more conservative settings
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama3.1:8b",  
        base_url="http://localhost:11434",
        temperature=0.1,
        max_tokens=200,  # Reduced for faster responses
        timeout=60.0     # Increased timeout
    )
    
    try:
        # Initialize the client
        print("🔧 Initializing LLM client...")
        client = LLMClient(config)
        
        # Test connection first
        print("🔌 Testing connection...")
        test_result = client.test_connection()
        
        print("Connection test results:")
        for key, value in test_result.items():
            print(f"   {key}: {value}")
        
        if not test_result['generation_test']:
            print("❌ Connection test failed. Cannot proceed.")
            return
        
        print("✅ Connection successful!")
        
        # Create a simpler conflict scenario for testing
        print("\\n📊 Creating simple test scenario...")
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
            lookahead_minutes=5.0,  # Reduced for simpler analysis
            constraints={}
        )
        
        print(f"   Ownship: {context.ownship_callsign} at FL350")
        print(f"   Intruder: UAL456 at FL350 (head-on scenario)")
        
        # Test conflict detection with timeout monitoring
        print("\\n🔍 Testing conflict detection...")
        import time
        start_time = time.time()
        
        try:
            conflict_result = client.detect_conflicts(context)
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✅ Detection completed in {duration:.1f} seconds")
            print(f"   Conflicts detected: {conflict_result.get('conflicts_detected', False)}")
            
            if conflict_result.get('conflicts'):
                for i, conflict in enumerate(conflict_result['conflicts']):
                    print(f"   Conflict {i+1}: {conflict.get('intruder_callsign')}")
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"❌ Detection failed after {duration:.1f} seconds: {e}")
            return
        
        # Test resolution generation if conflicts detected
        if conflict_result.get('conflicts_detected'):
            print("\\n⚡ Testing resolution generation...")
            start_time = time.time()
            
            try:
                resolution = client.generate_resolution(context, conflict_result)
                end_time = time.time()
                duration = end_time - start_time
                
                print(f"✅ Resolution completed in {duration:.1f} seconds")
                if resolution.success:
                    print(f"   Type: {resolution.resolution_type}")
                    print(f"   Parameters: {resolution.parameters}")
                    print(f"   Confidence: {resolution.confidence:.2f}")
                else:
                    print("   ❌ Resolution generation failed")
                    
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                print(f"❌ Resolution failed after {duration:.1f} seconds: {e}")
        
        print("\\n🎯 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\\nTroubleshooting steps:")
        print("1. Run diagnostic: python diagnostic_ollama.py")
        print("2. Check Ollama status: ollama list")
        print("3. Try smaller model: ollama pull mistral:7b")


def test_different_models():
    """Test with different model configurations"""
    models_to_test = [
        ("llama3.1:8b", 60),
        ("mistral:7b", 30),
        ("llama3:7b", 45)
    ]
    
    print("\\n🔄 Testing different models...")
    
    for model_name, timeout in models_to_test:
        print(f"\\n--- Testing {model_name} ---")
        
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model_name=model_name,
            base_url="http://localhost:11434",
            temperature=0.1,
            max_tokens=100,
            timeout=timeout
        )
        
        try:
            client = LLMClient(config)
            test_result = client.test_connection()
            
            if test_result['generation_test']:
                print(f"✅ {model_name}: Working")
            else:
                print(f"❌ {model_name}: Failed")
                
        except Exception as e:
            print(f"❌ {model_name}: Error - {e}")


if __name__ == "__main__":
    # Run the optimized test
    test_with_shorter_timeout()
    
    # Test different models
    test_different_models()
