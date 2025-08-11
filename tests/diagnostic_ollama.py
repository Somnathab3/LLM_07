#!/usr/bin/env python3
"""
Diagnostic script for Ollama connection issues
"""

import requests
import json
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_ollama_server():
    """Test basic Ollama server connectivity"""
    print("🔍 Testing Ollama Server Connection")
    print("-" * 50)
    
    base_url = "http://localhost:11434"
    
    try:
        # Test if server is running
        print("1. Testing server availability...")
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            
            # List available models
            data = response.json()
            models = data.get('models', [])
            print(f"✅ Found {len(models)} models:")
            for model in models:
                name = model.get('name', 'Unknown')
                size = model.get('size', 0) / (1024**3)  # Convert to GB
                print(f"   - {name} ({size:.1f} GB)")
            
            return models
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return []
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama server")
        print("   Make sure Ollama is installed and running:")
        print("   1. Install Ollama from https://ollama.ai")
        print("   2. Start Ollama service")
        print("   3. Run: ollama serve")
        return []
    except requests.exceptions.Timeout:
        print("❌ Connection timed out")
        return []
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return []

def test_model_generation(model_name="llama3.1:8b"):
    """Test simple text generation"""
    print(f"\n🤖 Testing Model Generation: {model_name}")
    print("-" * 50)
    
    base_url = "http://localhost:11434"
    
    # Simple test prompt
    test_prompt = "Say 'Hello, I am working correctly!' and nothing else."
    
    try:
        print("1. Sending simple test prompt...")
        start_time = time.time()
        
        payload = {
            "model": model_name,
            "prompt": test_prompt,
            "options": {
                "temperature": 0.1,
                "num_predict": 20  # Limit tokens for faster response
            },
            "stream": False
        }
        
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=60  # Longer timeout for generation
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '').strip()
            print(f"✅ Model responded in {duration:.1f} seconds")
            print(f"✅ Response: '{generated_text}'")
            return True
        else:
            print(f"❌ Generation failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ Generation timed out after 60 seconds")
        print("   This might indicate:")
        print("   - Model is not loaded in memory")
        print("   - System resources are insufficient") 
        print("   - Model is too large for your system")
        return False
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False

def test_json_generation(model_name="llama3.1:8b"):
    """Test JSON response generation"""
    print(f"\n📋 Testing JSON Generation: {model_name}")
    print("-" * 50)
    
    base_url = "http://localhost:11434"
    
    # JSON test prompt
    json_prompt = """Respond with exactly this JSON and nothing else:
{
  "test": true,
  "message": "JSON generation working",
  "number": 42
}"""
    
    try:
        print("1. Sending JSON test prompt...")
        start_time = time.time()
        
        payload = {
            "model": model_name,
            "prompt": json_prompt,
            "options": {
                "temperature": 0.0,  # Most deterministic
                "num_predict": 100
            },
            "stream": False
        }
        
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=60
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '').strip()
            print(f"✅ Model responded in {duration:.1f} seconds")
            print(f"Raw response: {generated_text}")
            
            # Try to parse JSON
            try:
                # Extract JSON from response
                start_idx = generated_text.find('{')
                end_idx = generated_text.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = generated_text[start_idx:end_idx]
                    parsed_json = json.loads(json_str)
                    print(f"✅ JSON parsed successfully: {parsed_json}")
                    return True
                else:
                    print("❌ No JSON found in response")
                    return False
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                return False
        else:
            print(f"❌ Generation failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ JSON generation error: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("🩺 OLLAMA DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Test 1: Server connectivity
    models = test_ollama_server()
    if not models:
        print("\n❌ Cannot proceed - Ollama server not accessible")
        return
    
    # Find target model
    target_model = "llama3.1:8b"
    model_available = any(target_model in model.get('name', '') for model in models)
    
    if not model_available:
        print(f"\n⚠️  Target model '{target_model}' not found")
        print("Available models:")
        for model in models:
            print(f"   - {model.get('name', 'Unknown')}")
        
        # Try with first available model
        if models:
            target_model = models[0].get('name', target_model)
            print(f"Using available model: {target_model}")
        else:
            print("\n❌ No models available. Please pull a model:")
            print("   ollama pull llama3.1:8b")
            return
    
    # Test 2: Simple generation
    if not test_model_generation(target_model):
        print("\n❌ Basic generation failed - check system resources")
        return
    
    # Test 3: JSON generation
    if not test_json_generation(target_model):
        print("\n⚠️  JSON generation issues detected")
        print("This might affect conflict detection accuracy")
    
    print("\n" + "=" * 60)
    print("📊 DIAGNOSIS SUMMARY")
    print("=" * 60)
    print("✅ Server connectivity: OK")
    print("✅ Model availability: OK") 
    print("✅ Basic generation: OK")
    print("\n🔧 RECOMMENDATIONS:")
    print("1. If responses are slow, consider using a smaller model:")
    print("   - ollama pull mistral:7b")
    print("   - ollama pull llama3:7b")
    print("2. Increase timeout in LLM client configuration")
    print("3. Ensure sufficient system memory (8GB+ recommended)")
    
if __name__ == "__main__":
    main()
