#!/usr/bin/env python3
"""
Debug test to see what Ollama is actually returning
"""

import requests
import json

def test_direct_ollama():
    """Test Ollama directly to see raw responses"""
    print("🔍 Testing Ollama Direct Response")
    print("=" * 50)
    
    base_url = "http://localhost:11434"
    
    # Super simple JSON prompt
    simple_prompt = """Return only this JSON, nothing else:
{"conflicts_detected": true, "message": "test"}"""
    
    print("Prompt:")
    print(simple_prompt)
    print("\\nSending to Ollama...")
    
    try:
        payload = {
            "model": "llama3.1:8b",
            "prompt": simple_prompt,
            "options": {
                "temperature": 0.0,
                "num_predict": 50
            },
            "stream": False
        }
        
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '').strip()
            
            print("Raw response:")
            print(f"'{generated_text}'")
            print(f"\\nLength: {len(generated_text)} characters")
            
            # Try to parse
            try:
                parsed = json.loads(generated_text)
                print("✅ JSON parsing successful!")
                print(f"Parsed: {parsed}")
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                
                # Try to extract
                start_idx = generated_text.find('{')
                end_idx = generated_text.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = generated_text[start_idx:end_idx]
                    print(f"\\nExtracted JSON: '{json_str}'")
                    try:
                        parsed = json.loads(json_str)
                        print("✅ Extraction successful!")
                        print(f"Parsed: {parsed}")
                    except json.JSONDecodeError as e2:
                        print(f"❌ Extraction failed: {e2}")
                
        else:
            print(f"❌ Request failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_conflict_prompt():
    """Test the actual conflict detection prompt"""
    print("\\n🔍 Testing Conflict Detection Prompt")
    print("=" * 50)
    
    base_url = "http://localhost:11434"
    
    # Our simplified conflict prompt
    conflict_prompt = """You must respond with valid JSON only. No explanations, no text before or after.

Aircraft conflict analysis:
- Ownship: AAL123 at FL350, heading 90°
- Intruders: UAL456
- Timeframe: 5 minutes

JSON response required:
{
  "conflicts_detected": true,
  "conflicts": [
    {
      "intruder_callsign": "UAL456",
      "conflict_type": "head_on"
    }
  ]
}"""
    
    print("Conflict Prompt:")
    print(conflict_prompt)
    print("\\nSending to Ollama...")
    
    try:
        payload = {
            "model": "llama3.1:8b",
            "prompt": conflict_prompt,
            "options": {
                "temperature": 0.0,
                "num_predict": 200
            },
            "stream": False
        }
        
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '').strip()
            
            print("\\nRaw response:")
            print(f"'{generated_text}'")
            
            # Try to parse
            try:
                parsed = json.loads(generated_text)
                print("✅ JSON parsing successful!")
                print(f"Parsed: {parsed}")
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                
                # Try to extract
                start_idx = generated_text.find('{')
                end_idx = generated_text.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = generated_text[start_idx:end_idx]
                    print(f"\\nExtracted JSON: '{json_str}'")
                    try:
                        parsed = json.loads(json_str)
                        print("✅ Extraction successful!")
                        print(f"Parsed: {parsed}")
                    except json.JSONDecodeError as e2:
                        print(f"❌ Extraction failed: {e2}")
                
        else:
            print(f"❌ Request failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_direct_ollama()
    test_conflict_prompt()
