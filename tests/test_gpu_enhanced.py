#!/usr/bin/env python3
"""
Test GPU utilization after Ollama configuration changes
"""

import sys
import time
import subprocess
import threading
import requests
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def monitor_gpu_realtime(duration=20):
    """Monitor GPU in real-time during generation"""
    print("🔥 Real-time GPU Monitoring")
    print("=" * 50)
    
    max_util = 0
    max_power = 0
    samples = []
    
    start_time = time.time()
    while time.time() - start_time < duration:
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                if len(parts) >= 4:
                    gpu_util = int(parts[0])
                    mem_used = int(parts[1])
                    power = float(parts[2])
                    temp = int(parts[3])
                    
                    max_util = max(max_util, gpu_util)
                    max_power = max(max_power, power)
                    
                    elapsed = time.time() - start_time
                    print(f"[{elapsed:4.1f}s] GPU: {gpu_util:2d}% | Mem: {mem_used:5d}MB | Power: {power:4.1f}W | Temp: {temp}°C | Max GPU: {max_util:2d}%")
                    
                    samples.append({
                        'time': elapsed,
                        'gpu_util': gpu_util,
                        'memory_mb': mem_used,
                        'power_w': power,
                        'temp_c': temp
                    })
                    
        except Exception as e:
            print(f"Monitoring error: {e}")
        
        time.sleep(0.5)
    
    return max_util, samples


def test_direct_ollama_api():
    """Test direct Ollama API call with longer generation"""
    print("\n🤖 Testing Direct Ollama API with Long Generation")
    print("=" * 60)
    
    # Start monitoring in background
    monitoring = True
    max_util = [0]
    
    def monitor_background():
        while monitoring:
            try:
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=utilization.gpu',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=1)
                if result.returncode == 0:
                    util = int(result.stdout.strip())
                    max_util[0] = max(max_util[0], util)
            except:
                pass
            time.sleep(0.2)
    
    monitor_thread = threading.Thread(target=monitor_background)
    monitor_thread.start()
    
    try:
        # Direct API call with long generation
        payload = {
            "model": "llama3.1:8b",
            "prompt": """Write a detailed technical report about aircraft conflict detection and resolution algorithms. Include mathematical formulas, implementation details, and real-world examples. The report should be comprehensive and detailed, covering at least 10 different aspects of the topic.""",
            "options": {
                "temperature": 0.7,
                "num_predict": 1000,  # Much longer generation
                "top_p": 0.9,
                "top_k": 40
            },
            "stream": False
        }
        
        print("📡 Sending request to Ollama API...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120  # Longer timeout for long generation
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            
            print(f"✅ Generation completed!")
            print(f"⏱️  Time: {generation_time:.1f} seconds")
            print(f"📝 Length: {len(generated_text)} characters")
            print(f"🔥 Max GPU utilization: {max_util[0]}%")
            
            # Show preview of generated text
            if len(generated_text) > 300:
                print(f"📄 Preview: {generated_text[:300]}...")
            else:
                print(f"📄 Full text: {generated_text}")
                
        else:
            print(f"❌ API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    finally:
        monitoring = False
        monitor_thread.join(timeout=2)
    
    return max_util[0]


def check_ollama_environment():
    """Check Ollama environment and configuration"""
    print("\n🔧 Ollama Environment Check")
    print("=" * 40)
    
    # Check environment variables
    import os
    env_vars = [
        'OLLAMA_GPU_LAYERS',
        'OLLAMA_NUM_PARALLEL', 
        'OLLAMA_CONTEXT_LENGTH',
        'CUDA_VISIBLE_DEVICES'
    ]
    
    print("🌐 Environment Variables:")
    for var in env_vars:
        value = os.environ.get(var, '(not set)')
        print(f"   {var}: {value}")
    
    # Check Ollama status
    try:
        result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\n📊 Ollama Status:")
            for line in result.stdout.strip().split('\n'):
                print(f"   {line}")
    except Exception as e:
        print(f"❌ Failed to get Ollama status: {e}")


def main():
    """Run comprehensive GPU test"""
    print("🚀 Enhanced GPU Utilization Test")
    print("=" * 70)
    
    # Check environment
    check_ollama_environment()
    
    # Test with longer generation
    max_util = test_direct_ollama_api()
    
    print(f"\n🏁 Final Results")
    print("=" * 30)
    print(f"📊 Maximum GPU utilization: {max_util}%")
    
    # Get final GPU status
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 3:
                gpu_util = parts[0]
                mem_used = int(parts[1])
                mem_total = int(parts[2])
                mem_percent = (mem_used / mem_total) * 100
                
                print(f"🎯 Current GPU state:")
                print(f"   Utilization: {gpu_util}%")
                print(f"   Memory: {mem_used}MB / {mem_total}MB ({mem_percent:.1f}%)")
    except Exception as e:
        print(f"Error getting GPU status: {e}")
    
    # Analysis
    print(f"\n📈 Analysis:")
    if max_util > 50:
        print("✅ HIGH GPU utilization - GPU is being effectively used")
        status = "EXCELLENT"
    elif max_util > 20:
        print("⚠️  MODERATE GPU utilization - GPU is partially used")
        status = "GOOD"
    elif max_util > 5:
        print("⚠️  LOW GPU utilization - GPU usage is minimal")
        status = "FAIR"
    else:
        print("❌ MINIMAL GPU utilization - GPU may not be used effectively")
        status = "POOR"
    
    print(f"🎯 Overall GPU performance: {status}")
    
    return max_util > 10


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
