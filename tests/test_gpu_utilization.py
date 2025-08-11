#!/usr/bin/env python3
"""
GPU utilization test for Ollama LLM inference
Monitors GPU usage during LLM calls to verify RTX 5070 Ti utilization
"""

import sys
import time
import subprocess
import threading
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider

class GPUMonitor:
    """Monitor GPU utilization during LLM inference"""
    
    def __init__(self):
        self.monitoring = False
        self.gpu_stats = []
        
    def start_monitoring(self, duration_seconds=30):
        """Start monitoring GPU utilization"""
        self.monitoring = True
        self.gpu_stats = []
        
        def monitor_loop():
            while self.monitoring:
                try:
                    # Get GPU stats using nvidia-smi
                    result = subprocess.run([
                        'nvidia-smi', 
                        '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                        '--format=csv,noheader,nounits'
                    ], capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0:
                        # Parse the output
                        line = result.stdout.strip()
                        parts = line.split(', ')
                        if len(parts) >= 5:
                            gpu_util = int(parts[0])
                            mem_used = int(parts[1])
                            mem_total = int(parts[2])
                            temp = int(parts[3])
                            power = float(parts[4])
                            
                            stats = {
                                'timestamp': time.time(),
                                'gpu_utilization': gpu_util,
                                'memory_used_mb': mem_used,
                                'memory_total_mb': mem_total,
                                'memory_percent': (mem_used / mem_total) * 100,
                                'temperature_c': temp,
                                'power_draw_w': power
                            }
                            self.gpu_stats.append(stats)
                            
                except Exception as e:
                    print(f"GPU monitoring error: {e}")
                
                time.sleep(1)  # Monitor every second
        
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.start()
        
        # Auto-stop after duration
        def auto_stop():
            time.sleep(duration_seconds)
            self.stop_monitoring()
        
        threading.Thread(target=auto_stop).start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
    
    def get_stats_summary(self):
        """Get summary of GPU stats"""
        if not self.gpu_stats:
            return {}
        
        gpu_utils = [s['gpu_utilization'] for s in self.gpu_stats]
        mem_percents = [s['memory_percent'] for s in self.gpu_stats]
        temps = [s['temperature_c'] for s in self.gpu_stats]
        powers = [s['power_draw_w'] for s in self.gpu_stats]
        
        return {
            'total_samples': len(self.gpu_stats),
            'gpu_utilization': {
                'min': min(gpu_utils),
                'max': max(gpu_utils),
                'avg': sum(gpu_utils) / len(gpu_utils)
            },
            'memory_usage': {
                'min': min(mem_percents),
                'max': max(mem_percents),
                'avg': sum(mem_percents) / len(mem_percents)
            },
            'temperature': {
                'min': min(temps),
                'max': max(temps),
                'avg': sum(temps) / len(temps)
            },
            'power_draw': {
                'min': min(powers),
                'max': max(powers),
                'avg': sum(powers) / len(powers)
            }
        }


def test_gpu_utilization_during_llm():
    """Test GPU utilization during LLM inference"""
    print("🔥 GPU Utilization Test for LLM Inference")
    print("=" * 60)
    
    # Check initial GPU status
    print("📊 Initial GPU Status:")
    result = subprocess.run([
        'nvidia-smi', 
        '--query-gpu=name,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"   {result.stdout.strip()}")
    print()
    
    # Check Ollama model status
    print("🤖 Ollama Model Status:")
    result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"   {result.stdout.strip()}")
    print()
    
    # Initialize LLM client
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama3.1:8b",
        temperature=0.3,
        num_predict=256  # Longer generation to see GPU usage
    )
    
    client = LLMClient(config)
    
    # Start GPU monitoring
    monitor = GPUMonitor()
    print("🚀 Starting GPU monitoring...")
    monitor.start_monitoring(duration_seconds=45)
    
    print("⏱️  Waiting 5 seconds for baseline...")
    time.sleep(5)
    
    # Test multiple LLM calls to stress GPU
    test_prompts = [
        "Explain conflict resolution in air traffic control in detail.",
        "Describe the physics of aircraft separation and wake turbulence.",
        "What are the safety protocols for enroute air traffic management?",
        "Analyze the complexity of multi-aircraft conflict scenarios."
    ]
    
    print("🧠 Running LLM inference tests...")
    start_time = time.time()
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"   Test {i}/4: Generating response...")
        try:
            response = client._call_llm(prompt)
            print(f"   ✅ Response {i} completed ({len(response)} chars)")
        except Exception as e:
            print(f"   ❌ Response {i} failed: {e}")
        
        time.sleep(2)  # Brief pause between calls
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"⏱️  Total inference time: {total_time:.1f} seconds")
    print("⏳ Waiting for monitoring to complete...")
    
    # Wait for monitoring to finish
    time.sleep(5)
    monitor.stop_monitoring()
    
    # Analyze results
    print("\n📈 GPU Utilization Analysis:")
    print("=" * 60)
    
    stats = monitor.get_stats_summary()
    if stats:
        print(f"📊 Monitoring Summary ({stats['total_samples']} samples):")
        print(f"   GPU Utilization:  {stats['gpu_utilization']['min']}-{stats['gpu_utilization']['max']}% (avg: {stats['gpu_utilization']['avg']:.1f}%)")
        print(f"   Memory Usage:     {stats['memory_usage']['min']:.1f}-{stats['memory_usage']['max']:.1f}% (avg: {stats['memory_usage']['avg']:.1f}%)")
        print(f"   Temperature:      {stats['temperature']['min']}-{stats['temperature']['max']}°C (avg: {stats['temperature']['avg']:.1f}°C)")
        print(f"   Power Draw:       {stats['power_draw']['min']:.1f}-{stats['power_draw']['max']:.1f}W (avg: {stats['power_draw']['avg']:.1f}W)")
        print()
        
        # Analysis
        max_gpu_util = stats['gpu_utilization']['max']
        avg_gpu_util = stats['gpu_utilization']['avg']
        
        print("🔍 Analysis:")
        if max_gpu_util > 80:
            print("   ✅ HIGH GPU utilization detected - GPU is being heavily used")
        elif max_gpu_util > 40:
            print("   ✅ MODERATE GPU utilization detected - GPU is being used")
        elif max_gpu_util > 10:
            print("   ⚠️  LOW GPU utilization detected - GPU may not be optimally used")
        else:
            print("   ❌ MINIMAL GPU utilization detected - GPU may not be used")
        
        if stats['memory_usage']['avg'] > 80:
            print("   ✅ HIGH GPU memory usage - Model loaded on GPU")
        elif stats['memory_usage']['avg'] > 50:
            print("   ✅ MODERATE GPU memory usage - Model partially on GPU")
        else:
            print("   ⚠️  LOW GPU memory usage - Model may be on CPU")
        
        return max_gpu_util > 20  # Consider successful if GPU util > 20%
    
    else:
        print("❌ No GPU monitoring data collected")
        return False


def check_ollama_gpu_config():
    """Check Ollama GPU configuration and environment"""
    print("🔧 Ollama GPU Configuration Check")
    print("=" * 50)
    
    # Check CUDA availability
    print("🏗️  CUDA Environment:")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = [line for line in result.stdout.split('\n') if 'release' in line.lower()]
            if version_line:
                print(f"   ✅ CUDA: {version_line[0].strip()}")
            else:
                print("   ✅ CUDA: Available")
        else:
            print("   ❌ CUDA: Not found")
    except FileNotFoundError:
        print("   ❌ nvcc not found in PATH")
    
    # Check GPU driver
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ GPU Driver: {result.stdout.strip()}")
    except Exception:
        print("   ❌ GPU Driver: Not accessible")
    
    # Check Ollama environment variables
    print("\n⚙️  Ollama Environment:")
    
    import os
    gpu_vars = ['OLLAMA_GPU_LAYERS', 'OLLAMA_NUM_PARALLEL', 'OLLAMA_ORIGINS', 'CUDA_VISIBLE_DEVICES']
    for var in gpu_vars:
        value = os.environ.get(var)
        if value:
            print(f"   {var}: {value}")
        else:
            print(f"   {var}: (not set)")
    
    print()


def main():
    """Run GPU utilization tests"""
    print("🚀 GPU Utilization Test for Ollama LLM")
    print("=" * 70)
    print()
    
    # Check configuration
    check_ollama_gpu_config()
    
    # Run GPU utilization test
    try:
        gpu_used = test_gpu_utilization_during_llm()
        
        print("\n🏁 Final Assessment:")
        print("=" * 50)
        
        if gpu_used:
            print("✅ SUCCESS: RTX 5070 Ti is being utilized for LLM inference!")
            print("   Ollama is properly configured to use GPU acceleration")
        else:
            print("❌ CONCERN: Limited GPU utilization detected")
            print("   Consider checking Ollama GPU configuration")
        
        # Get final GPU status
        print("\n📊 Final GPU Status:")
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], capture_output=True, text=True)
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 3:
                print(f"   GPU Utilization: {parts[0]}%")
                print(f"   GPU Memory: {parts[1]}MB / {parts[2]}MB ({int(parts[1])/int(parts[2])*100:.1f}%)")
        
        return gpu_used
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
