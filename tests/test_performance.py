#!/usr/bin/env python3
"""Performance test for LLM enhancements with verifier disabled"""

import json
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider, ConflictContext
from src.cdr.ai.performance_config import PerformanceConfig

def test_performance_optimizations():
    """Test the performance-optimized LLM resolution pipeline"""
    print("🚀 Testing Performance-Optimized LLM Pipeline")
    print("=" * 55)
    
    # Show performance configuration
    print("📊 Performance Configuration:")
    print(f"   FlashAttention 2: {'✅ Enabled' if PerformanceConfig.ENABLE_FLASH_ATTENTION else '❌ Disabled'}")
    print(f"   CUDA Graphs: {'✅ Enabled' if PerformanceConfig.ENABLE_CUDA_GRAPHS else '❌ Disabled'}")
    print(f"   Torch Compile: {'✅ Enabled' if PerformanceConfig.ENABLE_TORCH_COMPILE else '❌ Disabled'}")
    print(f"   Verifier: ❌ DISABLED (was causing issues)")
    print(f"   Agreement-of-Two: ✅ Enabled")
    print(f"   Ollama Keep-Alive: {PerformanceConfig.KEEP_ALIVE_TIMEOUT}")
    print(f"   Request Timeout: {PerformanceConfig.REQUEST_TIMEOUT_NORMAL}s")
    
    # Initialize optimized LLM client
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama3.1:8b",
        base_url="http://localhost:11434",
        enable_verifier=False,  # DISABLED
        enable_agree_on_two=True,
        enable_reprompt_on_failure=False,
        temperature=0.2,
        seed=1337
    )
    
    try:
        llm_client = LLMClient(config)
        print("\n✅ LLM client initialized with performance optimizations")
    except Exception as e:
        print(f"\n❌ Failed to initialize LLM client: {e}")
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
        "conflict_id": "PERF_TEST_001",
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
    
    # Performance benchmark: 3 resolution generations
    print("\n🏃 Performance Benchmark (3 iterations):")
    total_time = 0
    successful_resolutions = 0
    
    for i in range(3):
        print(f"\n   Iteration {i+1}/3:")
        start_time = time.perf_counter()
        
        try:
            # Update conflict ID for each iteration
            conflict_info["conflict_id"] = f"PERF_TEST_{i+1:03d}"
            
            resolution_response = llm_client.generate_resolution(context, conflict_info)
            elapsed = time.perf_counter() - start_time
            total_time += elapsed
            
            print(f"   ⏱️  Completed in {elapsed:.2f}s")
            
            if resolution_response.success:
                successful_resolutions += 1
                print(f"   ✅ Resolution: {resolution_response.resolution_type}")
                print(f"      Parameters: {resolution_response.parameters}")
                print(f"      Confidence: {resolution_response.confidence:.2f}")
                
                # Validate heading change meets requirements
                if resolution_response.resolution_type == "heading_change":
                    new_heading = resolution_response.parameters.get("new_heading_deg", 90)
                    current_heading = context.ownship_state["heading"]
                    heading_change = abs(new_heading - current_heading)
                    
                    if heading_change >= 15:
                        print(f"      ✅ Valid heading change: {heading_change:.1f}° (≥15°)")
                    else:
                        print(f"      ⚠️  Small heading change: {heading_change:.1f}° (<15°)")
                        
            else:
                print(f"   ❌ Resolution failed: {resolution_response.reasoning}")
                
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            total_time += elapsed
            print(f"   ❌ Error in {elapsed:.2f}s: {e}")
    
    # Performance summary
    avg_time = total_time / 3
    success_rate = (successful_resolutions / 3) * 100
    
    print(f"\n📊 Performance Summary:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average time: {avg_time:.2f}s")
    print(f"   Success rate: {success_rate:.1f}% ({successful_resolutions}/3)")
    print(f"   Speed improvement: ~{60/avg_time:.1f}x faster than previous verifier failures")
    
    # Display telemetry
    telemetry = llm_client.get_telemetry()
    print(f"\n📈 LLM Telemetry:")
    print(f"   Total calls: {telemetry.get('total_calls', 0)}")
    print(f"   Schema violations: {telemetry.get('schema_violations', 0)}")
    print(f"   Verifier failures: {telemetry.get('verifier_failures', 0)} (should be 0)")
    print(f"   Agreement mismatches: {telemetry.get('agreement_mismatches', 0)}")
    print(f"   Average latency: {telemetry.get('average_latency', 0):.2f}s")
    
    print(f"\n🎯 Performance test completed!")
    
    # Performance improvement analysis
    if avg_time < 30:
        print("🚀 EXCELLENT: Sub-30s average response time achieved!")
    elif avg_time < 45:
        print("✅ GOOD: Sub-45s average response time achieved!")
    else:
        print("⚠️  NEEDS WORK: Response time still over 45s")
    
    if success_rate >= 80:
        print("🎯 HIGH RELIABILITY: >80% success rate without verifier!")
    elif success_rate >= 60:
        print("✅ ACCEPTABLE: >60% success rate without verifier")
    else:
        print("❌ LOW RELIABILITY: <60% success rate - needs prompt tuning")

if __name__ == "__main__":
    test_performance_optimizations()
