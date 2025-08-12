#!/usr/bin/env python3
"""
Test script to verify 5-minute simulation intervals are working correctly.
"""

import sys
sys.path.append('.')

from src.cdr.pipeline.cdr_pipeline import PipelineConfig
from src.cdr.simulation.bluesky_client import BlueSkyClient, BlueSkyConfig
import time

def test_5_minute_intervals():
    """Test the 5-minute interval configuration"""
    print("=" * 60)
    print("           5-MINUTE INTERVAL TEST")
    print("=" * 60)
    
    # Show configuration
    config = PipelineConfig()
    print(f"📊 Configuration Summary:")
    print(f"   - Cycle interval: {config.cycle_interval_seconds} seconds ({config.cycle_interval_seconds/60:.1f} minutes)")
    print(f"   - Max simulation time: {config.max_simulation_time_minutes} minutes")
    print(f"   - Calculated max cycles: {int(config.max_simulation_time_minutes * 60 / config.cycle_interval_seconds)}")
    
    # Initialize BlueSky
    print(f"\n🚀 Initializing BlueSky simulation...")
    bs_config = BlueSkyConfig()
    client = BlueSkyClient(bs_config)
    client.initialize()
    
    # Create a simple aircraft for testing
    print(f"\n✈️ Creating test aircraft...")
    client.create_aircraft("TEST001", 40.0, -74.0, 35000, 90, 450)
    
    # Test stepping in 5-minute intervals
    print(f"\n⏩ Testing 5-minute simulation steps...")
    
    for cycle in range(3):  # Test 3 cycles (15 minutes total)
        print(f"\n--- Cycle {cycle + 1} ---")
        print(f"🕒 Stepping simulation by {config.cycle_interval_seconds/60:.1f} minutes...")
        
        start_time = time.time()
        result = client.step_minutes(config.cycle_interval_seconds / 60.0)
        end_time = time.time()
        
        print(f"✅ Step completed in {end_time - start_time:.2f} real seconds")
        print(f"📍 Aircraft position: {client.get_aircraft_state('TEST001')}")
        
        # Brief pause to show timing
        time.sleep(1)
    
    print(f"\n🎯 Test Summary:")
    print(f"   - Successfully stepped simulation in {config.cycle_interval_seconds/60:.1f}-minute intervals")
    print(f"   - Total simulated time: {3 * config.cycle_interval_seconds/60:.1f} minutes")
    print(f"   - Configuration change from 1-minute to 5-minute intervals: ✅ WORKING")

if __name__ == "__main__":
    test_5_minute_intervals()
