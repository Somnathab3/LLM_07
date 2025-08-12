#!/usr/bin/env python3
"""
Debug Aircraft Persistence

This test checks aircraft availability and persistence in BlueSky simulation
using the new simplified client.
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.append('.')

from src.cdr.simulation.bluesky_client import BlueSkyClient


def setup_logging():
    """Setup detailed logging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def test_aircraft_persistence():
    """Test aircraft creation and persistence with simplified client"""
    logger = setup_logging()
    print("🔍 Testing Aircraft Persistence in BlueSky with Simplified Client")
    print("=" * 70)
    
    # Create simplified BlueSky client
    client = BlueSkyClient()
    
    try:
        # Initialize BlueSky
        print("🔌 Initializing BlueSky...")
        client.initialize()
        client.reset()
        print("✅ BlueSky initialized and reset")
        
        # Create first aircraft
        print("\n✈️ Creating first aircraft TEST001...")
        success = client.create_aircraft(
            acid="TEST001",
            lat=41.9786,
            lon=-87.9048,
            hdg=90,
            alt=37000,
            spd=450
        )
        
        if not success:
            print("❌ Failed to create TEST001")
            return False
        
        print("✅ Created TEST001")
        
        # Check aircraft state immediately
        print("\n📊 Checking aircraft state immediately after creation...")
        state = client.get_aircraft_state("TEST001")
        if state:
            print(f"📍 TEST001: {state.lat:.4f}, {state.lon:.4f}, "
                  f"{state.alt}ft, {state.hdg}°, {state.tas}kt")
        else:
            print("❌ TEST001 not found immediately after creation!")
        
        # Check all aircraft
        all_states = client.get_all_aircraft_states()
        print(f"Total aircraft found: {len(all_states)}")
        for aircraft_state in all_states:
            print(f"  - {aircraft_state.id}")
        
        # Step simulation a few times and check persistence
        for step in range(5):
            print(f"\n⏩ Simulation step {step + 1}...")
            client.step_simulation(1.0)  # 1 second step
            
            # Check aircraft state after each step
            state = client.get_aircraft_state("TEST001")
            if state:
                print(f"📍 TEST001: {state.lat:.4f}, {state.lon:.4f}, "
                      f"{state.alt}ft, {state.hdg}°, {state.tas}kt")
            else:
                print("❌ TEST001 disappeared!")
                break
        
        # Try to create second aircraft
        print("\n✈️ Creating second aircraft TEST002...")
        success = client.create_aircraft(
            acid="TEST002",
            lat=41.9786,
            lon=-87.8848,
            hdg=270,
            alt=37000,
            spd=420
        )
        
        if not success:
            print("❌ Failed to create TEST002")
        else:
            print("✅ Created TEST002")
        
        # Final state check
        print("\n📊 Final aircraft state check...")
        all_states = client.get_all_aircraft_states()
        print(f"Final total: {len(all_states)} aircraft")
        
        for aircraft_state in all_states:
            print(f"📍 {aircraft_state.id}: {aircraft_state.lat:.4f}, {aircraft_state.lon:.4f}, "
                  f"{aircraft_state.alt}ft, {aircraft_state.hdg}°, {aircraft_state.tas}kt")
        
        # Test simulation time
        sim_time = client.get_simulation_time()
        print(f"\n⏰ Current simulation time: {sim_time:.1f}s")
        
        return len(all_states) >= 1
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        print("\n✅ Test completed")


if __name__ == "__main__":
    success = test_aircraft_persistence()
    if success:
        print("\n✅ Aircraft persistence test completed successfully")
    else:
        print("\n❌ Aircraft persistence test failed")
    
    sys.exit(0 if success else 1)
