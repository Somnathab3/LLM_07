#!/usr/bin/env python3
"""
Debug Aircraft Persistenc        # Create first aircraft
        print("\n✈️ Cr        # Try to create second aircraft
        print("\n✈️ Creating second aircraft TEST002...")
        success = client.create_aircraft(
            callsign="TEST002", 
            aircraft_type="A320",
            lat=41.9786,
            lon=-87.8848,
            altitude_ft=37000,
            heading=270,
            speed_kt=420
        )t aircraft TEST001...")
        success = client.create_aircraft(
            callsign="TEST001",
            aircraft_type="B738",
            lat=41.9786,
            lon=-87.9048,
            altitude_ft=37000,
            heading=90,
            speed_kt=450
        )his test checks why aircraft are disappearing from BlueSky simulation.
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.append('.')

from src.cdr.simulation.bluesky_client import BlueSkyClient, BlueSkyConfig
from src.cdr.simulation.bluesky_direct_bridge import get_direct_bridge


def setup_logging():
    """Setup detailed logging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def test_aircraft_persistence():
    """Test aircraft creation and persistence"""
    logger = setup_logging()
    print("🔍 Testing Aircraft Persistence in BlueSky")
    
    # Create BlueSky client
    config = BlueSkyConfig(
        headless=True,
        dt=1.0,
        dtmult=1.0,  # Real-time for debugging
        seed=1337,
        asas_enabled=True,
        reso_off=True
    )
    
    client = BlueSkyClient(config)
    
    try:
        # Connect to BlueSky
        print("🔌 Connecting to BlueSky...")
        if not client.connect():
            print("❌ Failed to connect to BlueSky")
            return False
        
        print("✅ Connected to BlueSky")
        
        # Create first aircraft
        print("\\n✈️ Creating first aircraft TEST001...")
        success = client.create_aircraft(
            callsign="TEST001",
            aircraft_type="B738",
            lat=41.9786,
            lon=-87.9048,
            altitude_ft=37000,
            heading=90,
            speed_kt=450
        )
        
        if not success:
            print("❌ Failed to create TEST001")
            return False
        
        print("✅ Created TEST001")
        
        # Check aircraft state immediately
        print("\\n📊 Checking aircraft state immediately after creation...")
        states = client.get_aircraft_states()
        print(f"Found {len(states)} aircraft: {list(states.keys())}")
        
        if "TEST001" in states:
            state = states["TEST001"]
            print(f"📍 TEST001: {state.latitude:.4f}, {state.longitude:.4f}, "
                  f"{state.altitude_ft}ft, {state.heading_deg}°")
        else:
            print("❌ TEST001 not found immediately after creation!")
        
        # Step simulation a few times and check persistence
        for step in range(5):
            print(f"\\n⏩ Simulation step {step + 1}...")
            client.step_simulation(0.1)  # 0.1 minute step
            
            # Check aircraft state after each step
            states = client.get_aircraft_states()
            print(f"After step {step + 1}: {len(states)} aircraft: {list(states.keys())}")
            
            if "TEST001" in states:
                state = states["TEST001"]
                print(f"📍 TEST001: {state.latitude:.4f}, {state.longitude:.4f}, "
                      f"{state.altitude_ft}ft, {state.heading_deg}°")
            else:
                print("❌ TEST001 disappeared!")
                break
        
        # Try to create second aircraft
        print("\\n✈️ Creating second aircraft TEST002...")
        success = client.create_aircraft(
            callsign="TEST002", 
            aircraft_type="A320",
            lat=41.9786,
            lon=-87.8848,
            altitude_ft=37000,
            heading=270,
            speed_kt=420
        )
        
        if not success:
            print("❌ Failed to create TEST002")
        else:
            print("✅ Created TEST002")
        
        # Final state check
        print("\\n📊 Final aircraft state check...")
        states = client.get_aircraft_states()
        print(f"Final: {len(states)} aircraft: {list(states.keys())}")
        
        for callsign, state in states.items():
            print(f"📍 {callsign}: {state.latitude:.4f}, {state.longitude:.4f}, "
                  f"{state.altitude_ft}ft, {state.heading_deg}°")
        
        # Check with direct bridge
        print("\\n🔍 Checking with direct bridge...")
        bridge = get_direct_bridge()
        if bridge and bridge.is_available():
            direct_states = bridge.get_aircraft_states_direct()
            print(f"Direct bridge: {len(direct_states)} aircraft: {list(direct_states.keys())}")
            
            for callsign, state in direct_states.items():
                print(f"📍 (Direct) {callsign}: {state.latitude:.4f}, {state.longitude:.4f}, "
                      f"{state.altitude_ft}ft, {state.heading_deg}°")
        else:
            print("❌ Direct bridge not available")
        
        return len(states) >= 1
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        print("\\n🔌 Disconnecting from BlueSky...")
        client.disconnect()
        print("✅ Disconnected")


if __name__ == "__main__":
    success = test_aircraft_persistence()
    if success:
        print("\\n✅ Aircraft persistence test completed")
    else:
        print("\\n❌ Aircraft persistence test failed")
    
    sys.exit(0 if success else 1)
