#!/usr/bin/env python3
"""
Test BlueSky Direct Bridge Communication

This test verifies that the direct bridge fixes the communication issues
between BlueSky and LLM by providing direct access to BlueSky internals.
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
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def test_direct_bridge_communication():
    """Test the direct bridge communication improvements"""
    logger = setup_logging()
    print("🔗 Testing BlueSky Direct Bridge Communication")
    print("=" * 60)
    
    try:
        # Create BlueSky client with direct bridge
        config = BlueSkyConfig()
        client = BlueSkyClient(config)
        
        print("🚀 Connecting to BlueSky...")
        if not client.connect():
            print("❌ Failed to connect to BlueSky")
            return False
        
        print("✅ Connected to BlueSky")
        
        # Test direct bridge availability
        bridge = get_direct_bridge()
        if bridge.is_available():
            print("✅ Direct bridge is available and functional")
        else:
            print("❌ Direct bridge is not available")
            return False
        
        # Initialize simulation
        client._initialize_simulation()
        
        # Create test aircraft
        print("\n📍 Creating test aircraft...")
        
        # Aircraft 1 (ownship)
        success1 = client.create_aircraft(
            callsign="OWNSHIP",
            aircraft_type="B738",
            lat=41.9786,
            lon=-87.9048,
            heading=90,
            altitude_ft=37000,
            speed_kt=450
        )
        
        # Aircraft 2 (intruder, close proximity)
        success2 = client.create_aircraft(
            callsign="INTRUDER", 
            aircraft_type="A320",
            lat=41.9786,        # Same latitude (collision course)
            lon=-87.8948,       # About 0.6 NM apart
            heading=270,        # Opposite direction
            altitude_ft=37000,  # Same altitude
            speed_kt=420
        )
        
        if not (success1 and success2):
            print("❌ Failed to create test aircraft")
            return False
        
        print("✅ Created test aircraft successfully")
        
        # Start simulation
        client.op()
        time.sleep(2)  # Let aircraft move
        
        # Test 1: Aircraft State Retrieval via Direct Bridge
        print("\n🔍 Test 1: Aircraft State Retrieval")
        
        # Test direct bridge state access
        direct_states = bridge.get_aircraft_states_direct()
        print(f"Direct bridge found {len(direct_states)} aircraft")
        
        for callsign, state in direct_states.items():
            print(f"   📊 {callsign}: {state.latitude:.4f},{state.longitude:.4f}, "
                  f"{state.altitude_ft:.0f}ft, {state.heading_deg:.1f}°, {state.speed_kt:.0f}kt")
        
        # Test client state access (should use direct bridge internally)
        client_states = client.get_aircraft_states()
        print(f"Client found {len(client_states)} aircraft")
        
        # Verify consistency
        if len(direct_states) == len(client_states):
            print("✅ State retrieval consistency verified")
        else:
            print(f"⚠️ State count mismatch: direct={len(direct_states)}, client={len(client_states)}")
        
        # Test 2: Conflict Detection via Direct Bridge
        print("\n⚠️  Test 2: Conflict Detection")
        
        # Test direct bridge conflict detection
        direct_conflicts = bridge.get_conflicts_direct()
        print(f"Direct bridge found {len(direct_conflicts)} conflicts")
        
        for conflict in direct_conflicts:
            print(f"   🚨 {conflict.aircraft1} vs {conflict.aircraft2}: "
                  f"h={conflict.horizontal_distance:.1f}NM, v={conflict.vertical_distance:.0f}ft, "
                  f"type={conflict.conflict_type}, source={conflict.detection_source}")
        
        # Test client conflict detection (should use direct bridge internally)
        client_conflicts = client.get_conflicts()
        print(f"Client found {len(client_conflicts)} conflicts")
        
        # Test 3: Command Application via Direct Bridge
        print("\n🎯 Test 3: Command Application")
        
        # Test heading command via direct bridge
        print("Testing heading command...")
        new_heading = 120
        
        # Direct bridge command
        bridge_success = bridge.apply_heading_command_direct("OWNSHIP", new_heading)
        print(f"Direct bridge heading command: {'✅ Success' if bridge_success else '❌ Failed'}")
        
        # Client command (should use direct bridge internally)
        client_success = client.heading_command("INTRUDER", 240)
        print(f"Client heading command: {'✅ Success' if client_success else '❌ Failed'}")
        
        # Wait for commands to take effect
        time.sleep(2)
        
        # Verify heading changes
        updated_states = client.get_aircraft_states()
        
        if "OWNSHIP" in updated_states:
            actual_heading = updated_states["OWNSHIP"].heading_deg
            print(f"OWNSHIP heading updated to: {actual_heading:.1f}°")
            
            if abs(actual_heading - new_heading) < 5:
                print("✅ Heading command verification successful")
            else:
                print(f"⚠️ Heading not updated as expected (target: {new_heading}°, actual: {actual_heading:.1f}°)")
        
        # Test 4: Data Conversion and Bridge Reliability
        print("\n🔄 Test 4: Data Conversion")
        
        # Test format conversion
        standard_states = bridge.convert_to_standard_format(direct_states)
        standard_conflicts = bridge.convert_conflicts_to_standard(direct_conflicts)
        
        print(f"✅ Converted {len(standard_states)} states and {len(standard_conflicts)} conflicts to standard format")
        
        # Verify data integrity
        for callsign in direct_states:
            if callsign in standard_states:
                direct_lat = direct_states[callsign].latitude
                standard_lat = standard_states[callsign]['latitude']
                
                if abs(direct_lat - standard_lat) < 0.0001:
                    print(f"✅ Data integrity verified for {callsign}")
                else:
                    print(f"❌ Data integrity check failed for {callsign}")
                    return False
        
        # Summary
        print("\n📊 Test Summary:")
        print(f"   Aircraft states: {len(client_states)} found")
        print(f"   Conflicts detected: {len(client_conflicts)}")
        print(f"   Direct bridge: {'Available' if bridge.is_available() else 'Unavailable'}")
        print(f"   Bridge usage: {'Enabled' if client.use_direct_bridge else 'Disabled'}")
        
        if len(client_states) >= 2:
            print("✅ SUCCESS: Direct bridge communication working properly")
            return True
        else:
            print("❌ FAILURE: Insufficient aircraft states retrieved")
            return False
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            client.disconnect()
            print("🔌 BlueSky disconnected")
        except:
            pass


def test_bridge_vs_fallback():
    """Compare direct bridge performance vs fallback methods"""
    print("\n🏁 Comparing Direct Bridge vs Fallback Performance")
    print("=" * 50)
    
    config = BlueSkyConfig()
    client = BlueSkyClient(config)
    
    if not client.connect():
        print("❌ Failed to connect for performance test")
        return
    
    client._initialize_simulation()
    
    # Create several test aircraft
    aircraft_count = 5
    for i in range(aircraft_count):
        client.create_aircraft(
            callsign=f"TEST{i:03d}",
            aircraft_type="A320",
            lat=42.0 + i * 0.01,
            lon=-87.0 + i * 0.01,
            heading=90 + i * 10,
            altitude_ft=35000 + i * 1000,
            speed_kt=450
        )
    
    client.op()
    time.sleep(1)
    
    # Test 1: State retrieval performance
    print(f"\n📊 Testing state retrieval for {aircraft_count} aircraft...")
    
    # Direct bridge
    start_time = time.time()
    direct_states = client.direct_bridge.get_aircraft_states_direct() if client.direct_bridge else {}
    bridge_time = time.time() - start_time
    
    # Fallback method
    client.use_direct_bridge = False  # Force fallback
    start_time = time.time()
    fallback_states = client._get_aircraft_states_embedded()
    fallback_time = time.time() - start_time
    client.use_direct_bridge = True  # Re-enable
    
    print(f"   Direct bridge: {len(direct_states)} states in {bridge_time:.3f}s")
    print(f"   Fallback method: {len(fallback_states)} states in {fallback_time:.3f}s")
    
    if fallback_time > 0 and bridge_time > 0:
        if bridge_time < fallback_time:
            print(f"   ✅ Direct bridge is {(fallback_time/bridge_time):.1f}x faster")
        else:
            print(f"   ⚠️ Fallback is {(bridge_time/fallback_time):.1f}x faster")
    else:
        print("   ⚡ Both methods very fast (< 1ms)")
    
    client.disconnect()


if __name__ == "__main__":
    print("🧪 BlueSky Direct Bridge Communication Test Suite")
    print("=" * 60)
    
    # Main communication test
    main_success = test_direct_bridge_communication()
    
    if main_success:
        # Performance comparison
        test_bridge_vs_fallback()
        
        print("\n🎉 COMPLETE SUCCESS: Direct bridge communication verified!")
        print("✅ BlueSky-to-LLM communication issues should be resolved")
    else:
        print("\n❌ FAILURE: Direct bridge communication issues remain")
        print("💡 Check BlueSky installation and direct bridge implementation")
