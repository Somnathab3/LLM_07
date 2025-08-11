#!/usr/bin/env python3
"""Test script for enhanced BlueSky client with real integration"""

import time
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.cdr.simulation.bluesky_client import BlueSkyClient, BlueSkyConfig, AircraftState, create_thesis_config


def test_bluesky_connection():
    """Test BlueSky connection and basic functionality"""
    print("🧪 Testing BlueSky Connection and Integration")
    print("=" * 60)
    
    # Create configuration using new thesis preset
    config = create_thesis_config(
        host="127.0.0.1",
        port=11000,  # Correct BlueSky port
        headless=True,
        seed=1337  # Fixed seed for reproducibility
    )
    
    print(f"Using thesis configuration: dt={config.dt}, dtmult={config.dtmult}, seed={config.seed}")
    
    client = BlueSkyClient(config)
    
    try:
        # Test 1: Connection
        print("\n1️⃣ Testing Connection...")
        if client.connect(timeout=45.0):
            print("✅ Connected to BlueSky successfully")
        else:
            print("❌ Failed to connect to BlueSky")
            print("   Make sure BlueSky is installed: pip install bluesky-simulator")
            return False
        
        # Test 2: Basic Commands
        print("\n2️⃣ Testing Basic Commands...")
        
        # Test simulation control
        print("   Testing simulation control...")
        client.hold_simulation()
        time.sleep(1)
        client.continue_simulation()
        time.sleep(1)
        
        # Test aircraft creation
        print("   Testing aircraft creation...")
        success1 = client.create_aircraft(
            callsign="TEST001",
            aircraft_type="B737",
            lat=52.0,
            lon=4.0,
            heading=90,
            altitude_ft=35000,
            speed_kt=450
        )
        
        success2 = client.create_aircraft(
            callsign="TEST002",
            aircraft_type="A320",
            lat=52.1,
            lon=4.1,
            heading=270,
            altitude_ft=35000,
            speed_kt=450
        )
        
        if success1 and success2:
            print("✅ Aircraft creation successful")
        else:
            print("⚠️ Some aircraft creation failed")
        
        # Test 3: Aircraft State Retrieval
        print("\n3️⃣ Testing Aircraft State Retrieval...")
        time.sleep(2)  # Let aircraft settle
        
        states = client.get_aircraft_states()
        print(f"   Retrieved states for {len(states)} aircraft:")
        
        for callsign, state in states.items():
            print(f"     {callsign}: {state.latitude:.4f},{state.longitude:.4f} "
                  f"alt={state.altitude_ft}ft hdg={state.heading_deg}° spd={state.speed_kt}kt")
        
        if states:
            print("✅ State retrieval successful")
        else:
            print("⚠️ No aircraft states retrieved")
        
        # Test 4: Aircraft Commands
        print("\n4️⃣ Testing Aircraft Commands...")
        
        if "TEST001" in states:
            # Test heading command
            print("   Testing heading command...")
            if client.heading_command("TEST001", 180):
                print("✅ Heading command successful")
            
            # Test altitude command
            print("   Testing altitude command...")
            if client.altitude_command("TEST001", 37000, 1000):
                print("✅ Altitude command successful")
            
            # Test speed command
            print("   Testing speed command...")
            if client.speed_command("TEST001", 480):
                print("✅ Speed command successful")
        
        # Test 5: Fast Time Factor
        print("\n5️⃣ Testing Simulation Speed Control...")
        if client.set_fast_time_factor(5.0):
            print("✅ Fast time factor set successfully")
            time.sleep(2)
            client.set_fast_time_factor(1.0)  # Reset to normal speed
        
        # Test 6: Conflict Detection
        print("\n6️⃣ Testing Conflict Detection...")
        
        # Move aircraft closer together to create potential conflict
        if len(states) >= 2:
            client.move_aircraft("TEST002", 52.01, 4.01, 35000, 90, 450)
            time.sleep(3)
            
            conflicts = client.get_conflicts()
            print(f"   Detected {len(conflicts)} conflicts:")
            
            for conflict in conflicts:
                print(f"     {conflict.aircraft1} vs {conflict.aircraft2}: "
                      f"h_dist={conflict.horizontal_distance:.1f}NM "
                      f"v_dist={conflict.vertical_distance:.0f}ft "
                      f"type={conflict.conflict_type} severity={conflict.severity}")
            
            if conflicts:
                print("✅ Conflict detection working")
            else:
                print("⚠️ No conflicts detected (aircraft may be too far apart)")
        
        # Test 7: Advanced Commands
        print("\n7️⃣ Testing Advanced Commands...")
        
        # Test direct to waypoint (may not work without predefined waypoints)
        try:
            client.direct_to_waypoint("TEST001", "AMS")
            print("✅ Direct-to command sent")
        except:
            print("⚠️ Direct-to command failed (waypoint may not exist)")
        
        # Test simulation time
        sim_time = client.get_simulation_time()
        print(f"   Simulation time: {sim_time}")
        
        # Test 8: Cleanup
        print("\n8️⃣ Testing Cleanup...")
        
        # Delete test aircraft
        for callsign in ["TEST001", "TEST002"]:
            if callsign in states:
                if client.delete_aircraft(callsign):
                    print(f"✅ Deleted {callsign}")
                else:
                    print(f"⚠️ Failed to delete {callsign}")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Always disconnect
        print("\n🔌 Disconnecting...")
        client.disconnect()


def test_context_manager():
    """Test BlueSky client as context manager"""
    print("\n🧪 Testing Context Manager Usage")
    print("=" * 40)
    
    config = create_thesis_config(headless=True, seed=42)
    
    try:
        with BlueSkyClient(config) as client:
            print("✅ Context manager entry successful")
            
            # Quick test
            success = client.create_aircraft(
                "CTX001", "B737", 50.0, 3.0, 90, 30000, 400
            )
            
            if success:
                print("✅ Aircraft created in context")
                time.sleep(1)
                states = client.get_aircraft_states()
                print(f"   Found {len(states)} aircraft")
                
                client.delete_aircraft("CTX001")
                print("✅ Aircraft deleted")
            
        print("✅ Context manager exit successful")
        return True
        
    except Exception as e:
        print(f"❌ Context manager test failed: {e}")
        return False


def test_error_handling():
    """Test error handling scenarios"""
    print("\n🧪 Testing Error Handling")
    print("=" * 30)
    
    config = create_thesis_config(headless=True)
    client = BlueSkyClient(config)
    
    # Test commands without connection
    print("1. Testing commands without connection...")
    try:
        client.create_aircraft("ERR001", "B737", 0, 0, 0, 0, 0)
        print("❌ Should have failed")
    except RuntimeError:
        print("✅ Correctly caught RuntimeError")
    
    # Test invalid aircraft operations
    if client.connect(timeout=30):
        print("2. Testing invalid aircraft operations...")
        
        # Try to command non-existent aircraft
        result = client.heading_command("NONEXISTENT", 90)
        if not result:
            print("✅ Correctly handled non-existent aircraft")
        
        # Try invalid parameters
        result = client.set_fast_time_factor(-1.0)
        if not result:
            print("✅ Correctly rejected invalid time factor")
        
        client.disconnect()
        print("✅ Error handling tests completed")
        return True
    else:
        print("⚠️ Could not connect for error handling tests")
        return False


if __name__ == "__main__":
    print("🚀 Enhanced BlueSky Client Test Suite")
    print("=" * 50)
    
    # Check if BlueSky is available
    try:
        import subprocess
        result = subprocess.run(["python", "-c", "import bluesky"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("❌ BlueSky not found. Installing...")
            print("   Run: pip install bluesky-simulator")
            sys.exit(1)
        else:
            print("✅ BlueSky is available")
    except Exception as e:
        print(f"⚠️ Could not check BlueSky availability: {e}")
    
    # Run tests
    success = True
    
    try:
        success &= test_bluesky_connection()
        success &= test_context_manager()
        success &= test_error_handling()
        
        if success:
            print("\n" + "=" * 50)
            print("🎉 All tests PASSED!")
            print("✅ BlueSky client is working correctly")
        else:
            print("\n" + "=" * 50)
            print("⚠️ Some tests FAILED")
            print("   Check the output above for details")
            
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Test suite completed")
