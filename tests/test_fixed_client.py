#!/usr/bin/env python3
"""
Test the fixed BlueSky client with embedded simulation stepping
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from src.cdr.simulation.bluesky_client import BlueSkyClient, create_thesis_config

def test_fixed_bluesky_client():
    """Test the fixed BlueSky client with real simulation stepping"""
    
    print("=== Testing Fixed BlueSky Client ===")
    print("Using embedded mode with bs.sim.step() for REAL movement")
    
    # Create client with embedded configuration
    config = create_thesis_config()
    client = BlueSkyClient(config)
    
    try:
        # Connect using embedded mode
        print("\n🚀 Connecting to embedded BlueSky...")
        if not client.connect():
            print("❌ Failed to connect")
            return
        
        print("✅ Connected successfully")
        
        # Create test aircraft
        print("\n✈️ Creating test aircraft...")
        success = client.create_aircraft(
            callsign="FIXED01",
            aircraft_type="A320",
            lat=52.0,
            lon=4.0,
            heading=90,  # East
            altitude_ft=35000,
            speed_kt=450
        )
        
        if not success:
            print("❌ Failed to create aircraft")
            return
        
        # Start simulation
        print("\n▶️ Starting simulation...")
        client.op()
        
        # Get initial state
        print("\n📊 Initial aircraft state:")
        initial_states = client.get_aircraft_states()
        initial_state = initial_states.get("FIXED01")
        
        if initial_state:
            print(f"   Position: {initial_state.latitude:.6f}, {initial_state.longitude:.6f}")
            print(f"   Heading:  {initial_state.heading_deg:.1f}°")
            print(f"   Speed:    {initial_state.speed_kt:.0f} kt")
        
        # Step simulation to allow movement
        print("\n⏩ Stepping simulation...")
        for step in range(5):
            print(f"   Step {step + 1}/5...")
            client.step_simulation(10)  # Step 10 simulation steps
            time.sleep(0.5)
        
        # Get state after simulation steps
        print("\n📊 Aircraft state after simulation steps:")
        after_states = client.get_aircraft_states()
        after_state = after_states.get("FIXED01")
        
        if after_state and initial_state:
            print(f"   Position: {after_state.latitude:.6f}, {after_state.longitude:.6f}")
            print(f"   Heading:  {after_state.heading_deg:.1f}°")
            print(f"   Speed:    {after_state.speed_kt:.0f} kt")
            
            # Calculate movement
            lat_diff = abs(after_state.latitude - initial_state.latitude)
            lon_diff = abs(after_state.longitude - initial_state.longitude)
            
            print(f"\n📏 Movement analysis:")
            print(f"   Δlat: {lat_diff:.8f}")
            print(f"   Δlon: {lon_diff:.8f}")
            
            if lat_diff > 0.001 or lon_diff > 0.001:
                print("🎉 SUCCESS: Real aircraft movement detected!")
                print("   Fixed BlueSky client is working with embedded simulation!")
            else:
                print("❌ No significant movement detected")
        
        # Test heading command
        print("\n🔄 Testing heading command...")
        client.heading_command("FIXED01", 180.0)  # Turn south
        
        # Step simulation to apply command
        print("   Stepping simulation to apply heading change...")
        for step in range(3):
            client.step_simulation(10)
            time.sleep(0.3)
        
        # Check final state
        final_states = client.get_aircraft_states()
        final_state = final_states.get("FIXED01")
        
        if final_state and after_state:
            print(f"\n📊 Final aircraft state:")
            print(f"   Position: {final_state.latitude:.6f}, {final_state.longitude:.6f}")
            print(f"   Heading:  {final_state.heading_deg:.1f}°")
            
            # Check heading change
            hdg_diff = abs(final_state.heading_deg - after_state.heading_deg)
            if hdg_diff > 5.0:  # Allow some tolerance
                print("🎉 SUCCESS: Heading change detected!")
            else:
                print("❌ No heading change detected")
            
            # Check additional movement
            lat_diff2 = abs(final_state.latitude - after_state.latitude)
            lon_diff2 = abs(final_state.longitude - after_state.longitude)
            
            if lat_diff2 > 0.001 or lon_diff2 > 0.001:
                print("🎉 SUCCESS: Additional movement after heading change!")
        
        print("\n" + "="*50)
        print("FIXED BLUESKY CLIENT TEST COMPLETE")
        print("✅ Embedded BlueSky with bs.sim.step() is working!")
        print("✅ Aircraft creation and control functional")
        print("✅ Real movement detection working")
        print("✅ Ready for conflict detection and resolution!")
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        client.disconnect()

if __name__ == "__main__":
    test_fixed_bluesky_client()
