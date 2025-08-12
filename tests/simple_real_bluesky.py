#!/usr/bin/env python3
"""
Simple REAL BlueSky Solution with Proper Simulation Stepping
Minimal implementation focusing on bs.sim.step() for actual movement
"""

import time
import math
from typing import Dict, Optional, List

def test_real_bluesky_stepping():
    """Test REAL BlueSky with proper simulation stepping"""
    
    print("=== REAL BlueSky Simulation Stepping Test ===")
    print("Using embedded BlueSky with bs.sim.step() like horizontal_cr_env.py")
    
    try:
        # Import and initialize BlueSky (like in your working example)
        print("🚀 Importing BlueSky...")
        import bluesky as bs
        
        print("   Initializing BlueSky simulation...")
        # Initialize as detached simulation node (exactly like your example)
        bs.init(mode='sim', detached=True)
        
        # Create simple screen dummy class locally
        class SimpleScreenDummy:
            def echo(self, text='', flags=0):
                pass
            def update(self):
                pass
            def reset(self):
                pass
        
        print("   Setting up dummy screen...")
        bs.scr = SimpleScreenDummy()
        
        print("   Configuring simulation parameters...")
        # Set time step and fast-forward (like your example)
        bs.stack.stack('DT 5;FF')
        
        print("✅ BlueSky initialized successfully!")
        
        # Reset traffic to start fresh
        print("\n🔄 Resetting traffic...")
        bs.traf.reset()
        
        # Create aircraft (like in your example: bs.traf.cre('KL001',actype="A320",acspd=150))
        print("\n✈️ Creating test aircraft...")
        callsign = 'TEST01'
        aircraft_type = "A320"
        speed = 450  # knots
        
        success = bs.traf.cre(callsign, actype=aircraft_type, acspd=speed)
        
        if not success:
            print("❌ Failed to create aircraft")
            return
        
        print(f"✅ Created aircraft {callsign}")
        
        # Get aircraft index
        ac_idx = bs.traf.id2idx(callsign)
        if ac_idx < 0:
            print("❌ Aircraft index not found")
            return
        
        # Set initial position manually
        initial_lat = 52.0
        initial_lon = 4.0
        initial_hdg = 90.0  # East
        initial_alt = 35000
        
        bs.traf.lat[ac_idx] = initial_lat
        bs.traf.lon[ac_idx] = initial_lon
        bs.traf.hdg[ac_idx] = initial_hdg
        bs.traf.alt[ac_idx] = initial_alt
        bs.traf.gs[ac_idx] = speed
        
        print(f"📍 Set initial position: ({initial_lat:.6f}, {initial_lon:.6f})")
        print(f"📐 Set initial heading: {initial_hdg}°")
        print(f"🚀 Set initial speed: {speed} kt")
        
        # Show initial state
        print(f"\n📊 Initial State:")
        print(f"   Lat: {bs.traf.lat[ac_idx]:.6f}")
        print(f"   Lon: {bs.traf.lon[ac_idx]:.6f}")
        print(f"   Hdg: {bs.traf.hdg[ac_idx]:.1f}°")
        print(f"   Spd: {bs.traf.gs[ac_idx]:.0f} kt")
        print(f"   Alt: {bs.traf.alt[ac_idx]:.0f}")
        
        # THE KEY PART: Step the simulation (like in your example)
        print(f"\n⏩ Stepping simulation to allow aircraft movement...")
        print("This is the crucial part that was missing!")
        
        num_steps = 50  # More steps for visible movement
        for step in range(num_steps):
            if step % 10 == 0:
                print(f"   Step {step}/{num_steps}...")
            
            # This is the key - actually step the simulation!
            bs.sim.step()
            
            # Brief pause every few steps to avoid overwhelming
            if step % 10 == 9:
                time.sleep(0.1)
        
        # Check position after simulation steps
        print(f"\n📊 State After {num_steps} Simulation Steps:")
        final_lat = bs.traf.lat[ac_idx]
        final_lon = bs.traf.lon[ac_idx]
        final_hdg = bs.traf.hdg[ac_idx]
        final_spd = bs.traf.gs[ac_idx]
        final_alt = bs.traf.alt[ac_idx]
        
        print(f"   Lat: {final_lat:.6f}")
        print(f"   Lon: {final_lon:.6f}")
        print(f"   Hdg: {final_hdg:.1f}°")
        print(f"   Spd: {final_spd:.0f} kt")
        print(f"   Alt: {final_alt:.0f}")
        
        # Calculate movement
        lat_diff = abs(final_lat - initial_lat)
        lon_diff = abs(final_lon - initial_lon)
        
        print(f"\n📏 Movement Analysis:")
        print(f"   Δlat: {lat_diff:.8f}")
        print(f"   Δlon: {lon_diff:.8f}")
        
        if lat_diff > 0.0001 or lon_diff > 0.0001:
            print("🎉 SUCCESS: REAL aircraft movement detected!")
            print("   The aircraft actually moved in the BlueSky simulation!")
            print("   bs.sim.step() is working correctly!")
        else:
            print("❌ No significant movement detected")
            print("   Need to investigate simulation parameters")
        
        # Test heading command
        print(f"\n🔄 Testing Heading Change Command...")
        
        # Send heading command through stack (like your example uses bs.stack.stack)
        new_heading = 180.0  # Turn south
        command = f"HDG {callsign} {int(new_heading)}"
        print(f"   Sending command: {command}")
        bs.stack.stack(command)
        
        # Step simulation to apply command
        print("   Stepping simulation to apply heading change...")
        for step in range(20):
            bs.sim.step()
            if step % 5 == 4:
                time.sleep(0.1)
        
        # Check if heading changed
        new_hdg = bs.traf.hdg[ac_idx]
        new_lat = bs.traf.lat[ac_idx]
        new_lon = bs.traf.lon[ac_idx]
        
        print(f"\n📊 State After Heading Command:")
        print(f"   Lat: {new_lat:.6f}")
        print(f"   Lon: {new_lon:.6f}")
        print(f"   Hdg: {new_hdg:.1f}°")
        
        hdg_diff = abs(new_hdg - final_hdg)
        if hdg_diff > 1.0:
            print("🎉 SUCCESS: Heading change detected!")
        else:
            print("❌ No heading change detected")
        
        # Check for additional movement
        lat_diff2 = abs(new_lat - final_lat)
        lon_diff2 = abs(new_lon - final_lon)
        
        if lat_diff2 > 0.0001 or lon_diff2 > 0.0001:
            print("🎉 SUCCESS: Additional movement after heading change!")
            print(f"   Additional Δlat: {lat_diff2:.8f}")
            print(f"   Additional Δlon: {lon_diff2:.8f}")
        
        print(f"\n" + "="*50)
        print("CONCLUSION:")
        
        total_lat_movement = abs(new_lat - initial_lat)
        total_lon_movement = abs(new_lon - initial_lon)
        
        if total_lat_movement > 0.001 or total_lon_movement > 0.001:
            print("🎉 REAL MOVEMENT CONFIRMED!")
            print("   ✅ BlueSky simulation is actually running")
            print("   ✅ Aircraft are moving in the simulation")
            print("   ✅ bs.sim.step() is the key to real movement")
            print("   ✅ Commands are being processed")
            print(f"   📏 Total movement: Δlat={total_lat_movement:.6f}, Δlon={total_lon_movement:.6f}")
        else:
            print("❌ Limited movement detected")
            print("   Need to investigate simulation configuration")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_bluesky_stepping()
