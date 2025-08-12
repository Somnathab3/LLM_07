#!/usr/bin/env python3
"""
Quick test of the new simplified BlueSky client
"""

import time
import sys

# Add project root to path
sys.path.append('.')

try:
    from src.cdr.simulation.bluesky_client import BlueSkyClient
except ImportError:
    print("❌ Could not import BlueSky client")
    sys.exit(1)


def quick_autopilot_test():
    """Quick test of autopilot functionality"""
    print("🚁 Quick Autopilot Test with Simplified BlueSky Client")
    print("=" * 60)
    
    # Initialize simplified BlueSky client
    bs_client = BlueSkyClient()
    bs_client.initialize()
    bs_client.reset()
    
    # Create one test aircraft
    print("✈️ Creating test aircraft...")
    success = bs_client.create_aircraft(
        acid="TEST001",
        lat=42.0,
        lon=-86.5,
        hdg=90,
        alt=35000,
        spd=450
    )
    
    if success:
        print("   📝 Created TEST001")
    else:
        print("   ❌ Failed to create TEST001")
    
    # Configure autopilot
    print("🤖 Configuring autopilot...")
    autopilot_success = bs_client.set_autopilot(
        acid="TEST001",
        alt=35000,
        spd=450,
        hdg=90
    )
    
    if autopilot_success:
        print("   ✅ Autopilot configured")
    else:
        print("   ❌ Autopilot failed")
    
    # Monitor for 60 seconds
    print("📊 Monitoring for 60 seconds...")
    start_time = time.time()
    initial_state = None
    
    while time.time() - start_time < 60:
        current_time = time.time() - start_time
        
        # Step simulation
        bs_client.step_simulation(5)
        
        # Get state
        state = bs_client.get_aircraft_state("TEST001")
        if state:
            if initial_state is None:
                initial_state = state
                print(f"   Initial: SPD={state.tas:.1f}kt, ALT={state.alt:.0f}ft")
            
            if int(current_time) % 15 == 0:  # Every 15 seconds
                print(f"   {current_time:4.0f}s: SPD={state.tas:.1f}kt, ALT={state.alt:.0f}ft, HDG={state.hdg:.1f}°")
        
        time.sleep(5)
    
    # Final analysis
    final_state = bs_client.get_aircraft_state("TEST001")
    if initial_state and final_state:
        speed_loss = initial_state.tas - final_state.tas
        speed_loss_percent = (speed_loss / initial_state.tas * 100) if initial_state.tas > 0 else 0
        
        print(f"\n📈 Results:")
        print(f"   Initial Speed: {initial_state.tas:.1f}kt")
        print(f"   Final Speed: {final_state.tas:.1f}kt")
        print(f"   Speed Loss: {speed_loss:.1f}kt ({speed_loss_percent:.1f}%)")
        
        if speed_loss_percent < 5:
            print(f"   ✅ EXCELLENT: Speed very stable")
        elif speed_loss_percent < 15:
            print(f"   ✅ GOOD: Acceptable stability")
        else:
            print(f"   ⚠️ CONCERNING: Significant degradation")
    
    print("✅ Test completed")


if __name__ == "__main__":
    quick_autopilot_test()
