#!/usr/bin/env python3
"""
Test aircraft movement with BlueSky time advancement
"""

from src.cdr.pipeline.cdr_pipeline import CDRPipeline, CDRConfig
from src.cdr.simulation.bluesky_client import BlueSkyClient, create_thesis_config
from src.cdr.models.scenario import MockScenario
import time

def test_aircraft_movement():
    """Test that aircraft actually move when simulation time advances"""
    
    print("Testing aircraft movement with BlueSky time advancement...")
    
    # Create configuration with shorter cycle interval for faster testing
    config = CDRConfig(
        max_simulation_time_minutes=5.0,  # Short test
        cycle_interval_seconds=10.0,      # 10 second cycles
        save_trajectories=False,
        max_llm_calls_per_scenario=1
    )
    
    # Create BlueSky client
    bluesky_config = create_thesis_config()
    bluesky_client = BlueSkyClient(bluesky_config)
    
    try:
        # Connect to BlueSky
        print("Connecting to BlueSky...")
        if not bluesky_client.connect():
            print("❌ Failed to connect to BlueSky")
            return
        
        print("✅ Connected to BlueSky")
        
        # Create a simple test aircraft
        print("Creating test aircraft...")
        success = bluesky_client.create_aircraft(
            callsign="TEST1",
            aircraft_type="B738",
            lat=42.0,
            lon=-87.0,
            heading=90,  # Flying east
            altitude_ft=35000,
            speed_kt=450
        )
        
        if not success:
            print("❌ Failed to create aircraft")
            return
        
        print("✅ Created aircraft TEST1")
        
        # Record initial position
        print("Getting initial position...")
        bluesky_client.op()  # Start simulation
        initial_states = bluesky_client.get_aircraft_states()
        initial_pos = initial_states.get("TEST1")
        
        if not initial_pos:
            print("❌ Failed to get initial position")
            return
            
        print(f"✅ Initial position: lat={initial_pos.latitude:.6f}, lon={initial_pos.longitude:.6f}")
        
        # Pause simulation
        bluesky_client.hold()
        
        # Advance time by 60 seconds and check movement
        print("Advancing simulation by 60 seconds...")
        ff_success = bluesky_client.ff(60.0)
        
        if not ff_success:
            print("❌ Failed to fast-forward simulation")
            return
            
        print("✅ Fast-forwarded 60 seconds")
        
        # Get new position
        print("Getting new position...")
        new_states = bluesky_client.get_aircraft_states()
        new_pos = new_states.get("TEST1")
        
        if not new_pos:
            print("❌ Failed to get new position")
            return
            
        print(f"✅ New position: lat={new_pos.latitude:.6f}, lon={new_pos.longitude:.6f}")
        
        # Calculate movement
        lat_diff = abs(new_pos.latitude - initial_pos.latitude)
        lon_diff = abs(new_pos.longitude - initial_pos.longitude)
        
        print(f"Movement: Δlat={lat_diff:.6f}, Δlon={lon_diff:.6f}")
        
        if lat_diff > 0.001 or lon_diff > 0.001:  # Significant movement
            print("✅ SUCCESS: Aircraft moved significantly!")
            print(f"   Initial: {initial_pos.latitude:.6f}, {initial_pos.longitude:.6f}")
            print(f"   Final:   {new_pos.latitude:.6f}, {new_pos.longitude:.6f}")
        else:
            print("❌ FAILURE: Aircraft did not move")
            print("   This indicates BlueSky simulation time is not advancing properly")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        bluesky_client.disconnect()
        print("Disconnected from BlueSky")

if __name__ == "__main__":
    test_aircraft_movement()
