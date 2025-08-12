#!/usr/bin/env python3
"""
Test to verify:
1. Are hardcoded positions actually used to create aircraft in BlueSky?
2. Does BlueSky simulation time advance properly after creation?
"""

from src.cdr.pipeline.cdr_pipeline import CDRPipeline, PipelineConfig
from src.cdr.simulation.bluesky_client import BlueSkyClient, create_thesis_config
import time

def test_hardcoded_positions_and_movement():
    """Test hardcoded position creation and subsequent movement"""
    
    print("🔍 Testing hardcoded positions and aircraft movement...")
    
    # Create configuration
    config = PipelineConfig(
        max_simulation_time_minutes=2.0,
        cycle_interval_seconds=30.0,  # 30 second cycles for faster testing
        save_trajectories=False,
        llm_enabled=False  # Disable LLM for pure movement test
    )
    
    # Create BlueSky client
    bluesky_config = create_thesis_config()
    bluesky_client = BlueSkyClient(bluesky_config)
    
    try:
        # Connect to BlueSky
        if not bluesky_client.connect():
            print("❌ Failed to connect to BlueSky")
            return
        
        print("✅ Connected to BlueSky")
        
        # Create pipeline (this will use the hardcoded scenario)
        pipeline = CDRPipeline(config, bluesky_client)
        
        # Create a mock scenario (will trigger hardcoded positions)
        mock_scenario = None  # This will trigger the "else" case with hardcoded scenario
        
        print("\n🎯 STEP 1: Verify hardcoded positions are used")
        print("Expected hardcoded positions:")
        print("  OWNSHIP: lat=41.978, lon=-87.904, hdg=270°, spd=450kt")
        print("  INTRUDER1: lat=42.0, lon=-87.9, hdg=90°, spd=420kt")
        
        # Initialize simulation (will create aircraft with hardcoded positions)
        pipeline._initialize_simulation(mock_scenario)
        
        # Get initial positions
        initial_states = bluesky_client.get_aircraft_states()
        
        print("\n📍 Actual initial positions from BlueSky:")
        for callsign, state in initial_states.items():
            print(f"  {callsign}: lat={state.latitude:.6f}, lon={state.longitude:.6f}, "
                  f"hdg={state.heading_deg:.1f}°, spd={state.speed_kt:.0f}kt")
        
        # Verify hardcoded positions were used
        ownship = initial_states.get('OWNSHIP')
        if ownship:
            expected_lat, expected_lon = 41.978, -87.904
            if abs(ownship.latitude - expected_lat) < 0.001 and abs(ownship.longitude - expected_lon) < 0.001:
                print("✅ Hardcoded positions WERE used for aircraft creation")
            else:
                print("❌ Hardcoded positions were NOT used correctly")
                print(f"   Expected: {expected_lat}, {expected_lon}")
                print(f"   Actual: {ownship.latitude}, {ownship.longitude}")
        
        print("\n🎯 STEP 2: Test simulation time advancement")
        
        # Pause simulation
        bluesky_client.hold()
        print("⏸️ Simulation paused")
        
        # Advance simulation by 60 seconds using FF
        print("⏩ Advancing simulation by 60 seconds...")
        ff_success = bluesky_client.ff(60.0)
        
        if not ff_success:
            print("❌ Fast-forward command failed")
            return
        
        # Get new positions
        new_states = bluesky_client.get_aircraft_states()
        
        print("\n📍 Positions after 60 seconds:")
        for callsign, state in new_states.items():
            print(f"  {callsign}: lat={state.latitude:.6f}, lon={state.longitude:.6f}, "
                  f"hdg={state.heading_deg:.1f}°, spd={state.speed_kt:.0f}kt")
        
        # Calculate movement
        print("\n📊 Movement Analysis:")
        for callsign in initial_states:
            if callsign in new_states:
                initial = initial_states[callsign]
                new = new_states[callsign]
                
                lat_diff = abs(new.latitude - initial.latitude)
                lon_diff = abs(new.longitude - initial.longitude)
                
                print(f"  {callsign}: Δlat={lat_diff:.6f}, Δlon={lon_diff:.6f}")
                
                if lat_diff > 0.001 or lon_diff > 0.001:
                    print(f"    ✅ {callsign} MOVED significantly!")
                    
                    # Calculate expected movement for verification
                    speed_kt = initial.speed_kt
                    heading_deg = initial.heading_deg
                    
                    # Rough calculation: in 60 seconds at given speed
                    distance_nm = (speed_kt / 3600) * 60  # nautical miles in 60 seconds
                    print(f"    📏 Expected distance: ~{distance_nm:.3f} NM at {speed_kt}kt")
                else:
                    print(f"    ❌ {callsign} did NOT move (positions unchanged)")
        
        print("\n🎯 STEP 3: Test continuous simulation")
        
        # Resume simulation for continuous movement
        bluesky_client.op()
        print("▶️ Resuming simulation for continuous movement...")
        
        # Wait 3 seconds of real time (should be 24 seconds of sim time at 8x speed)
        time.sleep(3)
        
        # Pause and check positions again
        bluesky_client.hold()
        continuous_states = bluesky_client.get_aircraft_states()
        
        print("\n📍 Positions after continuous simulation:")
        for callsign, state in continuous_states.items():
            print(f"  {callsign}: lat={state.latitude:.6f}, lon={state.longitude:.6f}")
        
        # Compare with previous positions
        print("\n📊 Continuous Movement Analysis:")
        for callsign in new_states:
            if callsign in continuous_states:
                prev = new_states[callsign]
                current = continuous_states[callsign]
                
                lat_diff = abs(current.latitude - prev.latitude)
                lon_diff = abs(current.longitude - prev.longitude)
                
                print(f"  {callsign}: Additional Δlat={lat_diff:.6f}, Δlon={lon_diff:.6f}")
                
                if lat_diff > 0.001 or lon_diff > 0.001:
                    print(f"    ✅ {callsign} continued moving in continuous mode!")
                else:
                    print(f"    ❌ {callsign} stopped moving in continuous mode")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        bluesky_client.disconnect()
        print("\n🧹 Disconnected from BlueSky")

if __name__ == "__main__":
    test_hardcoded_positions_and_movement()
