#!/usr/bin/env python3
"""
Enhanced BlueSky Client Test with Native Conflict Detection

This test demonstrates the enhanced SimpleBlueSkyClient with integrated
BlueSky native ConflictDetection capabilities.
"""

import sys
import time
sys.path.append('.')

from src.cdr.simulation.bluesky_client import SimpleBlueSkyClient


def test_enhanced_bluesky_client():
    """Test enhanced BlueSky client with native conflict detection"""
    print("🚀 Enhanced BlueSky Client with Native Conflict Detection")
    print("=" * 70)
    
    # Initialize enhanced client
    client = SimpleBlueSkyClient()
    client.initialize()
    client.reset()
    
    print("✅ Enhanced BlueSky client initialized")
    
    # Configure native conflict detection
    print("\n🛡️  Configuring Native Conflict Detection...")
    success = client.configure_conflict_detection(
        pz_radius_nm=5.0,      # 5 nautical mile protected zone
        pz_height_ft=1000,     # 1000 feet vertical separation
        lookahead_sec=300      # 5 minute lookahead
    )
    
    if success:
        print("✅ Native conflict detection configured")
    else:
        print("❌ Failed to configure conflict detection")
        return False
    
    # Create multiple aircraft scenarios
    print("\n✈️  Creating Multi-Aircraft Conflict Scenario...")
    
    aircraft_scenarios = [
        {"id": "EAST1", "lat": 42.0, "lon": -87.2, "hdg": 90, "alt": 35000, "spd": 450},
        {"id": "WEST1", "lat": 42.0, "lon": -86.8, "hdg": 270, "alt": 35000, "spd": 450},
        {"id": "NORTH1", "lat": 41.9, "lon": -87.0, "hdg": 0, "alt": 35500, "spd": 400},
        {"id": "SOUTH1", "lat": 42.1, "lon": -87.0, "hdg": 180, "alt": 34500, "spd": 400}
    ]
    
    created_aircraft = []
    
    for i, ac in enumerate(aircraft_scenarios):
        success = client.create_aircraft(
            acid=ac["id"],
            lat=ac["lat"],
            lon=ac["lon"],
            hdg=ac["hdg"],
            alt=ac["alt"],
            spd=ac["spd"]
        )
        
        if success:
            created_aircraft.append(ac["id"])
            print(f"   ✅ Created {ac['id']}: {ac['hdg']}° @ {ac['alt']}ft")
        else:
            print(f"   ❌ Failed to create {ac['id']}")
        
        # Stagger creation by 10 seconds
        if i < len(aircraft_scenarios) - 1:
            for _ in range(10):
                client.step_simulation(1.0)
    
    print(f"Created {len(created_aircraft)} aircraft: {created_aircraft}")
    
    # Monitor with both native and custom conflict detection
    print(f"\n📊 Monitoring with Native and Custom Conflict Detection...")
    print(f"   Duration: 120 seconds")
    print(f"   Native CD: BlueSky's built-in ConflictDetection")
    print(f"   Custom CD: Distance-based detection")
    
    native_conflicts_detected = 0
    custom_conflicts_detected = 0
    
    for step in range(120):
        current_time = step
        
        # Step simulation
        client.step_simulation(1.0)
        
        # Get native conflict detection results
        native_summary = client.get_conflict_summary()
        native_conflicts = client.get_bluesky_conflicts()
        
        # Custom conflict detection (distance-based)
        custom_conflicts = []
        aircraft_states = client.get_all_aircraft_states()
        
        for i in range(len(aircraft_states)):
            for j in range(i + 1, len(aircraft_states)):
                ac1, ac2 = aircraft_states[i], aircraft_states[j]
                
                # Calculate distance using BlueSky's function
                from bluesky.tools.geo import kwikqdrdist
                qdr, dist = kwikqdrdist(ac1.lat, ac1.lon, ac2.lat, ac2.lon)
                alt_diff = abs(ac1.alt - ac2.alt)
                
                # Check for conflict (< 5 NM, < 1000 ft)
                if dist < 5.0 and alt_diff < 1000:
                    custom_conflicts.append((ac1.id, ac2.id, dist, alt_diff))
        
        # Count conflicts
        if native_summary['active_conflicts'] > 0:
            native_conflicts_detected += 1
            
        if len(custom_conflicts) > 0:
            custom_conflicts_detected += 1
        
        # Report conflicts every 20 seconds
        if step % 20 == 0:
            print(f"\n   ⏰ {current_time}s Status:")
            print(f"      Native CD - Active: {native_summary['active_conflicts']}, "
                  f"Unique: {native_summary['unique_conflicts']}, "
                  f"Total: {native_summary['total_conflicts_detected']}")
            print(f"      Custom CD - Active: {len(custom_conflicts)}")
            
            # Show aircraft positions
            print(f"      Aircraft positions:")
            for state in aircraft_states:
                print(f"        {state.id}: {state.lat:.4f}, {state.lon:.4f}, "
                      f"{state.alt:.0f}ft, {state.hdg:.1f}°")
        
        # Report conflicts when they occur
        if native_summary['active_conflicts'] > 0 and step % 5 == 0:
            if native_conflicts and native_conflicts['confpairs']:
                pairs = [f"{pair[0]}-{pair[1]}" for pair in native_conflicts['confpairs'][::2]]  # Skip duplicates
                print(f"      🚨 Native conflicts at {current_time}s: {pairs}")
        
        if len(custom_conflicts) > 0 and step % 5 == 0:
            pairs = [f"{c[0]}-{c[1]} ({c[2]:.2f}NM)" for c in custom_conflicts]
            print(f"      🚨 Custom conflicts at {current_time}s: {pairs}")
    
    # Final results
    print(f"\n📈 Final Results Summary:")
    final_summary = client.get_conflict_summary()
    
    print(f"   🛡️  Native ConflictDetection:")
    print(f"      Total conflicts detected: {final_summary['total_conflicts_detected']}")
    print(f"      Total LoS detected: {final_summary['total_los_detected']}")
    print(f"      Steps with conflicts: {native_conflicts_detected}")
    
    print(f"   🎯 Custom ConflictDetection:")
    print(f"      Steps with conflicts: {custom_conflicts_detected}")
    
    print(f"   ✈️  Aircraft Status:")
    final_states = client.get_all_aircraft_states()
    for state in final_states:
        print(f"      {state.id}: {state.lat:.4f}, {state.lon:.4f}, "
              f"{state.alt:.0f}ft, {state.tas:.1f}kt")
    
    # Comparison
    print(f"\n🔍 Detection Comparison:")
    print(f"   Native CD detected conflicts in {native_conflicts_detected}/120 steps ({native_conflicts_detected/120*100:.1f}%)")
    print(f"   Custom CD detected conflicts in {custom_conflicts_detected}/120 steps ({custom_conflicts_detected/120*100:.1f}%)")
    
    success = (len(created_aircraft) >= 3 and 
              final_summary['total_conflicts_detected'] > 0 and
              custom_conflicts_detected > 0)
    
    return success


def demonstrate_trajectory_tracking():
    """Demonstrate trajectory tracking with conflict zones"""
    print(f"\n🛩️  Trajectory Tracking with Conflict Zones")
    print("=" * 50)
    
    client = SimpleBlueSkyClient()
    client.initialize()
    client.reset()
    client.configure_conflict_detection()
    
    # Create simple two-aircraft scenario
    client.create_aircraft("TRK1", lat=42.0, lon=-87.0, hdg=90, alt=35000, spd=400)
    client.create_aircraft("TRK2", lat=42.0, lon=-86.5, hdg=270, alt=35000, spd=400)
    
    trajectory_data = {"TRK1": [], "TRK2": []}
    conflict_zones = []
    
    for step in range(60):  # 1 minute
        client.step_simulation(1.0)
        
        # Record trajectories
        for acid in ["TRK1", "TRK2"]:
            state = client.get_aircraft_state(acid)
            if state:
                trajectory_data[acid].append({
                    "time": step,
                    "lat": state.lat,
                    "lon": state.lon,
                    "alt": state.alt,
                    "speed": state.tas
                })
        
        # Record conflict zones
        summary = client.get_conflict_summary()
        if summary['active_conflicts'] > 0:
            conflict_zones.append({
                "time": step,
                "conflicts": summary['active_conflicts'],
                "aircraft_positions": {
                    acid: (trajectory_data[acid][-1]["lat"], trajectory_data[acid][-1]["lon"])
                    for acid in ["TRK1", "TRK2"] if trajectory_data[acid]
                }
            })
    
    print(f"   📊 Trajectory Summary:")
    for acid in ["TRK1", "TRK2"]:
        traj = trajectory_data[acid]
        if traj:
            print(f"      {acid}: {len(traj)} points")
            print(f"         Start: {traj[0]['lat']:.4f}, {traj[0]['lon']:.4f}")
            print(f"         End:   {traj[-1]['lat']:.4f}, {traj[-1]['lon']:.4f}")
    
    print(f"   🚨 Conflict Zones: {len(conflict_zones)} detected")
    if conflict_zones:
        print(f"      First conflict at: {conflict_zones[0]['time']}s")
        print(f"      Last conflict at: {conflict_zones[-1]['time']}s")
    
    return len(conflict_zones) > 0


if __name__ == "__main__":
    print("🚀 Enhanced BlueSky Client Integration Test")
    print("=" * 70)
    
    # Test enhanced client
    success1 = test_enhanced_bluesky_client()
    
    # Test trajectory tracking
    success2 = demonstrate_trajectory_tracking()
    
    if success1 and success2:
        print("\n✅ ALL TESTS PASSED")
        print("   ✅ Native conflict detection working")
        print("   ✅ Multiple aircraft support working")
        print("   ✅ Trajectory tracking working")
        print("   ✅ Custom + Native CD comparison working")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    sys.exit(0 if (success1 and success2) else 1)
