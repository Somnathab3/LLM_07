#!/usr/bin/env python3
"""
BlueSky Conflict Detection Test

This test checks:
1. If BlueSky's built-in conflict detection is active
2. Multiple aircraft creation at different intervals
3. Trajectory tracking and conflict detection
4. ASAS (Automatic Separation Assurance System) functionality
"""

import sys
import time
import math
import logging
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
sys.path.append('.')

import bluesky as bs
from src.cdr.simulation.bluesky_client import BlueSkyClient, AircraftState


def setup_logging():
    """Setup detailed logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in nautical miles"""
    # Using BlueSky's built-in distance calculation
    from bluesky.tools.geo import kwikqdrdist
    qdr, dist = kwikqdrdist(lat1, lon1, lat2, lon2)
    return dist


def create_conflict_scenario(client: BlueSkyClient) -> List[str]:
    """Create aircraft on collision course to test conflict detection"""
    logger = logging.getLogger(__name__)
    
    # Aircraft configurations for collision course
    aircraft_configs = [
        {
            "id": "AC001",
            "lat": 42.0,     # Start position
            "lon": -87.0,
            "hdg": 90,       # Eastbound
            "alt": 35000,
            "spd": 400
        },
        {
            "id": "AC002", 
            "lat": 42.0,     # Same latitude (collision course)
            "lon": -86.5,    # 0.5 degrees east
            "hdg": 270,      # Westbound (head-on)
            "alt": 35000,    # Same altitude
            "spd": 400
        },
        {
            "id": "AC003",
            "lat": 41.95,    # Slightly south
            "lon": -86.8,
            "hdg": 45,       # Northeast
            "alt": 35000,
            "spd": 350
        }
    ]
    
    created_aircraft = []
    
    for i, config in enumerate(aircraft_configs):
        logger.info(f"Creating aircraft {config['id']} at interval {i}")
        
        success = client.create_aircraft(
            acid=config["id"],
            lat=config["lat"],
            lon=config["lon"],
            hdg=config["hdg"],
            alt=config["alt"],
            spd=config["spd"]
        )
        
        if success:
            created_aircraft.append(config["id"])
            logger.info(f"✅ Created {config['id']} - {config['hdg']}° heading")
            
            # Check immediate state
            state = client.get_aircraft_state(config["id"])
            if state:
                logger.info(f"   Initial: {state.lat:.4f}, {state.lon:.4f}, "
                           f"{state.alt}ft, {state.hdg}°, {state.tas:.1f}kt")
        else:
            logger.error(f"❌ Failed to create {config['id']}")
        
        # Stagger aircraft creation by 30 seconds
        if i < len(aircraft_configs) - 1:
            logger.info(f"Stepping simulation 30s before next aircraft...")
            for _ in range(30):
                client.step_simulation(1.0)
    
    return created_aircraft


def check_bluesky_conflict_detection() -> bool:
    """Check if BlueSky's built-in conflict detection is active"""
    logger = logging.getLogger(__name__)
    
    # Check ASAS (Automatic Separation Assurance System)
    try:
        asas_active = hasattr(bs, 'traf') and hasattr(bs.traf, 'asas')
        if asas_active:
            logger.info(f"✅ ASAS (conflict detection) available: {bs.traf.asas}")
            
            # Check conflict detection parameters
            if hasattr(bs.traf.asas, 'dtlook'):
                logger.info(f"   Look-ahead time: {bs.traf.asas.dtlook}s")
            if hasattr(bs.traf.asas, 'dtconf'):
                logger.info(f"   Conflict time: {bs.traf.asas.dtconf}s")
            if hasattr(bs.traf.asas, 'rpz'):
                logger.info(f"   Protected zone radius: {bs.traf.asas.rpz} NM")
                
        return asas_active
    except Exception as e:
        logger.error(f"Error checking ASAS: {e}")
        return False


def monitor_conflicts_and_trajectories(client: BlueSkyClient, aircraft_ids: List[str], 
                                     duration_minutes: int = 10) -> Dict:
    """Monitor aircraft for conflicts and track trajectories"""
    logger = logging.getLogger(__name__)
    
    trajectories = {acid: {"times": [], "positions": [], "conflicts": []} 
                   for acid in aircraft_ids}
    
    total_steps = duration_minutes * 60  # Convert to seconds
    conflict_count = 0
    min_separation = float('inf')
    min_separation_time = 0
    
    logger.info(f"📊 Starting {duration_minutes}-minute monitoring...")
    logger.info(f"   Tracking {len(aircraft_ids)} aircraft: {aircraft_ids}")
    
    for step in range(total_steps):
        current_time = step  # seconds
        
        # Step simulation
        client.step_simulation(1.0)
        
        # Get all aircraft states
        aircraft_states = {}
        for acid in aircraft_ids:
            state = client.get_aircraft_state(acid)
            if state:
                aircraft_states[acid] = state
                
                # Record trajectory
                trajectories[acid]["times"].append(current_time)
                trajectories[acid]["positions"].append((state.lat, state.lon, state.alt))
        
        # Check for conflicts (proximity)
        current_conflicts = []
        aircraft_list = list(aircraft_states.items())
        
        for i in range(len(aircraft_list)):
            for j in range(i + 1, len(aircraft_list)):
                acid1, state1 = aircraft_list[i]
                acid2, state2 = aircraft_list[j]
                
                # Calculate horizontal distance
                distance = calculate_distance(state1.lat, state1.lon, state2.lat, state2.lon)
                
                # Calculate vertical separation
                alt_diff = abs(state1.alt - state2.alt)
                
                # Track minimum separation
                if distance < min_separation:
                    min_separation = distance
                    min_separation_time = current_time
                
                # Check for conflict (< 5 NM horizontal, < 1000 ft vertical)
                if distance < 5.0 and alt_diff < 1000:
                    conflict_info = {
                        "time": current_time,
                        "aircraft": [acid1, acid2],
                        "distance": distance,
                        "altitude_diff": alt_diff
                    }
                    current_conflicts.append(conflict_info)
                    
                    # Add to both aircraft's conflict records
                    trajectories[acid1]["conflicts"].append(conflict_info)
                    trajectories[acid2]["conflicts"].append(conflict_info)
        
        # Log conflicts
        if current_conflicts:
            conflict_count += len(current_conflicts)
            for conflict in current_conflicts:
                logger.warning(f"⚠️  CONFLICT at {current_time}s: "
                             f"{conflict['aircraft'][0]} <-> {conflict['aircraft'][1]} "
                             f"({conflict['distance']:.2f} NM, {conflict['altitude_diff']:.0f} ft)")
        
        # Periodic status updates
        if step % 60 == 0:  # Every minute
            logger.info(f"   {current_time/60:.1f}min: {len(aircraft_states)} aircraft active")
            for acid, state in aircraft_states.items():
                logger.info(f"     {acid}: {state.lat:.4f}, {state.lon:.4f}, "
                           f"{state.alt:.0f}ft, {state.hdg:.1f}°, {state.tas:.1f}kt")
    
    # Check BlueSky's internal conflict detection
    bluesky_conflicts = []
    try:
        if hasattr(bs.traf, 'asas') and hasattr(bs.traf.asas, 'confpairs'):
            confpairs = bs.traf.asas.confpairs
            if confpairs is not None and len(confpairs) > 0:
                logger.info(f"🔍 BlueSky detected {len(confpairs)} internal conflicts")
                bluesky_conflicts = confpairs
    except Exception as e:
        logger.error(f"Error accessing BlueSky conflicts: {e}")
    
    return {
        "trajectories": trajectories,
        "total_conflicts": conflict_count,
        "min_separation": min_separation,
        "min_separation_time": min_separation_time,
        "bluesky_conflicts": bluesky_conflicts,
        "aircraft_count": len(aircraft_states)
    }


def test_conflict_detection():
    """Main test function for conflict detection"""
    logger = setup_logging()
    
    print("🔍 BlueSky Conflict Detection and Trajectory Test")
    print("=" * 70)
    
    # Initialize client
    client = BlueSkyClient()
    client.initialize()
    client.reset()
    
    logger.info("✅ BlueSky initialized")
    
    # Check conflict detection capabilities
    print("\n🛡️  Checking BlueSky Conflict Detection Capabilities...")
    asas_available = check_bluesky_conflict_detection()
    
    # Create conflict scenario
    print("\n✈️  Creating Conflict Scenario...")
    aircraft_ids = create_conflict_scenario(client)
    
    if not aircraft_ids:
        logger.error("❌ No aircraft created - cannot proceed with test")
        return False
    
    logger.info(f"Created {len(aircraft_ids)} aircraft: {aircraft_ids}")
    
    # Monitor conflicts and trajectories
    print(f"\n📊 Monitoring Conflicts and Trajectories...")
    results = monitor_conflicts_and_trajectories(client, aircraft_ids, duration_minutes=5)
    
    # Results summary
    print(f"\n📈 Test Results Summary:")
    print(f"   ASAS Available: {'✅' if asas_available else '❌'}")
    print(f"   Aircraft Created: {len(aircraft_ids)}")
    print(f"   Total Conflicts: {results['total_conflicts']}")
    print(f"   Minimum Separation: {results['min_separation']:.2f} NM at {results['min_separation_time']}s")
    print(f"   BlueSky Internal Conflicts: {len(results['bluesky_conflicts'])}")
    
    # Trajectory summary
    print(f"\n🛩️  Trajectory Summary:")
    for acid in aircraft_ids:
        traj = results["trajectories"][acid]
        if traj["positions"]:
            start_pos = traj["positions"][0]
            end_pos = traj["positions"][-1]
            conflicts = len(traj["conflicts"])
            
            print(f"   {acid}: {len(traj['positions'])} positions, {conflicts} conflicts")
            print(f"     Start: {start_pos[0]:.4f}, {start_pos[1]:.4f}, {start_pos[2]:.0f}ft")
            print(f"     End:   {end_pos[0]:.4f}, {end_pos[1]:.4f}, {end_pos[2]:.0f}ft")
    
    success = (len(aircraft_ids) >= 2 and 
              results['total_conflicts'] > 0 and 
              results['min_separation'] < 10.0)  # At least got within 10 NM
    
    return success


if __name__ == "__main__":
    success = test_conflict_detection()
    
    if success:
        print("\n✅ Conflict detection test completed successfully")
        print("   - Multiple aircraft created at intervals")
        print("   - Trajectories tracked")
        print("   - Conflicts detected")
    else:
        print("\n❌ Conflict detection test failed")
    
    sys.exit(0 if success else 1)
