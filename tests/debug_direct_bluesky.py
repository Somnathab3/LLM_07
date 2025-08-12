#!/usr/bin/env python3
"""
Debug BlueSky Aircraft Creation

Testing the basic aircraft creation commands directly
"""

import sys
sys.path.append('.')

import bluesky as bs
from src.cdr.simulation.bluesky_client import BlueSkyClient


def test_direct_bluesky():
    """Test BlueSky commands directly"""
    print("🔍 Testing BlueSky Commands Directly")
    print("=" * 50)
    
    # Initialize BlueSky
    bs.init(mode='sim', detached=True)
    bs.sim.reset()
    
    print("✅ BlueSky initialized directly")
    
    # Test direct command
    print("\n📝 Testing direct CRE command...")
    cmd = "CRE TEST001 B738 42.0 -87.0 90 35000 450"
    print(f"Command: {cmd}")
    
    result = bs.stack.stack(cmd)
    print(f"Command result: {result}")
    print(f"Command result type: {type(result)}")
    
    # Check traffic
    print(f"\nTraffic count: {len(bs.traf.id)}")
    print(f"Aircraft IDs: {list(bs.traf.id)}")
    
    if len(bs.traf.id) > 0:
        idx = 0
        print(f"Aircraft 0: ID={bs.traf.id[idx]}, LAT={bs.traf.lat[idx]}, LON={bs.traf.lon[idx]}")
    
    return len(bs.traf.id) > 0


def test_simplified_client():
    """Test simplified client"""
    print("\n🔍 Testing Simplified Client")
    print("=" * 50)
    
    client = BlueSkyClient()
    client.initialize()
    client.reset()
    
    print("✅ Client initialized")
    
    # Test aircraft creation
    print("\n📝 Testing client aircraft creation...")
    success = client.create_aircraft(
        acid="TEST002",
        lat=42.0,
        lon=-87.0,
        hdg=90,
        alt=35000,
        spd=450
    )
    
    print(f"Create aircraft result: {success}")
    
    # Check state
    state = client.get_aircraft_state("TEST002")
    print(f"Aircraft state: {state}")
    
    all_states = client.get_all_aircraft_states()
    print(f"All aircraft count: {len(all_states)}")
    
    return state is not None


if __name__ == "__main__":
    print("🚁 BlueSky Direct Testing")
    print("=" * 60)
    
    # Test direct BlueSky
    direct_success = test_direct_bluesky()
    
    # Test simplified client
    client_success = test_simplified_client()
    
    print(f"\n📊 Results:")
    print(f"  Direct BlueSky: {'✅' if direct_success else '❌'}")
    print(f"  Simplified Client: {'✅' if client_success else '❌'}")
    
    if direct_success or client_success:
        print("\n✅ Aircraft creation working!")
    else:
        print("\n❌ Aircraft creation failing!")
