#!/usr/bin/env python3
"""Test CLI integration with embedded BlueSky client"""

from src.cdr.simulation.bluesky_client import BlueSkyClient, BlueSkyConfig

def test_get_aircraft_state():
    """Test the get_aircraft_state method works"""
    print("🧪 Testing BlueSky client get_aircraft_state method...")
    
    # Initialize client
    config = BlueSkyConfig()
    client = BlueSkyClient(config)
    
    # Connect to embedded mode
    if not client.connect():
        print("❌ Failed to connect to BlueSky")
        return False
    
    # Create test aircraft
    success = client.create_aircraft(
        callsign="TEST123",
        aircraft_type="B738",
        lat=52.0,
        lon=4.0,
        heading=90,
        altitude_ft=35000,
        speed_kt=450
    )
    
    if not success:
        print("❌ Failed to create aircraft")
        return False
    
    # Step simulation to update state
    client.step_simulation(5)
    
    # Test get_aircraft_state method
    state = client.get_aircraft_state("TEST123")
    
    if state:
        print(f"✅ get_aircraft_state works!")
        print(f"   Aircraft: {state.callsign}")
        print(f"   Position: {state.latitude:.6f}, {state.longitude:.6f}")
        print(f"   Altitude: {state.altitude_ft:.1f} ft")
        print(f"   Heading: {state.heading_deg:.1f}°")
        print(f"   Speed: {state.speed_kt:.1f} kt")
        return True
    else:
        print("❌ get_aircraft_state returned None")
        return False

if __name__ == "__main__":
    success = test_get_aircraft_state()
    print(f"\n{'✅ Test PASSED' if success else '❌ Test FAILED'}")
