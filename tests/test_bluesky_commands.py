#!/usr/bin/env python3
"""
Simple test to check BlueSky command responses for debugging
"""

from src.cdr.simulation.bluesky_client import BlueSkyClient, create_thesis_config

def test_bluesky_commands():
    """Test individual BlueSky command responses"""
    
    print("🔍 Testing BlueSky command responses...")
    
    # Create BlueSky client
    bluesky_config = create_thesis_config()
    bluesky_client = BlueSkyClient(bluesky_config)
    
    try:
        # Connect to BlueSky
        if not bluesky_client.connect():
            print("❌ Failed to connect to BlueSky")
            return
        
        print("✅ Connected to BlueSky")
        
        # Create a test aircraft with known position
        print("\n📍 Creating test aircraft at known position...")
        success = bluesky_client.create_aircraft(
            callsign="TEST1",
            aircraft_type="B738",
            lat=41.978,    # Chicago latitude
            lon=-87.904,   # Chicago longitude
            heading=270,
            altitude_ft=35000,
            speed_kt=450
        )
        
        if not success:
            print("❌ Failed to create aircraft")
            return
        
        # Start simulation
        bluesky_client.op()
        
        print("\n🔍 Testing individual property commands...")
        
        # Test different command formats
        commands_to_test = [
            "TEST1 LAT",      # Current format we're using
            "TEST1 LON", 
            "TEST1 ALT",
            "TEST1 HDG",
            "TEST1 TAS",
            "POS TEST1",      # Binary position command
            "TEST1",          # Just callsign
            "TEST1 POS",      # Alternative format
            "ECHO TEST1 LAT", # With ECHO prefix
        ]
        
        for cmd in commands_to_test:
            print(f"\n🔧 Testing command: {cmd}")
            try:
                response = bluesky_client._send_command(cmd, expect_response=True, timeout=2.0)
                print(f"   Response: {response[:200]}...")  # Truncate long responses
                
                # Try to extract numbers
                import re
                numbers = re.findall(r'-?\d+\.?\d*', response)
                if numbers:
                    print(f"   Numbers found: {numbers[:5]}")  # Show first 5 numbers
                else:
                    print("   No numbers found")
                    
            except Exception as e:
                print(f"   Error: {e}")
        
        print("\n🔍 Testing POSTEXT or similar commands...")
        
        # Try different position query formats
        pos_commands = [
            "POSTEXT TEST1",
            "LISTAC TEST1",
            "ACINFO TEST1",
            "TEST1 INFO",
            "LNAV TEST1",
        ]
        
        for cmd in pos_commands:
            print(f"\n🔧 Testing position command: {cmd}")
            try:
                response = bluesky_client._send_command(cmd, expect_response=True, timeout=2.0)
                print(f"   Response: {response[:200]}...")
                
                # Check if this looks like position data
                if any(keyword in response.lower() for keyword in ['lat', 'lon', 'latitude', 'longitude', '41.', '-87.']):
                    print("   ✅ This command might contain position data!")
                    
            except Exception as e:
                print(f"   Error: {e}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        bluesky_client.disconnect()
        print("\n🧹 Disconnected from BlueSky")

if __name__ == "__main__":
    test_bluesky_commands()
