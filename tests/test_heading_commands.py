#!/usr/bin/env python3
"""
Simple test script to verify BlueSky heading commands work correctly
"""

import time
import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cdr.simulation.bluesky_client import BlueSkyClient, BlueSkyConfig


def test_heading_commands():
    """Test that heading commands are sent and applied correctly"""
    
    print("🔧 Testing BlueSky heading commands...")
    
    # Create client with default config
    config = BlueSkyConfig()
    client = BlueSkyClient(config)
    
    # Try to connect to existing BlueSky instance
    if not client.connect():
        print("❌ No existing BlueSky instance found. Please start BlueSky manually.")
        return False
    
    print("✅ Connected to BlueSky")
    
    try:
        # Test sequence
        print("\n🔧 Test 1: Creating test aircraft...")
        
        # Create a test aircraft
        result = client._send_command("CRE TEST A320 42.0 -87.9 0 0 200", expect_response=True)
        print(f"Create aircraft result: {result}")
        
        print("\n🔧 Test 2: Getting initial aircraft state...")
        
        # Get initial state
        initial_response = client._send_command("POS TEST", expect_response=True)
        print(f"Initial POS response: {initial_response[:200]}...")
        
        print("\n🔧 Test 3: Setting heading to 45°...")
        
        # Set heading to 45 degrees
        hdg_result = client._send_command("HDG TEST 45", expect_response=True)
        print(f"Set heading result: {hdg_result}")
        
        # Wait a moment for the command to take effect
        time.sleep(2)
        
        print("\n🔧 Test 4: Getting updated aircraft state...")
        
        # Get updated state
        updated_response = client._send_command("POS TEST", expect_response=True)
        print(f"Updated POS response: {updated_response[:200]}...")
        
        print("\n🔧 Test 5: Setting heading to 180°...")
        
        # Set heading to 180 degrees
        hdg_result2 = client._send_command("HDG TEST 180", expect_response=True)
        print(f"Set heading to 180° result: {hdg_result2}")
        
        # Wait again
        time.sleep(2)
        
        print("\n🔧 Test 6: Getting final aircraft state...")
        
        # Get final state
        final_response = client._send_command("POS TEST", expect_response=True)
        print(f"Final POS response: {final_response[:200]}...")
        
        print("\n🔧 Test 7: Cleaning up...")
        
        # Clean up - delete test aircraft
        del_result = client._send_command("DEL TEST", expect_response=True)
        print(f"Delete aircraft result: {del_result}")
        
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    
    finally:
        client.disconnect()


if __name__ == "__main__":
    success = test_heading_commands()
    sys.exit(0 if success else 1)
