#!/usr/bin/env python3
"""
Quick test to see which BlueSky commands actually work
This is the RIGHT way to communicate with a separate BlueSky process
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.cdr.simulation.bluesky_client import BlueSkyClient, create_thesis_config

def quick_test():
    print("=== Quick BlueSky Command Test ===")
    print("(This tests the RIGHT way to communicate with BlueSky)")
    
    bluesky_config = create_thesis_config()
    bluesky_client = BlueSkyClient(bluesky_config)
    
    if not bluesky_client.connect():
        print("❌ Failed to connect")
        return
    
    try:
        # Reset and create aircraft
        print("\n🔄 Setting up test...")
        bluesky_client._send_command("RESET")
        time.sleep(2)
        
        print("🔄 Creating aircraft...")
        success = bluesky_client.create_aircraft("TEST01", "B738", 42.0, -87.0, 90, 35000, 450)
        print(f"Aircraft created: {success}")
        
        bluesky_client.op()
        time.sleep(3)
        
        print("\n=== Testing Individual Commands ===")
        
        # Test commands that should work based on BlueSky documentation
        test_commands = [
            # Basic commands
            ("HELP", "Show help"),
            
            # List commands  
            ("LISTAC", "List aircraft"),
            
            # Individual property queries
            ("LAT TEST01", "Get latitude"),
            ("LON TEST01", "Get longitude"), 
            ("ALT TEST01", "Get altitude"),
            ("HDG TEST01", "Get heading"),
            ("SPD TEST01", "Get speed"),
            
            # Alternative syntax
            ("TEST01 LAT", "Get latitude (alt syntax)"),
            ("TEST01 LON", "Get longitude (alt syntax)"),
            
            # Info commands
            ("INFO TEST01", "Get aircraft info"),
            
            # The problematic command
            ("POS TEST01", "Position command (problematic)"),
        ]
        
        working_commands = []
        
        for cmd, description in test_commands:
            try:
                print(f"\n--- Testing: {cmd} ---")
                response = bluesky_client._send_command(cmd, expect_response=True, timeout=4.0)
                
                if response and len(response) > 5:
                    print(f"✅ SUCCESS: {len(response)} chars")
                    print(f"   Response: {response[:100]}...")
                    
                    # Check if response contains useful data
                    if "ERROR" not in response.upper() and "FAIL" not in response.upper():
                        working_commands.append(cmd)
                        
                        # Try to extract numbers from response
                        import re
                        numbers = re.findall(r'-?\d+\.?\d*', response)
                        if numbers:
                            print(f"   📊 Numbers found: {numbers[:5]}...")
                    else:
                        print(f"   ⚠️ Contains error: {response[:50]}...")
                else:
                    print(f"❌ No response or too short: '{response}'")
                    
            except Exception as e:
                print(f"❌ Command failed: {e}")
        
        print(f"\n📋 Summary:")
        print(f"   Working commands: {len(working_commands)}")
        print(f"   Commands: {working_commands}")
        
        # If we found working commands, test aircraft movement
        if working_commands:
            print(f"\n=== Testing Aircraft Movement ===")
            
            # Try to get initial position
            if "LAT TEST01" in working_commands:
                lat_response = bluesky_client._send_command("LAT TEST01", expect_response=True)
                print(f"Initial LAT: {lat_response}")
            
            # Send movement commands
            print("🔄 Sending heading command...")
            hdg_response = bluesky_client._send_command("HDG TEST01,135", expect_response=True)
            print(f"HDG command response: {hdg_response}")
            
            print("🔄 Sending altitude command...")
            alt_response = bluesky_client._send_command("ALT TEST01,37000", expect_response=True)  
            print(f"ALT command response: {alt_response}")
            
            # Wait and check again
            print("⏳ Waiting 10 seconds...")
            time.sleep(10)
            
            if "LAT TEST01" in working_commands:
                lat_response2 = bluesky_client._send_command("LAT TEST01", expect_response=True)
                print(f"Final LAT: {lat_response2}")
                
            if "HDG TEST01" in working_commands:
                hdg_response2 = bluesky_client._send_command("HDG TEST01", expect_response=True)
                print(f"Final HDG: {hdg_response2}")
        
        else:
            print("\n❌ No working commands found - there may be a deeper issue")
            
            # Debug the connection
            print("\n=== Connection Debug ===")
            try:
                # Try the most basic command
                basic_response = bluesky_client._send_command("OP", expect_response=True)
                print(f"OP command: {basic_response}")
                
                # Try checking simulation state
                time_response = bluesky_client._send_command("TIME", expect_response=True) 
                print(f"TIME command: {time_response}")
                
            except Exception as e:
                print(f"Even basic commands fail: {e}")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        bluesky_client.disconnect()
        print("\n✅ Test completed")

if __name__ == "__main__":
    quick_test()
