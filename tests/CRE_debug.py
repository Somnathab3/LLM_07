#!/usr/bin/env python3
"""
Quick debug for the CRE command hanging issue
"""

import socket
import time
import threading

def test_cre_command_debug():
    """Debug the CRE command hanging issue"""
    
    print("=== Quick CRE Command Debug ===")
    
    # Start BlueSky
    import subprocess
    process = subprocess.Popen(["python", "-m", "bluesky", "--headless"])
    time.sleep(3)
    
    try:
        # Connect
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", 11000))
        
        # Get handshake
        handshake = sock.recv(1024)
        print(f"Handshake: {handshake}")
        
        # Test commands step by step
        commands_to_test = [
            ("HELP", "Basic help command"),
            ("TIME", "Time command"), 
            ("RESET", "Reset simulation"),
            ("OP", "Start operation"),
            ("CRE TEST01,B738,42,-87,90,35000,450", "Create aircraft - full format"),
            ("CRE TEST01,B738,42.0,-87.0,90,35000,450", "Create aircraft - with decimals"),
            ("CRE TEST01 B738 42 -87 90 35000 450", "Create aircraft - space format"),
        ]
        
        for cmd, description in commands_to_test:
            print(f"\n--- Testing: {description} ---")
            print(f"Command: {cmd}")
            
            try:
                # Set a short timeout to avoid hanging
                sock.settimeout(3.0)
                
                # Send command
                sock.send((cmd + "\n").encode())
                print("✅ Command sent")
                
                # Try to read response
                try:
                    response = sock.recv(2048)
                    print(f"✅ Response: {len(response)} bytes")
                    print(f"   Content: {response[:100]}")
                    
                except socket.timeout:
                    print("⏰ Timeout - command may have hung")
                    # Try to continue anyway
                    
                except Exception as e:
                    print(f"❌ Response error: {e}")
                
                # Small delay between commands
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Command error: {e}")
                break
        
        sock.close()
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
        
    finally:
        process.terminate()

def test_alternative_approach():
    """Test using your working client but with better command handling"""
    
    print("\n=== Testing with Your Working Client ===")
    
    # Import your client
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.cdr.simulation.bluesky_client import BlueSkyClient, create_thesis_config
    
    try:
        config = create_thesis_config()
        client = BlueSkyClient(config)
        
        if not client.connect():
            print("❌ Failed to connect")
            return
        
        print("✅ Connected with your client")
        
        # Try sending commands one by one with delays
        print("\n--- Testing individual commands ---")
        
        # Reset first
        print("1. Sending RESET...")
        response = client._send_command("RESET", expect_response=True, timeout=5.0)
        print(f"   RESET response: {response[:100] if response else 'None'}...")
        time.sleep(2)
        
        # Try different creation formats
        creation_attempts = [
            "CRE AC01,B738,42,87,90,35000,450",  # Simple format
            "CRE AC02,A320,43,-86,180,37000,480",  # Another aircraft
        ]
        
        for i, cmd in enumerate(creation_attempts, 1):
            print(f"{i+1}. Sending: {cmd}")
            try:
                # Send with shorter timeout to avoid hanging
                response = client._send_command(cmd, expect_response=True, timeout=3.0)
                print(f"   Response: {response[:100] if response else 'None'}...")
                time.sleep(1)
                
                # If that worked, try to start simulation
                if response and "ERROR" not in str(response).upper():
                    print(f"   ✅ Aircraft creation seems successful")
                    
                    # Start simulation
                    op_response = client._send_command("OP", expect_response=True, timeout=2.0)
                    print(f"   OP response: {op_response[:50] if op_response else 'None'}...")
                    
                    # Try to query the aircraft
                    time.sleep(1)
                    query_response = client._send_command("POS AC01", expect_response=True, timeout=2.0)
                    print(f"   Query response: {query_response[:100] if query_response else 'None'}...")
                    
                else:
                    print(f"   ❌ Creation may have failed")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        client.disconnect()
        
    except Exception as e:
        print(f"❌ Client test error: {e}")

if __name__ == "__main__":
    test_cre_command_debug()
    test_alternative_approach()