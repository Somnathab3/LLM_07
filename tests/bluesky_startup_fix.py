#!/usr/bin/env python3
"""
Fixed BlueSky binary protocol handler
Based on the discovery that port 11000 works with binary responses
"""

import socket
import time
import struct
import sys
import os

class FixedBlueSkyClient:
    """Fixed BlueSky client that handles the binary protocol properly"""
    
    def __init__(self, host="127.0.0.1", port=11000):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to BlueSky with proper binary protocol handling"""
        try:
            print(f"🔌 Connecting to BlueSky at {self.host}:{self.port}")
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)  # Longer timeout
            self.socket.connect((self.host, self.port))
            
            # Read the handshake
            handshake = self.socket.recv(1024)
            print(f"📡 Received handshake: {handshake}")
            
            self.connected = True
            print("✅ Connected successfully")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            if self.socket:
                self.socket.close()
                self.socket = None
            return False
    
    def send_command(self, command: str, wait_response: bool = True) -> bytes:
        """Send command with proper formatting and response handling"""
        if not self.connected:
            raise RuntimeError("Not connected to BlueSky")
        
        try:
            print(f"📤 Sending: '{command}'")
            
            # Try different command formats
            formats_to_try = [
                command + "\n",           # Standard newline
                command + "\r\n",         # Windows newline  
                command + "\x00",         # Null terminator
                command,                  # Raw command
            ]
            
            for cmd_format in formats_to_try:
                try:
                    print(f"   Trying format: {repr(cmd_format)}")
                    
                    # Send command
                    self.socket.send(cmd_format.encode('utf-8'))
                    
                    if wait_response:
                        # Wait for response with timeout
                        self.socket.settimeout(5.0)
                        
                        try:
                            response = self.socket.recv(4096)
                            print(f"📥 Response ({len(response)} bytes): {response[:100]}...")
                            
                            # Check if this looks like a good response
                            if response and len(response) > 0:
                                return response
                                
                        except socket.timeout:
                            print(f"⏰ Timeout waiting for response to format: {repr(cmd_format)}")
                            continue
                            
                except Exception as e:
                    print(f"❌ Error with format {repr(cmd_format)}: {e}")
                    continue
            
            print(f"❌ All command formats failed for: {command}")
            return b""
            
        except Exception as e:
            print(f"❌ Send command error: {e}")
            return b""
    
    def test_basic_commands(self):
        """Test basic commands to see what works"""
        
        print("\n=== Testing Basic Commands ===")
        
        basic_commands = [
            "HELP",
            "TIME", 
            "RESET",
            "OP",
            "HOLD",
        ]
        
        working_commands = []
        
        for cmd in basic_commands:
            print(f"\n--- Testing: {cmd} ---")
            
            try:
                response = self.send_command(cmd, wait_response=True)
                
                if response and len(response) > 0:
                    print(f"✅ {cmd} works! ({len(response)} bytes)")
                    working_commands.append(cmd)
                    
                    # Try to decode response
                    try:
                        text = response.decode('utf-8', errors='ignore')
                        if text.strip():
                            print(f"   Text: {text[:100]}...")
                    except:
                        pass
                else:
                    print(f"❌ {cmd} no response")
                    
            except Exception as e:
                print(f"❌ {cmd} error: {e}")
        
        print(f"\n📋 Working commands: {working_commands}")
        return working_commands
    
    def test_aircraft_creation(self):
        """Test different aircraft creation formats"""
        
        print("\n=== Testing Aircraft Creation Formats ===")
        
        # Different formats to try based on BlueSky documentation
        creation_formats = [
            # Standard format
            "CRE TEST01,B738,42.0,-87.0,90,35000,450",
            
            # Space-separated  
            "CRE TEST01 B738 42.0 -87.0 90 35000 450",
            
            # Mixed format
            "CREATE TEST01,B738,42.0,-87.0,90,35000,450",
            
            # Step-by-step approach
            "CRE TEST01,B738,42.0,-87.0,0,0,0",  # Minimal creation
        ]
        
        for fmt in creation_formats:
            print(f"\n--- Testing creation format ---")
            print(f"Format: {fmt}")
            
            try:
                # First send reset to clear any previous state
                self.send_command("RESET", wait_response=True)
                time.sleep(1)
                
                # Try the creation command with longer timeout
                print("Sending creation command...")
                self.socket.settimeout(10.0)  # Longer timeout for creation
                
                response = self.send_command(fmt, wait_response=True)
                
                if response:
                    print(f"✅ Creation response: {response[:200]}")
                    
                    # Check if aircraft was created by trying to query it
                    time.sleep(1)
                    print("Testing if aircraft exists...")
                    
                    test_commands = ["LISTAC", "POS TEST01", "HDG TEST01"]
                    for test_cmd in test_commands:
                        test_resp = self.send_command(test_cmd, wait_response=True)
                        if test_resp:
                            print(f"   {test_cmd}: {test_resp[:50]}...")
                
                else:
                    print(f"❌ No response to creation command")
                    
            except Exception as e:
                print(f"❌ Creation test error: {e}")
    
    def interactive_test(self):
        """Interactive command testing"""
        
        print("\n=== Interactive Command Test ===")
        print("Enter commands to test (or 'quit' to exit):")
        
        while True:
            try:
                cmd = input("BlueSky> ").strip()
                
                if cmd.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not cmd:
                    continue
                
                print(f"Sending: {cmd}")
                response = self.send_command(cmd, wait_response=True)
                
                if response:
                    print(f"Response ({len(response)} bytes):")
                    
                    # Try to decode as text
                    try:
                        text = response.decode('utf-8', errors='ignore')
                        print(f"  Text: {text}")
                    except:
                        print(f"  Binary: {response}")
                else:
                    print("No response")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def disconnect(self):
        """Disconnect from BlueSky"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        self.connected = False
        print("✅ Disconnected")

def test_fixed_protocol():
    """Test the fixed protocol implementation"""
    
    print("=== Testing Fixed BlueSky Protocol ===")
    
    # Start BlueSky first
    import subprocess
    print("🚀 Starting BlueSky...")
    
    process = subprocess.Popen(
        ["python", "-m", "bluesky", "--headless"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for startup
    time.sleep(3)
    
    try:
        # Connect with fixed client
        client = FixedBlueSkyClient()
        
        if not client.connect():
            print("❌ Failed to connect")
            return
        
        # Test basic commands first
        working_commands = client.test_basic_commands()
        
        if working_commands:
            print(f"✅ Basic commands work!")
            
            # Test aircraft creation
            client.test_aircraft_creation()
            
            # Optional: interactive testing
            print("\n" + "="*50)
            response = input("Run interactive test? (y/n): ")
            if response.lower().startswith('y'):
                client.interactive_test()
        
        else:
            print("❌ No basic commands working")
        
        client.disconnect()
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up process
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()

if __name__ == "__main__":
    test_fixed_protocol()