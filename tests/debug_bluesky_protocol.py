#!/usr/bin/env python3
"""Debug BlueSky communication protocol"""

import socket
import time
import subprocess
import threading
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


def start_bluesky():
    """Start BlueSky and return the process"""
    process = subprocess.Popen(
        ["python", "-m", "bluesky", "--headless"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return process


def debug_bluesky_protocol():
    """Debug what BlueSky actually sends/receives"""
    print("🔍 Debugging BlueSky Communication Protocol")
    print("=" * 45)
    
    # Start BlueSky
    print("Starting BlueSky...")
    process = start_bluesky()
    
    try:
        # Wait for BlueSky to start
        time.sleep(3)
        
        print("Connecting to BlueSky on port 11000...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(("127.0.0.1", 11000))
        
        print("✅ Connected successfully!")
        
        # Try to receive initial data
        print("\n1️⃣ Reading initial data from BlueSky...")
        try:
            sock.settimeout(2.0)
            initial_data = sock.recv(1024)
            print(f"   Raw bytes received: {initial_data}")
            print(f"   Length: {len(initial_data)} bytes")
            
            # Try different decoding methods
            try:
                utf8_data = initial_data.decode('utf-8')
                print(f"   UTF-8 decoded: {repr(utf8_data)}")
            except UnicodeDecodeError as e:
                print(f"   UTF-8 decode failed: {e}")
            
            try:
                latin1_data = initial_data.decode('latin-1')
                print(f"   Latin-1 decoded: {repr(latin1_data)}")
            except UnicodeDecodeError as e:
                print(f"   Latin-1 decode failed: {e}")
            
            # Check if it looks like JSON or other structured data
            if initial_data.startswith(b'{') or initial_data.startswith(b'['):
                print("   Looks like JSON data")
            elif initial_data.startswith(b'\x00'):
                print("   Looks like binary protocol with null bytes")
            elif b'\n' in initial_data:
                print("   Contains newlines - might be text-based")
                
        except socket.timeout:
            print("   No initial data received (timeout)")
        
        # Try sending a simple command
        print("\n2️⃣ Sending ECHO command...")
        try:
            command = "ECHO test\n"
            sock.send(command.encode('utf-8'))
            print(f"   Sent: {repr(command)}")
            
            # Try to receive response
            sock.settimeout(3.0)
            response = sock.recv(1024)
            print(f"   Raw response: {response}")
            print(f"   Response length: {len(response)} bytes")
            
            # Try decoding response
            try:
                utf8_response = response.decode('utf-8')
                print(f"   UTF-8 response: {repr(utf8_response)}")
            except UnicodeDecodeError as e:
                print(f"   UTF-8 decode failed: {e}")
                # Try partial decode
                try:
                    partial = response.decode('utf-8', errors='ignore')
                    print(f"   Partial UTF-8 (ignore errors): {repr(partial)}")
                except:
                    pass
            
        except Exception as e:
            print(f"   Command test error: {e}")
        
        # Try other common commands
        print("\n3️⃣ Testing other commands...")
        commands = ["IC\n", "TIME\n", "?\n", "HELP\n"]
        
        for cmd in commands:
            try:
                print(f"   Trying: {repr(cmd)}")
                sock.send(cmd.encode('utf-8'))
                time.sleep(0.5)
                
                try:
                    resp = sock.recv(1024)
                    if resp:
                        print(f"     Response: {resp[:100]}..." if len(resp) > 100 else f"     Response: {resp}")
                        # Try to decode
                        try:
                            decoded = resp.decode('utf-8', errors='replace')
                            print(f"     Decoded: {repr(decoded[:100])}")
                        except:
                            pass
                except socket.timeout:
                    print("     No response")
            except Exception as e:
                print(f"     Error: {e}")
        
        sock.close()
        print("\n✅ Protocol debugging complete")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            if process and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        except:
            pass


def test_binary_protocol():
    """Test if BlueSky uses a binary message protocol"""
    print("\n🔍 Testing Binary Protocol Hypothesis")
    print("=" * 35)
    
    # Start BlueSky
    process = start_bluesky()
    
    try:
        time.sleep(3)
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(("127.0.0.1", 11000))
        
        # Read all available data
        sock.settimeout(1.0)
        all_data = b''
        
        try:
            while True:
                chunk = sock.recv(1024)
                if not chunk:
                    break
                all_data += chunk
        except socket.timeout:
            pass
        
        print(f"Total data received: {len(all_data)} bytes")
        
        if all_data:
            # Analyze the data structure
            print("\nData analysis:")
            print(f"  First 32 bytes: {all_data[:32]}")
            print(f"  Last 32 bytes: {all_data[-32:]}")
            
            # Look for patterns
            null_count = all_data.count(b'\x00')
            newline_count = all_data.count(b'\n')
            json_start = all_data.count(b'{')
            
            print(f"  Null bytes: {null_count}")
            print(f"  Newlines: {newline_count}")
            print(f"  JSON starts: {json_start}")
            
            # Check if it's a length-prefixed protocol
            if len(all_data) >= 4:
                # Try interpreting first 4 bytes as length (big-endian)
                import struct
                try:
                    length_be = struct.unpack('>I', all_data[:4])[0]
                    length_le = struct.unpack('<I', all_data[:4])[0]
                    print(f"  Possible length (big-endian): {length_be}")
                    print(f"  Possible length (little-endian): {length_le}")
                    
                    if length_be < len(all_data) and length_be > 0:
                        print(f"  Big-endian length makes sense!")
                        payload = all_data[4:4+length_be]
                        print(f"  Payload: {payload}")
                        try:
                            decoded_payload = payload.decode('utf-8')
                            print(f"  Decoded payload: {repr(decoded_payload)}")
                        except:
                            pass
                            
                except Exception as e:
                    print(f"  Length analysis failed: {e}")
        
        sock.close()
        
    except Exception as e:
        print(f"❌ Binary test failed: {e}")
    
    finally:
        try:
            if process and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        except:
            pass


if __name__ == "__main__":
    debug_bluesky_protocol()
    test_binary_protocol()
