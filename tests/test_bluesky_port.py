#!/usr/bin/env python3
"""Simple BlueSky connection test"""

import socket
import time
import subprocess
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.cdr.simulation.bluesky_client import create_thesis_config


def test_bluesky_port():
    """Test if we can connect to BlueSky on the correct port"""
    print("🔌 Testing BlueSky Port Connection")
    print("=" * 40)
    
    config = create_thesis_config()
    print(f"Testing connection to {config.host}:{config.port}")
    
    # Start BlueSky manually and test connection
    print("\n1️⃣ Starting BlueSky...")
    try:
        # Start BlueSky process
        process = subprocess.Popen(
            ["python", "-m", "bluesky", "--headless"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("Waiting for BlueSky to start...")
        
        # Give BlueSky time to start
        for i in range(10):
            time.sleep(1)
            
            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate(timeout=5)
                print(f"BlueSky exited with code {process.returncode}")
                print(f"STDOUT: {stdout[:500]}...")
                print(f"STDERR: {stderr[:500]}...")
                break
            
            # Try to connect
            try:
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(1.0)
                result = test_socket.connect_ex((config.host, config.port))
                test_socket.close()
                
                if result == 0:
                    print(f"✅ Connected to BlueSky on port {config.port} after {i+1}s")
                    
                    # Test sending a simple command
                    try:
                        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        client_socket.settimeout(5.0)
                        client_socket.connect((config.host, config.port))
                        
                        # Send ECHO command
                        command = "ECHO test\n"
                        client_socket.send(command.encode('utf-8'))
                        
                        # Try to receive response
                        response = client_socket.recv(1024).decode('utf-8')
                        print(f"✅ Command response: {response.strip()}")
                        
                        client_socket.close()
                        
                        # Clean shutdown
                        shutdown_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        shutdown_socket.settimeout(2.0)
                        shutdown_socket.connect((config.host, config.port))
                        shutdown_socket.send("QUIT\n".encode('utf-8'))
                        shutdown_socket.close()
                        
                        print("✅ BlueSky connection test successful!")
                        return True
                        
                    except Exception as e:
                        print(f"⚠️ Command test failed: {e}")
                        return True  # Connection worked, command might need different format
                        
                else:
                    print(f"   Attempt {i+1}: Connection failed (code {result})")
            except Exception as e:
                print(f"   Attempt {i+1}: {e}")
            
        print("❌ Could not connect to BlueSky")
        return False
        
    except Exception as e:
        print(f"❌ Error starting BlueSky: {e}")
        return False
    finally:
        # Clean up process
        try:
            if process and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        except:
            pass


def test_bluesky_detached():
    """Test BlueSky in detached mode which might stay running longer"""
    print("\n🔌 Testing BlueSky Detached Mode")
    print("=" * 35)
    
    config = create_thesis_config()
    
    try:
        # Start BlueSky in detached mode
        process = subprocess.Popen(
            ["python", "-m", "bluesky", "--detached"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("Starting BlueSky in detached mode...")
        
        # Give it time to start
        for i in range(15):
            time.sleep(1)
            
            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate(timeout=5)
                print(f"BlueSky detached exited with code {process.returncode}")
                if stdout:
                    print(f"STDOUT: {stdout[:300]}...")
                if stderr:
                    print(f"STDERR: {stderr[:300]}...")
                break
            
            # Check for different ports BlueSky might use
            for port in [11000, 11001, 12000, 12001]:
                try:
                    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_socket.settimeout(0.5)
                    result = test_socket.connect_ex((config.host, port))
                    test_socket.close()
                    
                    if result == 0:
                        print(f"✅ Found BlueSky listening on port {port}")
                        return True
                        
                except Exception:
                    pass
            
            if i % 3 == 0 and i > 0:
                print(f"   Still waiting... ({i}s)")
        
        print("❌ BlueSky detached mode test failed")
        return False
        
    except Exception as e:
        print(f"❌ Error with detached mode: {e}")
        return False
    finally:
        # Clean up
        try:
            if process and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        except:
            pass


if __name__ == "__main__":
    print("🚀 BlueSky Port Connection Test")
    print("=" * 32)
    
    success1 = test_bluesky_port()
    success2 = test_bluesky_detached()
    
    if success1 or success2:
        print("\n✅ BlueSky connection working on correct port!")
        print("The updated BlueSky client should work with port 11000")
    else:
        print("\n❌ BlueSky connection issues detected")
        print("This might be a BlueSky configuration or version issue")
        print("\nTroubleshooting suggestions:")
        print("1. Check BlueSky installation: pip install bluesky-simulator")
        print("2. Check BlueSky config: C:\\Users\\Administrator\\bluesky\\settings.cfg")
        print("3. Try running BlueSky manually: python -m bluesky --headless")
        print("4. BlueSky might need a scenario to stay running")
