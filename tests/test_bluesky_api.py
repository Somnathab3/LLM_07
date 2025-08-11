#!/usr/bin/env python3
"""Test BlueSky internal API approach"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


def test_bluesky_internal_api():
    """Test if we can use BlueSky's internal API directly"""
    print("🔍 Testing BlueSky Internal API Approach")
    print("=" * 40)
    
    try:
        # Try to import BlueSky stack directly
        from bluesky import stack
        print("✅ BlueSky stack imported successfully")
        
        # Check available stack functions
        stack_functions = [func for func in dir(stack) if not func.startswith('_')]
        print(f"Available stack functions: {stack_functions}")
        
        # Try to initialize stack
        try:
            stack.init()
            print("✅ Stack initialized")
            
            # Try to process a command
            result = stack.process("ECHO test")
            print(f"ECHO command result: {result}")
            
        except Exception as e:
            print(f"❌ Stack initialization failed: {e}")
            
    except ImportError as e:
        print(f"❌ Cannot import BlueSky stack: {e}")
    
    # Try simulation approach
    try:
        print("\n🔍 Testing BlueSky Simulation API")
        from bluesky import simulation
        print("✅ BlueSky simulation imported")
        
        sim_functions = [func for func in dir(simulation) if not func.startswith('_')]
        print(f"Available simulation functions: {sim_functions}")
        
    except ImportError as e:
        print(f"❌ Cannot import BlueSky simulation: {e}")
    
    # Try core approach
    try:
        print("\n🔍 Testing BlueSky Core API")
        from bluesky.core import base
        print("✅ BlueSky core imported")
        
        print(f"Core base functions: {dir(base)}")
        
    except ImportError as e:
        print(f"❌ Cannot import BlueSky core: {e}")


def test_bluesky_console_approach():
    """Test using BlueSky console client approach"""
    print("\n🔍 Testing BlueSky Console Client Approach")
    print("=" * 42)
    
    import subprocess
    import time
    
    # Start BlueSky server
    print("Starting BlueSky server...")
    server_process = subprocess.Popen(
        ["python", "-m", "bluesky", "--headless"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        time.sleep(3)  # Wait for server to start
        
        # Try to start console client
        print("Starting console client...")
        console_process = subprocess.Popen(
            ["python", "-m", "bluesky", "--console", "127.0.0.1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            # Send commands via console
            commands = ["ECHO test", "TIME", "QUIT"]
            
            for cmd in commands:
                print(f"Sending: {cmd}")
                console_process.stdin.write(cmd + '\n')
                console_process.stdin.flush()
                time.sleep(1)
            
            # Get output
            stdout, stderr = console_process.communicate(timeout=5)
            print(f"Console output: {stdout}")
            if stderr:
                print(f"Console errors: {stderr}")
                
        except subprocess.TimeoutExpired:
            console_process.kill()
            print("❌ Console client timed out")
        
    finally:
        # Cleanup
        try:
            if server_process and server_process.poll() is None:
                server_process.terminate()
                server_process.wait(timeout=5)
        except:
            pass


if __name__ == "__main__":
    test_bluesky_internal_api()
    test_bluesky_console_approach()
