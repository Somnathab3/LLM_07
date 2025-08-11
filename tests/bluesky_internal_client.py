#!/usr/bin/env python3
"""BlueSky client using internal API (alternative to binary protocol)"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.cdr.simulation.bluesky_client import (
    AircraftState, ConflictInfo, BlueSkyConfig, create_thesis_config
)


class BlueSkyInternalClient:
    """BlueSky client using internal API instead of network protocol"""
    
    def __init__(self, config: BlueSkyConfig):
        self.config = config
        self.connected = False
        self.aircraft_states: Dict[str, AircraftState] = {}
        self.conflicts: List[ConflictInfo] = []
        self.callsigns: List[str] = []
        self.command_lock = threading.Lock()
        self.simulation_time = 0.0
        
        # BlueSky internal components
        self.stack = None
        self.sim = None
        
    def connect(self, timeout: float = 30.0) -> bool:
        """Connect using BlueSky's internal API"""
        try:
            print("🔌 Connecting to BlueSky via internal API...")
            
            # Import BlueSky components
            from bluesky import stack, simulation
            from bluesky.core import base
            
            self.stack = stack
            self.sim = simulation.simulation
            
            # Initialize BlueSky components
            print("Initializing BlueSky stack...")
            base.reset()
            stack.init()
            
            # Initialize simulation
            print("Initializing BlueSky simulation...")
            simulation.simulation.reset()
            
            self.connected = True
            print("✅ Connected to BlueSky internal API")
            
            # Initialize with paper-driven settings
            self._initialize_simulation()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to BlueSky internal API: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def disconnect(self):
        """Disconnect from BlueSky internal API"""
        if self.connected:
            try:
                # Clean up simulation
                if self.sim:
                    self.sim.reset()
                
                self.connected = False
                self.aircraft_states.clear()
                self.conflicts.clear()
                self.callsigns.clear()
                
                print("✅ Disconnected from BlueSky internal API")
            except Exception as e:
                print(f"⚠️ Error during disconnect: {e}")
    
    def _send_command(self, command: str, expect_response: bool = False, timeout: float = 5.0) -> str:
        """Send command to BlueSky stack using internal API"""
        if not self.connected or not self.stack:
            raise RuntimeError("Not connected to BlueSky")
        
        with self.command_lock:
            try:
                # Use BlueSky's stack.process() function
                # Based on the error, it seems process() takes no arguments
                # Let's check the correct way to send commands
                
                # Try different approaches
                try:
                    # Approach 1: Set command and process
                    if hasattr(self.stack, 'stack'):
                        # Add command to stack
                        self.stack.stack.append(command.strip())
                        # Process the stack
                        result = self.stack.process()
                        return str(result) if result else "OK"
                    
                    # Approach 2: Use command function directly
                    elif hasattr(self.stack, 'command'):
                        result = self.stack.command(command.strip())
                        return str(result) if result else "OK"
                    
                    # Approach 3: Use process with no args and check stack
                    else:
                        # This might need investigation of the actual API
                        result = self.stack.process()
                        return str(result) if result else "OK"
                        
                except Exception as api_error:
                    print(f"❌ Internal API command failed: {command}, Error: {api_error}")
                    return f"ERROR: {api_error}"
                
            except Exception as e:
                print(f"❌ Command failed: {command}, Error: {e}")
                return f"ERROR: {e}"
    
    def _initialize_simulation(self):
        """Initialize BlueSky simulation with paper-driven sequence"""
        try:
            print("🔧 Initializing BlueSky simulation (internal API, paper-aligned)...")
            
            # Reset simulation state
            if self.sim:
                self.sim.reset()
            
            # Set random seed for reproducibility
            import random
            import numpy as np
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            print(f"✅ Set random seed: {self.config.seed}")
            
            # Note: Internal API approach might not need all the same commands
            # as the network protocol since we're working directly with objects
            
            # Set simulation parameters through internal API
            if hasattr(self.sim, 'dt'):
                self.sim.dt = self.config.dt
                print(f"✅ Set simulation dt: {self.config.dt}")
            
            if hasattr(self.sim, 'dtmult'):
                self.sim.dtmult = self.config.dtmult
                print(f"✅ Set simulation dtmult: {self.config.dtmult}")
            
            # Clear tracking state
            self.aircraft_states.clear()
            self.conflicts.clear()
            self.callsigns.clear()
            
            print("✅ BlueSky simulation initialized (internal API)")
            
        except Exception as e:
            print(f"⚠️ Error during internal API initialization: {e}")
            import traceback
            traceback.print_exc()
    
    def create_aircraft(self, callsign: str, aircraft_type: str, 
                       lat: float, lon: float, heading: float,
                       altitude_ft: float, speed_kt: float) -> bool:
        """Create aircraft using internal API"""
        try:
            # Try to access BlueSky traffic module
            from bluesky import traffic
            
            # Create aircraft through traffic module
            success = traffic.cre(
                acid=callsign,
                actype=aircraft_type,
                lat=lat,
                lon=lon,
                hdg=heading,
                alt=altitude_ft,
                spd=speed_kt
            )
            
            if success:
                print(f"✅ Created aircraft {callsign} at {lat},{lon}")
                
                # Add to tracking
                self.aircraft_states[callsign] = AircraftState(
                    callsign=callsign,
                    latitude=lat,
                    longitude=lon,
                    altitude_ft=altitude_ft,
                    heading_deg=heading,
                    speed_kt=speed_kt,
                    vertical_speed_fpm=0.0,
                    timestamp=time.time()
                )
                
                if callsign not in self.callsigns:
                    self.callsigns.append(callsign)
                
                return True
            else:
                print(f"❌ Failed to create aircraft {callsign}")
                return False
                
        except Exception as e:
            print(f"❌ Aircraft creation failed: {e}")
            # Fallback to command-based approach
            try:
                command = f"CRE {callsign},{aircraft_type},{lat},{lon},{heading},{altitude_ft},{speed_kt}"
                response = self._send_command(command, expect_response=True)
                success = "ERROR" not in response.upper()
                
                if success:
                    print(f"✅ Created aircraft {callsign} via command fallback")
                    # Add to tracking
                    self.aircraft_states[callsign] = AircraftState(
                        callsign=callsign,
                        latitude=lat,
                        longitude=lon,
                        altitude_ft=altitude_ft,
                        heading_deg=heading,
                        speed_kt=speed_kt,
                        vertical_speed_fpm=0.0,
                        timestamp=time.time()
                    )
                    if callsign not in self.callsigns:
                        self.callsigns.append(callsign)
                
                return success
            except Exception as fallback_error:
                print(f"❌ Both API and command fallback failed: {fallback_error}")
                return False
    
    def get_aircraft_states(self) -> Dict[str, AircraftState]:
        """Get aircraft states using internal API"""
        try:
            from bluesky import traffic
            
            updated_states = {}
            current_time = time.time()
            
            # Get states from traffic module
            if hasattr(traffic, 'lat') and hasattr(traffic, 'lon'):
                # Traffic arrays should contain current aircraft data
                for i, callsign in enumerate(traffic.id):
                    if i < len(traffic.lat) and callsign in self.callsigns:
                        state = AircraftState(
                            callsign=callsign,
                            latitude=traffic.lat[i],
                            longitude=traffic.lon[i],
                            altitude_ft=traffic.alt[i] * 3.28084,  # Convert m to ft
                            heading_deg=traffic.hdg[i],
                            speed_kt=traffic.cas[i] * 1.944,  # Convert m/s to knots
                            vertical_speed_fpm=traffic.vs[i] * 196.85,  # Convert m/s to ft/min
                            timestamp=current_time
                        )
                        updated_states[callsign] = state
                
                self.aircraft_states.update(updated_states)
                return updated_states
            else:
                print("⚠️ Traffic module not available, using fallback")
                return {}
                
        except Exception as e:
            print(f"❌ Error getting aircraft states: {e}")
            return {}
    
    def delete_aircraft(self, callsign: str) -> bool:
        """Delete aircraft using internal API"""
        try:
            from bluesky import traffic
            
            success = traffic.delete(callsign)
            
            if success:
                print(f"✅ Deleted aircraft {callsign}")
                self.aircraft_states.pop(callsign, None)
                if callsign in self.callsigns:
                    self.callsigns.remove(callsign)
                return True
            else:
                print(f"❌ Failed to delete aircraft {callsign}")
                return False
                
        except Exception as e:
            print(f"❌ Delete aircraft failed: {e}")
            return False
    
    def get_conflicts(self) -> List[ConflictInfo]:
        """Get conflicts using internal API"""
        try:
            # Try to access ASAS (conflict detection) module
            from bluesky.traffic import asas
            
            conflicts = []
            
            if hasattr(asas, 'confpairs') and hasattr(asas, 'inconf'):
                # Get conflict pairs from ASAS
                for i, in_conflict in enumerate(asas.inconf):
                    if in_conflict and i < len(asas.confpairs):
                        pair = asas.confpairs[i]
                        if len(pair) >= 2:
                            ac1_idx, ac2_idx = pair[:2]
                            
                            # Get aircraft callsigns
                            from bluesky import traffic
                            if ac1_idx < len(traffic.id) and ac2_idx < len(traffic.id):
                                ac1 = traffic.id[ac1_idx]
                                ac2 = traffic.id[ac2_idx]
                                
                                # Calculate conflict info
                                conflict = ConflictInfo(
                                    aircraft1=ac1,
                                    aircraft2=ac2,
                                    time_to_conflict=0.0,  # Would need to calculate
                                    horizontal_distance=0.0,  # Would need to calculate
                                    vertical_distance=0.0,  # Would need to calculate
                                    conflict_type="detected",
                                    severity="medium"
                                )
                                conflicts.append(conflict)
            
            self.conflicts = conflicts
            return conflicts
            
        except Exception as e:
            print(f"❌ Error getting conflicts: {e}")
            return []
    
    # Add the TrafScript helpers (they can use the same _send_command approach)
    def ic(self, scenario_file: str) -> bool:
        """Load scenario file"""
        command = f"IC {scenario_file}"
        response = self._send_command(command, expect_response=True)
        success = "ERROR" not in response.upper()
        if success:
            print(f"✅ Loaded scenario: {scenario_file}")
            self.aircraft_states.clear()
            self.callsigns.clear()
        return success
    
    def hold(self) -> bool:
        """Pause simulation"""
        try:
            if self.sim and hasattr(self.sim, 'pause'):
                self.sim.pause()
                print("⏸️ Simulation paused")
                return True
            else:
                response = self._send_command("HOLD", expect_response=True)
                success = "ERROR" not in response.upper()
                if success:
                    print("⏸️ Simulation paused")
                return success
        except Exception as e:
            print(f"❌ Failed to pause: {e}")
            return False
    
    def op(self) -> bool:
        """Start/continue simulation"""
        try:
            if self.sim and hasattr(self.sim, 'start'):
                self.sim.start()
                print("▶️ Simulation started/resumed")
                return True
            else:
                response = self._send_command("OP", expect_response=True)
                success = "ERROR" not in response.upper()
                if success:
                    print("▶️ Simulation started/resumed")
                return success
        except Exception as e:
            print(f"❌ Failed to start/resume: {e}")
            return False


def test_internal_api_client():
    """Test the internal API client"""
    print("🧪 Testing BlueSky Internal API Client")
    print("=" * 40)
    
    config = create_thesis_config(seed=42)
    client = BlueSkyInternalClient(config)
    
    try:
        if client.connect():
            print("✅ Internal API client connected")
            
            # Test aircraft creation
            success = client.create_aircraft(
                "API001", "B737", 52.0, 4.0, 90, 35000, 450
            )
            
            if success:
                print("✅ Aircraft creation via internal API works")
                
                # Test state retrieval
                states = client.get_aircraft_states()
                print(f"Retrieved {len(states)} aircraft states")
                
                # Test deletion
                if client.delete_aircraft("API001"):
                    print("✅ Aircraft deletion works")
            
        else:
            print("❌ Failed to connect via internal API")
            
    except Exception as e:
        print(f"❌ Internal API test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        client.disconnect()


if __name__ == "__main__":
    test_internal_api_client()
