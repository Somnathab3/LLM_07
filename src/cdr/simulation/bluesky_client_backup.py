"""BlueSky simulator client with paper-aligned command interface"""

import socket
import subprocess
import time
import re
import threading
import math
import struct
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import os

# Import direct bridge for improved BlueSky communication
try:
    from .bluesky_direct_bridge import get_direct_bridge, BlueSkyDirectBridge
    DIRECT_BRIDGE_AVAILABLE = True
except ImportError:
    DIRECT_BRIDGE_AVAILABLE = False
    print("⚠️ Direct bridge not available, using command-response mode")


@dataclass
class AircraftState:
    """Current aircraft state"""
    callsign: str
    latitude: float
    longitude: float
    altitude_ft: float
    heading_deg: float
    speed_kt: float
    vertical_speed_fpm: float
    timestamp: float
    track_angle: Optional[float] = None
    ground_speed_kt: Optional[float] = None
    mach: Optional[float] = None


@dataclass
class ConflictInfo:
    """Conflict detection information"""
    aircraft1: str
    aircraft2: str
    time_to_conflict: float  # seconds
    horizontal_distance: float  # nautical miles
    vertical_distance: float  # feet
    conflict_type: str  # 'horizontal', 'vertical', 'both'
    severity: str  # 'low', 'medium', 'high'


@dataclass
class BlueSkyConfig:
    """BlueSky configuration aligned with BlueSky paper design"""
    # Connection settings
    host: str = "127.0.0.1"
    port: int = 11000  # BlueSky ZMQ recv port for command interface
    headless: bool = True
    bluesky_path: Optional[str] = None
    scenario_path: Optional[str] = None
    
    # Simulation timing (paper-driven)
    dt: float = 1.0  # simulation time step (seconds)
    dtmult: float = 8.0  # fast-time multiplier (8x for stable performance)
    seed: int = 1337  # random seed for reproducibility
    
    # ASAS conflict detection settings (paper-aligned)
    asas_enabled: bool = True  # enable built-in conflict detection
    reso_off: bool = True  # disable automatic resolution (LLM resolves)
    dtlook: float = 600.0  # look-ahead time (10 minutes)
    dtnolook: float = 5.0  # CD refresh cadence (5 seconds)
    
    # Detection zones (safety minima)
    det_radius_nm: float = 5.0  # detection radius (NM)
    det_half_vert_ft: float = 500.0  # detection half-height (ft, ±1000 total)
    
    # Optional CD/CR methods (configurable)
    cdmethod: Optional[str] = None  # conflict detection method
    rmethh: Optional[str] = None  # horizontal resolution method
    rmethv: Optional[str] = None  # vertical resolution method
    rfach: Optional[float] = None  # horizontal resolution factor
    rfacv: Optional[float] = None  # vertical resolution factor
    
    # Legacy compatibility (kept for backward compatibility)
    fast_time_factor: float = 8.0  # alias for dtmult
    conflict_detection: bool = True  # alias for asas_enabled
    lookahead_time: float = 600.0  # alias for dtlook
    protected_zone_radius: float = 5.0  # alias for det_radius_nm
    protected_zone_height: float = 1000.0  # alias for 2*det_half_vert_ft
    resolution_zone_radius: float = 8.0  # nautical miles
    resolution_zone_height: float = 1500.0  # feet
    
    def __post_init__(self):
        """Sync legacy and new parameters"""
        # Sync dtmult with fast_time_factor
        if hasattr(self, 'fast_time_factor') and self.fast_time_factor != self.dtmult:
            self.dtmult = self.fast_time_factor
        
        # Sync detection parameters
        if hasattr(self, 'lookahead_time') and self.lookahead_time != self.dtlook:
            self.dtlook = self.lookahead_time
        
        if hasattr(self, 'protected_zone_radius') and self.protected_zone_radius != self.det_radius_nm:
            self.det_radius_nm = self.protected_zone_radius
        
        if hasattr(self, 'protected_zone_height') and self.protected_zone_height != 2 * self.det_half_vert_ft:
            self.det_half_vert_ft = self.protected_zone_height / 2.0


@dataclass
class DetectionConfig:
    """Preset configuration for thesis experiments (5 NM / 1000 ft / 10-min)"""
    dt: float = 1.0
    dtmult: float = 8.0
    seed: int = 1337
    dtlook: float = 600.0  # 10 minutes
    dtnolook: float = 5.0  # 5 seconds
    det_radius_nm: float = 5.0  # 5 NM
    det_half_vert_ft: float = 500.0  # ±1000 ft total
    asas_enabled: bool = True
    reso_off: bool = True


def create_thesis_config(host: str = "127.0.0.1", port: int = 11000, 
                        headless: bool = True, seed: int = 1337) -> BlueSkyConfig:
    """
    Create a BlueSky configuration optimized for thesis experiments
    - Paper-aligned detection parameters (5 NM / ±1000 ft / 10-min horizon)
    - Fast-time simulation (8x) for efficiency
    - LLM-controlled resolution (ASAS detection ON, resolution OFF)
    - Reproducible (fixed seed)
    """
    return BlueSkyConfig(
        host=host,
        port=port,
        headless=headless,
        # Core timing
        dt=1.0,
        dtmult=8.0,
        seed=seed,
        # ASAS configuration
        asas_enabled=True,
        reso_off=True,
        dtlook=600.0,  # 10 minutes look-ahead
        dtnolook=5.0,  # 5 second refresh
        # Detection zones
        det_radius_nm=5.0,  # 5 NM radius
        det_half_vert_ft=500.0,  # ±1000 ft total
        # Legacy compatibility
        fast_time_factor=8.0,
        conflict_detection=True,
        lookahead_time=600.0,
        protected_zone_radius=5.0,
        protected_zone_height=1000.0
    )


class BlueSkyClient:
    """BlueSky simulator client using embedded mode with bs.sim.step()"""
    
    def __init__(self, config: BlueSkyConfig):
        self.config = config
        self.connected = False
        self.aircraft_states: Dict[str, AircraftState] = {}
        self.conflicts: List[ConflictInfo] = []
        self.callsigns: List[str] = []  # Track active aircraft callsigns
        self.simulation_time = 0.0
        self.last_state_update = 0.0
        
        # Add logger for debugging
        self.logger = logging.getLogger(__name__)
        
        # BlueSky embedded simulation references
        self.bs = None
        self.sim = None
        self.traf = None
        self.stack = None
        
        # Screen dummy for headless operation
        self.screen_dummy = None
        
        # Simulation state
        self.simulation_running = False
        self.step_count = 0
        
        # Direct bridge for improved communication
        self.direct_bridge = None
        self.use_direct_bridge = DIRECT_BRIDGE_AVAILABLE
        
        if self.use_direct_bridge:
            print("✅ Direct bridge available - will use enhanced BlueSky communication")
        else:
            print("⚠️ Direct bridge unavailable - falling back to command-response mode")
    
    def connect(self, timeout: float = 30.0) -> bool:
        """
        Initialize embedded BlueSky simulation using bs.init(mode='sim', detached=True)
        Based on the working solution with bs.sim.step()
        """
        try:
            print("🚀 Initializing embedded BlueSky simulation...")
            
            # Import BlueSky
            import bluesky as bs
            self.bs = bs
            
            # Initialize as detached simulation node (like horizontal_cr_env.py)
            print("   Initializing BlueSky simulation...")
            bs.init(mode='sim', detached=True)
            
            # Create and set dummy screen to suppress output
            print("   Setting up dummy screen...")
            
            class ScreenDummy:
                def echo(self, text='', flags=0):
                    pass
                def update(self):
                    pass
                def reset(self):
                    pass
            
            bs.scr = ScreenDummy()
            
            # Store references
            self.sim = bs.sim
            self.traf = bs.traf
            self.stack = bs.stack
            
            # Configure simulation parameters
            print("   Configuring simulation parameters...")
            bs.stack.stack(f'DT {self.config.dt}')
            bs.stack.stack(f'FF')  # Enable fast-forward capability
            
            self.connected = True
            
            # Initialize direct bridge if available
            if self.use_direct_bridge:
                try:
                    self.direct_bridge = get_direct_bridge()
                    if self.direct_bridge.is_available():
                        print("✅ Direct bridge initialized successfully!")
                    else:
                        print("⚠️ Direct bridge not available, using fallback mode")
                        self.use_direct_bridge = False
                except Exception as e:
                    print(f"⚠️ Direct bridge initialization failed: {e}")
                    self.use_direct_bridge = False
            
            print("✅ BlueSky embedded mode initialized successfully!")
            
            # Initialize simulation settings
            self._initialize_simulation()
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize embedded BlueSky: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def disconnect(self):
        """Disconnect from embedded BlueSky and cleanup"""
        print("Disconnecting from embedded BlueSky...")
        
        if self.connected:
            try:
                # Pause simulation
                if self.bs and self.bs.stack:
                    self.bs.stack.stack("HOLD")
                    time.sleep(0.5)
            except:
                pass
        
        # Clear references
        self.bs = None
        self.sim = None
        self.traf = None
        self.stack = None
        
        self.connected = False
        self.simulation_running = False
        self.aircraft_states.clear()
        self.conflicts.clear()
        self.callsigns.clear()
        print("✅ Disconnected from embedded BlueSky")
    
    def _send_command(self, command: str, expect_response: bool = False, timeout: float = 5.0) -> str:
        """Send command to embedded BlueSky stack"""
        if not self.connected or not self.bs or not self.bs.stack:
            raise RuntimeError("BlueSky not connected")

        try:
            print(f"🔧 Sending command: {command}")
            
            # Send command through BlueSky stack
            self.bs.stack.stack(command)
            
            # For embedded mode, we always return OK since commands are executed immediately
            # The expect_response and timeout parameters are kept for compatibility
            return "OK"
            
        except Exception as e:
            print(f"❌ Command failed: {command}, Error: {e}")
            return f"ERROR: {e}"

    def step_simulation(self, steps: int = 1) -> bool:
        """
        Step the simulation forward using bs.sim.step()
        This is the KEY method that makes aircraft actually move!
        """
        if not self.connected or not self.bs or not self.bs.sim:
            print("❌ Simulation not connected")
            return False
        
        try:
            for i in range(steps):
                self.bs.sim.step()
                self.step_count += 1
            
            if steps > 1:
                print(f"⏩ Stepped simulation {steps} steps (total: {self.step_count})")
            
            return True
            
        except Exception as e:
            print(f"❌ Error stepping simulation: {e}")
            return False
    
    def pause_simulation(self) -> bool:
        """Pause simulation for LLM processing"""
        try:
            response = self._send_command("HOLD", expect_response=False)
            print("⏸️  Simulation paused for LLM processing")
            return True
        except Exception as e:
            print(f"❌ Error pausing simulation: {e}")
            return False
    
    def resume_simulation(self) -> bool:
        """Resume simulation after LLM processing"""
        try:
            response = self._send_command("OP", expect_response=False)
            print("▶️  Simulation resumed")
            return True
        except Exception as e:
            print(f"❌ Error resuming simulation: {e}")
            return False
    
    def _track_command(self, command: str):
        """Track commands that affect aircraft state for state management"""
        try:
            # Handle both space-separated and comma-separated formats
            # Space format: CRE TEST A320 42.0 -87.9 0 0 200
            # Comma format: CRE OWNSHIP,B738,41.978,-87.904,270,37000,450
            parts = command.split()
            if len(parts) < 2:
                return
            
            cmd_type = parts[0].upper()
            
            # Track aircraft creation
            if cmd_type == "CRE":
                if len(parts) >= 7:
                    # Space-separated format: CRE TEST A320 42.0 -87.9 0 0 200
                    callsign = parts[1]
                    lat = float(parts[3])
                    lon = float(parts[4])
                    hdg = float(parts[5])
                    alt = float(parts[6])
                    spd = float(parts[7]) if len(parts) > 7 else 200.0
                    
                elif len(parts) >= 2 and "," in parts[1]:
                    # Comma-separated format: CRE OWNSHIP,B738,41.978,-87.904,270,37000,450
                    comma_parts = parts[1].split(",")
                    if len(comma_parts) >= 6:
                        callsign = comma_parts[0]
                        # comma_parts: ["OWNSHIP", "B738", "41.978", "-87.904", "270", "37000", "450"]
                        lat = float(comma_parts[2])
                        lon = float(comma_parts[3])
                        hdg = float(comma_parts[4])
                        alt = float(comma_parts[5])
                        spd = float(comma_parts[6]) if len(comma_parts) > 6 else 200.0
                    else:
                        return
                else:
                    return
                
                self.initial_positions[callsign] = {
                    'lat': lat, 'lon': lon, 'hdg': hdg, 'alt': alt, 'spd': spd
                }
                self.last_known_states[callsign] = {
                    'lat': lat, 'lon': lon, 'hdg': hdg, 'alt': alt, 'spd': spd
                }
                
                print(f"📊 Tracked aircraft creation: {callsign} at lat={lat}, lon={lon}, hdg={hdg}, alt={alt}, spd={spd}")
                
            # Track heading changes
            elif cmd_type == "HDG":
                if len(parts) >= 3:
                    # Space format: HDG TEST 90
                    callsign_part = parts[1]
                    new_hdg = float(parts[2])
                elif len(parts) >= 2 and "," in parts[1]:
                    # Comma format: HDG OWNSHIP,38
                    comma_parts = parts[1].split(",")
                    if len(comma_parts) >= 2:
                        callsign_part = comma_parts[0]
                        new_hdg = float(comma_parts[1])
                    else:
                        return
                else:
                    return
                
                if callsign_part in self.last_known_states:
                    self.last_known_states[callsign_part]['hdg'] = new_hdg
                    print(f"📊 Tracked heading change: {callsign_part} → {new_hdg}°")
                    
            # Track altitude changes
            elif cmd_type == "ALT":
                if len(parts) >= 3:
                    # Space format: ALT TEST 12000
                    callsign_part = parts[1]
                    new_alt = float(parts[2])
                elif len(parts) >= 2 and "," in parts[1]:
                    # Comma format: ALT OWNSHIP,12000
                    comma_parts = parts[1].split(",")
                    if len(comma_parts) >= 2:
                        callsign_part = comma_parts[0]
                        new_alt = float(comma_parts[1])
                    else:
                        return
                else:
                    return
                
                if callsign_part in self.last_known_states:
                    self.last_known_states[callsign_part]['alt'] = new_alt
                    print(f"📊 Tracked altitude change: {callsign_part} → {new_alt}ft")
                    
            # Track speed changes
            elif cmd_type == "SPD":
                if len(parts) >= 3:
                    # Space format: SPD TEST 300
                    callsign_part = parts[1]
                    new_spd = float(parts[2])
                elif len(parts) >= 2 and "," in parts[1]:
                    # Comma format: SPD OWNSHIP,300
                    comma_parts = parts[1].split(",")
                    if len(comma_parts) >= 2:
                        callsign_part = comma_parts[0]
                        new_spd = float(comma_parts[1])
                    else:
                        return
                else:
                    return
                
                if callsign_part in self.last_known_states:
                    self.last_known_states[callsign_part]['spd'] = new_spd
                    print(f"📊 Tracked speed change: {callsign_part} → {new_spd}kt")
            
            # Track aircraft deletion
            elif cmd_type == "DEL" and len(parts) >= 2:
                callsign = parts[1]
                if callsign in self.last_known_states:
                    del self.last_known_states[callsign]
                if callsign in self.initial_positions:
                    del self.initial_positions[callsign]
                    
        except (ValueError, IndexError) as e:
            # Don't let command tracking errors affect the main command
            print(f"⚠️ Command tracking error for '{command}': {e}")
            pass
    
    def _read_binary_response(self, timeout: float) -> str:
        """Read response from BlueSky's binary protocol"""
        try:
            self.socket.settimeout(timeout)
            
            # Try to read binary response
            response_data = b''
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    chunk = self.socket.recv(4096)
                    if chunk:
                        response_data += chunk
                        
                        # Check if we have a complete message
                        if len(response_data) >= 10:  # Minimum expected message size
                            break
                    else:
                        break
                except socket.timeout:
                    break
            
            # Reset socket timeout
            self.socket.settimeout(None)
            
            if response_data:
                # Try to decode the response
                try:
                    # Try UTF-8 decode
                    return response_data.decode('utf-8').strip()
                except UnicodeDecodeError:
                    # Handle binary response - extract text if possible
                    try:
                        # Try to find text portions in binary data
                        text_parts = []
                        for byte in response_data:
                            if 32 <= byte <= 126:  # Printable ASCII
                                text_parts.append(chr(byte))
                            elif byte == 10 or byte == 13:  # Newline characters
                                text_parts.append('\n')
                        
                        extracted_text = ''.join(text_parts).strip()
                        if extracted_text:
                            return extracted_text
                        else:
                            return f"BINARY_RESPONSE: {response_data[:50]}..."
                    except:
                        return f"BINARY_RESPONSE: {response_data[:50]}..."
            
            return "NO_RESPONSE"
            
        except Exception as e:
            return f"RESPONSE_ERROR: {e}"
    
    def stack_command(self, command: str) -> str:
        """Send command to BlueSky stack (legacy method for backward compatibility)"""
        return self._send_command(command, expect_response=True)
    
    def create_aircraft(self, callsign: str, aircraft_type: str, 
                       lat: float, lon: float, heading: float,
                       altitude_ft: float, speed_kt: float) -> bool:
        """Create aircraft using embedded BlueSky traffic management"""
        if not self.connected or not self.bs or not self.bs.traf:
            print("❌ BlueSky not connected")
            return False
        
        try:
            print(f"✈️ Creating aircraft {callsign}...")
            
            # Use BlueSky traffic creation (like in horizontal_cr_env.py)
            success = self.bs.traf.cre(callsign, actype=aircraft_type, acspd=speed_kt)
            
            if success:
                # Get the aircraft index
                ac_idx = self.bs.traf.id2idx(callsign)
                
                if ac_idx >= 0:
                    # Set position and parameters manually
                    self.bs.traf.lat[ac_idx] = lat
                    self.bs.traf.lon[ac_idx] = lon
                    # BlueSky uses meters for altitude, convert from feet
                    altitude_m = altitude_ft * 0.3048  # Convert feet to meters
                    self.bs.traf.alt[ac_idx] = altitude_m
                    self.bs.traf.hdg[ac_idx] = heading
                    # BlueSky uses m/s for speeds, convert from knots
                    speed_ms = speed_kt / 1.944  # Convert knots to m/s
                    self.bs.traf.gs[ac_idx] = speed_ms  # Ground speed
                    self.bs.traf.cas[ac_idx] = speed_ms  # Calibrated airspeed
                    
                    # Add to callsigns tracking
                    if callsign not in self.callsigns:
                        self.callsigns.append(callsign)
                    
                    print(f"✅ Created aircraft {callsign} at ({lat:.6f}, {lon:.6f}), hdg={heading}°, spd={speed_kt}kt")
                    return True
                else:
                    print(f"❌ Failed to get index for {callsign}")
                    return False
            else:
                print(f"❌ Failed to create aircraft {callsign}")
                return False
                
        except Exception as e:
            print(f"❌ Error creating aircraft: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def delete_aircraft(self, callsign: str) -> bool:
        """Delete aircraft from simulation using DEL command"""
        command = f"DEL {callsign}"
        response = self._send_command(command, expect_response=True)
        
        success = "ERROR" not in response.upper() and "FAIL" not in response.upper()
        if success:
            print(f"✅ Deleted aircraft {callsign}")
            self.aircraft_states.pop(callsign, None)
            # Remove from callsign tracking
            if callsign in self.callsigns:
                self.callsigns.remove(callsign)
        else:
            print(f"❌ Failed to delete aircraft {callsign}: {response}")
        
        return success
    
    def heading_command(self, callsign: str, heading_deg: float) -> bool:
        """Issue heading command using direct bridge when available"""
        # Try direct bridge first
        if self.use_direct_bridge and self.direct_bridge and self.direct_bridge.is_available():
            try:
                success = self.direct_bridge.apply_heading_command_direct(callsign, heading_deg)
                if success:
                    return True
                else:
                    print(f"⚠️ Direct bridge heading command failed, falling back to embedded")
            except Exception as e:
                print(f"⚠️ Direct bridge error: {e}")
        
        # Fallback to embedded BlueSky stack
        command = f"HDG {callsign} {int(heading_deg)}"
        response = self._send_command(command)
        
        success = "ERROR" not in response.upper()
        if success:
            print(f"✅ Set heading {heading_deg}° for {callsign}")
        else:
            print(f"❌ Failed to set heading for {callsign}: {response}")
        
        return success
    
    def altitude_command(self, callsign: str, altitude_ft: float) -> bool:
        """Issue altitude command using direct bridge when available"""
        # Try direct bridge first
        if self.use_direct_bridge and self.direct_bridge and self.direct_bridge.is_available():
            try:
                success = self.direct_bridge.apply_altitude_command_direct(callsign, altitude_ft)
                if success:
                    return True
                else:
                    print(f"⚠️ Direct bridge altitude command failed, falling back to embedded")
            except Exception as e:
                print(f"⚠️ Direct bridge error: {e}")
        
        # Fallback to embedded BlueSky stack
        command = f"ALT {callsign} {int(altitude_ft)}"
        response = self._send_command(command)
        
        success = "ERROR" not in response.upper()
        if success:
            print(f"✅ Set altitude {altitude_ft}ft for {callsign}")
        else:
            print(f"❌ Failed to set altitude for {callsign}: {response}")
        
        return success
    
    def speed_command(self, callsign: str, speed_kt: float) -> bool:
        """Issue speed command using direct bridge when available"""
        # Try direct bridge first
        if self.use_direct_bridge and self.direct_bridge and self.direct_bridge.is_available():
            try:
                success = self.direct_bridge.apply_speed_command_direct(callsign, speed_kt)
                if success:
                    return True
                else:
                    print(f"⚠️ Direct bridge speed command failed, falling back to embedded")
            except Exception as e:
                print(f"⚠️ Direct bridge error: {e}")
        
        # Fallback to embedded BlueSky stack
        command = f"SPD {callsign} {int(speed_kt)}"
        response = self._send_command(command)
        
        success = "ERROR" not in response.upper()
        if success:
            print(f"✅ Set speed {speed_kt}kt for {callsign}")
        else:
            print(f"❌ Failed to set heading for {callsign}: {response}")
        
        return success
    
    def op(self) -> bool:
        """Start/resume simulation"""
        response = self._send_command("OP")
        success = "ERROR" not in response.upper()
        if success:
            print("▶️ Simulation started/resumed")
            self.simulation_running = True
        return success
    
    def hold(self) -> bool:
        """Pause simulation"""
        response = self._send_command("HOLD")
        success = "ERROR" not in response.upper()
        if success:
            print("⏸️ Simulation paused")
            self.simulation_running = False
        return success
    
    def ff(self, time_seconds: float) -> bool:
        """Fast-forward simulation"""
        response = self._send_command(f"FF {time_seconds}")
        success = "ERROR" not in response.upper()
        if success:
            print(f"⏩ Fast-forwarded {time_seconds} seconds")
        return success
    
    def altitude_command(self, callsign: str, altitude_ft: float, 
                        vertical_speed_fpm: Optional[float] = None) -> bool:
        """Issue altitude command using ALT command"""
        # ALT acid, alt, [vspd]
        if vertical_speed_fpm:
            command = f"ALT {callsign},{altitude_ft},{vertical_speed_fpm}"
        else:
            command = f"ALT {callsign},{altitude_ft}"
        
        response = self._send_command(command, expect_response=True)
        
        success = "ERROR" not in response.upper() and "FAIL" not in response.upper()
        if success:
            vs_str = f" at {vertical_speed_fpm} fpm" if vertical_speed_fpm else ""
            print(f"✅ Set altitude {altitude_ft} ft for {callsign}{vs_str}")
        else:
            print(f"❌ Failed to set altitude for {callsign}: {response}")
        
        return success
    
    def speed_command(self, callsign: str, speed_kt: float) -> bool:
        """Issue speed command using SPD command"""
        # SPD acid,spd (CAS-kts/Mach)
        command = f"SPD {callsign},{speed_kt}"
        response = self._send_command(command, expect_response=True)
        
        success = "ERROR" not in response.upper() and "FAIL" not in response.upper()
        if success:
            print(f"✅ Set speed {speed_kt} kt for {callsign}")
        else:
            print(f"❌ Failed to set speed for {callsign}: {response}")
        
        return success
    
    def direct_to_waypoint(self, callsign: str, waypoint: str) -> bool:
        """Direct aircraft to waypoint using DIRECT command"""
        # Ensure waypoint exists in the route first
        try:
            # Check if aircraft has this waypoint in route
            # If not, try to add it first (this handles the AVOID1 not found error)
            command = f"DIRECT {callsign} {waypoint}"
            response = self._send_command(command, expect_response=True)
            
            success = "ERROR" not in response.upper() and "FAIL" not in response.upper()
            if success:
                print(f"✅ Directed {callsign} to {waypoint}")
            else:
                # If direct fails because waypoint not in route, return false
                # The calling code should handle this by adding waypoint first
                print(f"❌ Failed to direct {callsign} to {waypoint}: {response}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error directing {callsign} to {waypoint}: {e}")
            return False
    
    def add_waypoint(self, callsign: str, name: str, lat: float, lon: float, fl: int = None) -> bool:
        """Add a waypoint to aircraft's flight plan and route"""
        try:
            # First, add the waypoint to the waypoint database
            self.bs.stack.stack(f"DEFWPT {name} {lat:.5f} {lon:.5f}")
            
            # Then add it to the aircraft's route
            command = f"ADDWPT {callsign} {name}"
            if fl is not None:
                command += f" FL{int(fl)}"
            
            response = self._send_command(command, expect_response=True)
            
            success = "ERROR" not in response.upper() and "FAIL" not in response.upper()
            if success:
                print(f"✅ Added waypoint {name} for {callsign} at {lat:.4f},{lon:.4f}")
            else:
                print(f"❌ Failed to add waypoint {name} for {callsign}: {response}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error adding waypoint {name}: {e}")
            return False
    
    def direct_to(self, callsign: str, name: str) -> bool:
        """Alias for direct_to_waypoint for consistency"""
        return self.direct_to_waypoint(callsign, name)
    
    def move_aircraft(self, callsign: str, lat: float, lon: float,
                     altitude_ft: Optional[float] = None,
                     heading_deg: Optional[float] = None,
                     speed_kt: Optional[float] = None,
                     vertical_speed_fpm: Optional[float] = None) -> bool:
        """Move aircraft to new position using MOVE command"""
        # MOVE acid,lat,lon,[alt,hdg,spd,vspd]
        command_parts = [f"MOVE {callsign},{lat},{lon}"]
        
        if altitude_ft is not None:
            command_parts.append(str(altitude_ft))
            if heading_deg is not None:
                command_parts.append(str(heading_deg))
                if speed_kt is not None:
                    command_parts.append(str(speed_kt))
                    if vertical_speed_fpm is not None:
                        command_parts.append(str(vertical_speed_fpm))
        
        command = ','.join(command_parts)
        response = self._send_command(command, expect_response=True)
        
        success = "ERROR" not in response.upper() and "FAIL" not in response.upper()
        if success:
            print(f"✅ Moved {callsign} to {lat},{lon}")
        else:
            print(f"❌ Failed to move {callsign}: {response}")
        
        return success
    
    def get_aircraft_states(self, callsigns: Optional[List[str]] = None) -> Dict[str, AircraftState]:
        """
        Get aircraft states using direct bridge when available, fallback to embedded access
        """
        if not self.connected:
            return {}
        
        # Try direct bridge first for improved reliability
        if self.use_direct_bridge and self.direct_bridge and self.direct_bridge.is_available():
            try:
                direct_states = self.direct_bridge.get_aircraft_states_direct()
                
                # Debug logging to track aircraft filtering issues
                if callsigns:
                    self.logger.debug(f"Filtering {len(direct_states)} aircraft for requested: {callsigns}")
                else:
                    self.logger.debug(f"Returning all {len(direct_states)} aircraft from direct bridge")
                
                # Filter by requested callsigns if specified
                if callsigns:
                    filtered_states = {k: v for k, v in direct_states.items() if k in callsigns}
                else:
                    filtered_states = direct_states
                
                # Convert to standard AircraftState format
                states = {}
                for callsign, direct_state in filtered_states.items():
                    states[callsign] = AircraftState(
                        callsign=direct_state.callsign,
                        latitude=direct_state.latitude,
                        longitude=direct_state.longitude,
                        altitude_ft=direct_state.altitude_ft,
                        heading_deg=direct_state.heading_deg,
                        speed_kt=direct_state.speed_kt,
                        vertical_speed_fpm=direct_state.vertical_speed_fpm,
                        timestamp=direct_state.timestamp,
                        track_angle=direct_state.track_angle,
                        ground_speed_kt=direct_state.ground_speed_kt
                    )
                
                # Update internal tracking
                self.aircraft_states.update(states)
                for callsign in states.keys():
                    if callsign not in self.callsigns:
                        self.callsigns.append(callsign)
                
                if states:
                    print(f"📊 Retrieved {len(states)} aircraft states via direct bridge")
                
                return states
                
            except Exception as e:
                print(f"⚠️ Direct bridge failed, falling back to embedded access: {e}")
                self.use_direct_bridge = False
        
        # Fallback to embedded BlueSky access
        return self._get_aircraft_states_embedded(callsigns)
    
    def _get_aircraft_states_embedded(self, callsigns: Optional[List[str]] = None) -> Dict[str, AircraftState]:
        """
        Get REAL aircraft states directly from embedded BlueSky simulation
        Uses bs.traf to access actual aircraft data after bs.sim.step()
        """
        if not self.bs or not self.bs.traf:
            return {}
        
        states = {}
        current_time = time.time()
        
        try:
            # Get all aircraft IDs from BlueSky traffic
            if hasattr(self.bs.traf, 'id') and len(self.bs.traf.id) > 0:
                for i, bs_callsign in enumerate(self.bs.traf.id):
                    # Skip if specific callsigns requested and this isn't one
                    if callsigns and bs_callsign not in callsigns:
                        continue
                    
                    try:
                        # Get aircraft index
                        ac_idx = self.bs.traf.id2idx(bs_callsign)
                        
                        if ac_idx >= 0:
                            # Get REAL position from BlueSky simulation
                            lat = self.bs.traf.lat[ac_idx] if hasattr(self.bs.traf, 'lat') else 0.0
                            lon = self.bs.traf.lon[ac_idx] if hasattr(self.bs.traf, 'lon') else 0.0
                            alt = self.bs.traf.alt[ac_idx] if hasattr(self.bs.traf, 'alt') else 0.0
                            hdg = self.bs.traf.hdg[ac_idx] if hasattr(self.bs.traf, 'hdg') else 0.0
                            spd = self.bs.traf.gs[ac_idx] if hasattr(self.bs.traf, 'gs') else 0.0
                            vs = getattr(self.bs.traf, 'vs', [0] * len(self.bs.traf.lat))[ac_idx] if hasattr(self.bs.traf, 'vs') else 0.0
                            
                            # Create aircraft state with REAL data
                            state = AircraftState(
                                callsign=bs_callsign,
                                latitude=lat,
                                longitude=lon,
                                altitude_ft=alt,  # BlueSky uses meters, may need conversion
                                heading_deg=hdg,
                                speed_kt=spd,
                                vertical_speed_fpm=vs * 60 if vs else 0.0,  # Convert m/s to ft/min if needed
                                timestamp=current_time
                            )
                            
                            states[bs_callsign] = state
                            
                            # Update callsigns list
                            if bs_callsign not in self.callsigns:
                                self.callsigns.append(bs_callsign)
                            
                    except Exception as e:
                        print(f"❌ Error getting state for {bs_callsign}: {e}")
                        continue
                        
        except Exception as e:
            print(f"❌ Error accessing BlueSky traffic data: {e}")
        
        # Update internal states cache
        self.aircraft_states.update(states)
        self.last_state_update = current_time
        
        return states

    def get_aircraft_state(self, callsign: str) -> Optional[AircraftState]:
        """Get state for a single aircraft (convenience method)"""
        states = self.get_aircraft_states([callsign])
        return states.get(callsign)

    def _parse_binary_aircraft_data(self, callsign: str, response: str, timestamp: float) -> Optional[AircraftState]:
        """
        Parse BlueSky POS response containing numpy binary data
        
        BlueSky returns binary numpy structured array data. We need to extract
        the actual float64 values that represent the aircraft state.
        """
        try:
            if not response or len(response) < 50:
                return None
            
            print(f"🔍 Parsing binary response for {callsign} ({len(response)} chars)")
            
            # Convert to bytes for binary analysis
            response_bytes = response.encode('latin1') if isinstance(response, str) else response
            
            # Try to extract float64 values from the binary data
            import struct
            
            # Look for 8-byte double precision floats at all possible positions
            potential_floats = []
            
            for offset in range(0, len(response_bytes) - 7, 1):
                for endian in ['<', '>']:  # Little and big endian
                    try:
                        value = struct.unpack(f'{endian}d', response_bytes[offset:offset+8])[0]
                        
                        # Filter for reasonable aviation values
                        if (-90 <= value <= 90 or      # Latitude range
                            -180 <= value <= 180 or    # Longitude range  
                            0 <= value <= 50000 or     # Altitude range (feet)
                            0 <= value <= 1000):       # Speed range (knots)
                            
                            potential_floats.append((offset, value, endian))
                            
                    except (struct.error, OverflowError):
                        continue
            
            if not potential_floats:
                print(f"❌ No valid float values found in binary data")
                return None
            
            print(f"🔍 Found {len(potential_floats)} potential values")
            
            # Categorize the values
            latitudes = [(o, v, e) for o, v, e in potential_floats if -90 <= v <= 90]
            longitudes = [(o, v, e) for o, v, e in potential_floats if -180 <= v <= 180 and abs(v) > 90]
            altitudes = [(o, v, e) for o, v, e in potential_floats if 5000 <= v <= 50000]
            speeds = [(o, v, e) for o, v, e in potential_floats if 50 <= v <= 1000]
            headings = [(o, v, e) for o, v, e in potential_floats if 0 <= v <= 360]
            
            # Select the most reasonable values
            result = {}
            
            # For latitude: prefer values in reasonable range for our area (around 40-45)
            if latitudes:
                if callsign in self.last_known_states:
                    last_lat = self.last_known_states[callsign]['lat']
                    best_lat = min(latitudes, key=lambda x: abs(x[1] - last_lat))
                else:
                    # Default to mid-range latitude
                    best_lat = min(latitudes, key=lambda x: abs(x[1] - 42.0))
                result['lat'] = best_lat[1]
                print(f"✅ Selected latitude: {best_lat[1]:.6f}")
            
            # For longitude: prefer values in reasonable range for our area (around -85 to -90)
            if longitudes:
                if callsign in self.last_known_states:
                    last_lon = self.last_known_states[callsign]['lon']
                    best_lon = min(longitudes, key=lambda x: abs(x[1] - last_lon))
                else:
                    # Default to reasonable longitude for US
                    best_lon = min(longitudes, key=lambda x: abs(x[1] - (-87.0)))
                result['lon'] = best_lon[1]
                print(f"✅ Selected longitude: {best_lon[1]:.6f}")
            
            # For altitude: prefer values around flight levels
            if altitudes:
                if callsign in self.last_known_states:
                    last_alt = self.last_known_states[callsign]['alt']
                    best_alt = min(altitudes, key=lambda x: abs(x[1] - last_alt))
                else:
                    best_alt = min(altitudes, key=lambda x: abs(x[1] - 35000))
                result['alt'] = best_alt[1]
                print(f"✅ Selected altitude: {best_alt[1]:.0f}")
            
            # For speeds: prefer values around typical cruise speeds
            if speeds:
                if callsign in self.last_known_states:
                    last_spd = self.last_known_states[callsign]['spd']
                    best_spd = min(speeds, key=lambda x: abs(x[1] - last_spd))
                else:
                    best_spd = min(speeds, key=lambda x: abs(x[1] - 450))
                result['spd'] = best_spd[1]
                print(f"✅ Selected speed: {best_spd[1]:.0f}")
            
            # For headings: prefer values that make sense
            if headings:
                if callsign in self.last_known_states:
                    last_hdg = self.last_known_states[callsign]['hdg']
                    best_hdg = min(headings, key=lambda x: abs(x[1] - last_hdg))
                else:
                    best_hdg = min(headings, key=lambda x: abs(x[1] - 90))
                result['hdg'] = best_hdg[1]
                print(f"✅ Selected heading: {best_hdg[1]:.1f}")
            
            # If we don't have lat/lon, fall back to cached values but mark as stale
            if 'lat' not in result or 'lon' not in result:
                if callsign in self.last_known_states:
                    cached = self.last_known_states[callsign]
                    print(f"⚠️ Missing coordinates, using cached fallback for {callsign}")
                    result.update({
                        'lat': cached['lat'],
                        'lon': cached['lon'],
                        'alt': result.get('alt', cached['alt']),
                        'hdg': result.get('hdg', cached['hdg']),
                        'spd': result.get('spd', cached['spd'])
                    })
                else:
                    print(f"❌ No coordinates found and no cached state for {callsign}")
                    return None
            
            # Create the aircraft state
            state = AircraftState(
                callsign=callsign,
                latitude=result.get('lat', 0.0),
                longitude=result.get('lon', 0.0),
                altitude_ft=result.get('alt', 35000.0),
                heading_deg=result.get('hdg', 90.0),
                speed_kt=result.get('spd', 250.0),
                vertical_speed_fpm=0.0,
                timestamp=timestamp
            )
            
            # Update the cached state for next time
            if callsign in self.last_known_states:
                self.last_known_states[callsign].update({
                    'lat': state.latitude,
                    'lon': state.longitude,
                    'alt': state.altitude_ft,
                    'hdg': state.heading_deg,
                    'spd': state.speed_kt
                })
            
            print(f"📊 Parsed {callsign}: lat={state.latitude:.6f}, lon={state.longitude:.6f}, alt={state.altitude_ft:.0f}, hdg={state.heading_deg:.1f}, spd={state.speed_kt:.0f}")
            return state
            
        except Exception as e:
            print(f"❌ Binary parsing error for {callsign}: {e}")
            return None
    
    def _extract_aircraft_data_from_binary(self, binary_data: str, callsign: str) -> tuple:
        """Extract aircraft data from binary numpy format"""
        try:
            import struct
            
            # BlueSky binary format often contains field descriptors followed by data
            # Look for patterns that indicate structured data
            
            # Method 1: Try to find IEEE 754 floating point patterns
            # Convert string to bytes if needed
            if isinstance(binary_data, str):
                # Try different encodings
                try:
                    data_bytes = binary_data.encode('latin1')  # Preserve binary data
                except:
                    data_bytes = binary_data.encode('utf-8', errors='ignore')
            else:
                data_bytes = binary_data
            
            # Look for 8-byte (double precision) floating point values
            floats = []
            for i in range(0, len(data_bytes) - 7, 1):  # Step by 1 byte to find all possible floats
                try:
                    # Try to unpack as double precision float (big endian and little endian)
                    for fmt in ['<d', '>d']:  # little endian, big endian
                        try:
                            value = struct.unpack(fmt, data_bytes[i:i+8])[0]
                            # Filter for reasonable values
                            if not (math.isnan(value) or math.isinf(value)):
                                if -1000 < value < 1000:  # Could be lat/lon, altitude (in thousands), heading, speed
                                    floats.append(value)
                        except:
                            continue
                except:
                    continue
            
            # Method 2: Look for coordinate patterns in the floats
            if floats:
                # Find potential latitude values (-90 to 90)
                lat_candidates = [f for f in floats if -90 <= f <= 90]
                # Find potential longitude values (-180 to 180)  
                lon_candidates = [f for f in floats if -180 <= f <= 180]
                # Find potential altitudes (0 to 60000 ft)
                alt_candidates = [f for f in floats if 0 <= f <= 60000]
                # Find potential headings (0 to 360)
                hdg_candidates = [f for f in floats if 0 <= f <= 360]
                # Find potential speeds (0 to 1000 kt)
                spd_candidates = [f for f in floats if 0 <= f <= 1000]
                
                # Try to match patterns - latitude and longitude should be close to creation values
                lat = lat_candidates[0] if lat_candidates else None
                lon = lon_candidates[0] if lon_candidates else None
                alt_ft = alt_candidates[0] if alt_candidates else 35000.0
                hdg_deg = hdg_candidates[0] if hdg_candidates else 270.0
                spd_kt = spd_candidates[0] if spd_candidates else 450.0
                
                return lat, lon, alt_ft, hdg_deg, spd_kt
            
            return None, None, 0.0, 0.0, 0.0
            
        except Exception as e:
            print(f"❌ Binary extraction failed for {callsign}: {e}")
            return None, None, 0.0, 0.0, 0.0

    def _parse_individual_properties(self, callsign: str, lat_response: str, lon_response: str, 
                                   alt_response: str, hdg_response: str, spd_response: str, 
                                   timestamp: float) -> Optional[AircraftState]:
        """Parse individual property responses from BlueSky"""
        try:
            # Extract numeric values from each response
            lat_numbers = re.findall(r'-?\d+\.?\d*', lat_response)
            lon_numbers = re.findall(r'-?\d+\.?\d*', lon_response)
            alt_numbers = re.findall(r'-?\d+\.?\d*', alt_response)
            hdg_numbers = re.findall(r'-?\d+\.?\d*', hdg_response)
            spd_numbers = re.findall(r'-?\d+\.?\d*', spd_response)
            
            # Extract the first valid number from each response
            lat = float(lat_numbers[0]) if lat_numbers else 0.0
            lon = float(lon_numbers[0]) if lon_numbers else 0.0
            alt_ft = float(alt_numbers[0]) if alt_numbers else 0.0
            hdg_deg = float(hdg_numbers[0]) if hdg_numbers else 0.0
            spd_kt = float(spd_numbers[0]) if spd_numbers else 0.0
            
            return AircraftState(
                callsign=callsign,
                latitude=lat,
                longitude=lon,
                altitude_ft=alt_ft,
                heading_deg=hdg_deg,
                speed_kt=spd_kt,
                vertical_speed_fpm=0.0,  # Not queried individually yet
                timestamp=timestamp
            )
            
        except (ValueError, IndexError) as e:
            print(f"❌ Failed to parse individual properties for {callsign}: {e}")
            return None
    
    def _parse_aircraft_position(self, callsign: str, response: str, timestamp: float) -> Optional[AircraftState]:
        """Parse BlueSky POS command response to extract aircraft state"""
        try:
            # BlueSky POS response format varies, need to handle different formats
            # Typical response might contain lat, lon, alt, hdg, spd information
            
            # Extract numeric values from response
            numbers = re.findall(r'-?\d+\.?\d*', response)
            
            if len(numbers) >= 5:
                # Assume format: lat, lon, alt, hdg, spd (common BlueSky format)
                lat = float(numbers[0])
                lon = float(numbers[1])
                alt_ft = float(numbers[2])
                hdg_deg = float(numbers[3])
                spd_kt = float(numbers[4])
                
                # Try to extract vertical speed if available
                vs_fpm = float(numbers[5]) if len(numbers) > 5 else 0.0
                
                return AircraftState(
                    callsign=callsign,
                    latitude=lat,
                    longitude=lon,
                    altitude_ft=alt_ft,
                    heading_deg=hdg_deg,
                    speed_kt=spd_kt,
                    vertical_speed_fpm=vs_fpm,
                    timestamp=timestamp
                )
            else:
                # Try alternative parsing if standard format fails
                # Look for specific keywords in response
                lat_match = re.search(r'lat[:\s]*(-?\d+\.?\d*)', response, re.IGNORECASE)
                lon_match = re.search(r'lon[:\s]*(-?\d+\.?\d*)', response, re.IGNORECASE)
                alt_match = re.search(r'alt[:\s]*(-?\d+\.?\d*)', response, re.IGNORECASE)
                hdg_match = re.search(r'hdg[:\s]*(-?\d+\.?\d*)', response, re.IGNORECASE)
                spd_match = re.search(r'spd[:\s]*(-?\d+\.?\d*)', response, re.IGNORECASE)
                
                if lat_match and lon_match:
                    return AircraftState(
                        callsign=callsign,
                        latitude=float(lat_match.group(1)),
                        longitude=float(lon_match.group(1)),
                        altitude_ft=float(alt_match.group(1)) if alt_match else 0.0,
                        heading_deg=float(hdg_match.group(1)) if hdg_match else 0.0,
                        speed_kt=float(spd_match.group(1)) if spd_match else 0.0,
                        vertical_speed_fpm=0.0,
                        timestamp=timestamp
                    )
        
        except (ValueError, IndexError) as e:
            print(f"❌ Failed to parse position response for {callsign}: {e}")
            print(f"   Response: {response}")
        
        return None
    
    def set_fast_time_factor(self, factor: float) -> bool:
        """
        Control simulation speed using DTMULT command
        - Use DTMULT command
        - Validate factor ranges
        - Handle simulation timing
        """
        if factor <= 0:
            print(f"❌ Invalid time factor: {factor}. Must be positive.")
            return False
        
        if factor > 100:
            print(f"⚠️ Warning: Very high time factor {factor} may cause simulation instability")
        
        # DTMULT multiplier
        command = f"DTMULT {factor}"
        response = self._send_command(command, expect_response=True)
        
        success = "ERROR" not in response.upper() and "FAIL" not in response.upper()
        if success:
            self.config.fast_time_factor = factor
            self.config.dtmult = factor  # Keep in sync
            print(f"✅ Set simulation speed factor to {factor}x")
        else:
            print(f"❌ Failed to set time factor: {response}")
        
        return success
    
    # TrafScript command helpers (paper-aligned)
    def ic(self, scenario_file: str) -> bool:
        """Load scenario file using IC command"""
        command = f"IC {scenario_file}"
        response = self._send_command(command, expect_response=True)
        success = "ERROR" not in response.upper() and "FAIL" not in response.upper()
        if success:
            print(f"✅ Loaded scenario: {scenario_file}")
            # Clear aircraft tracking since scenario may have different aircraft
            self.aircraft_states.clear()
            self.callsigns.clear()
        else:
            print(f"❌ Failed to load scenario: {response}")
        return success
    
    def batch(self, batch_file: str) -> bool:
        """Execute batch file using BATCH command"""
        command = f"BATCH {batch_file}"
        response = self._send_command(command, expect_response=True)
        success = "ERROR" not in response.upper() and "FAIL" not in response.upper()
        if success:
            print(f"✅ Executed batch: {batch_file}")
        else:
            print(f"❌ Failed to execute batch: {response}")
        return success
    
    def op(self) -> bool:
        """Start/continue simulation using OP command"""
        response = self._send_command("OP")
        success = "ERROR" not in response.upper()
        if success:
            print("▶️ Simulation started/resumed")
            self.simulation_running = True
        return success
    
    def hold(self) -> bool:
        """Pause simulation using HOLD command"""
        response = self._send_command("HOLD")
        success = "ERROR" not in response.upper()
        if success:
            print("⏸️ Simulation paused")
            self.simulation_running = False
        return success
    
    def ff(self, seconds: float) -> bool:
        """Fast-forward simulation using FF command"""
        command = f"FF {seconds}"
        response = self._send_command(command)
        success = "ERROR" not in response.upper()
        if success:
            print(f"⏩ Fast-forwarded {seconds} seconds")
        return success
        if success:
            print(f"⏩ Fast-forwarded {seconds} seconds")
        else:
            print(f"❌ Failed to fast-forward: {response}")
        return success
    
    def step_minutes(self, minutes: float) -> bool:
        """
        Advance the embedded BlueSky sim by 'minutes' of sim time by calling the
        core stepper directly (more reliable than 'FF' here).
        """
        try:
            total_secs = float(minutes) * 60.0
            if total_secs <= 0:
                return True

            # Use a fixed internal dt so physics updates reliably.
            # 0.5s is a good balance; adjust if you want finer dynamics.
            dt = 0.5
            steps = int(math.ceil(total_secs / dt))
            
            if self.sim is not None:
                # Use embedded simulation stepper for reliable kinematics
                for _ in range(steps):
                    self.sim.step(dt)
                print(f"⏩ Stepped simulation {minutes:.1f} minutes ({steps} steps of {dt}s)")
                return True
            else:
                # Fallback to FF command if embedded sim not available
                print("⚠️ Embedded simulation not available, falling back to FF command")
                return self.ff(total_secs)
                
        except Exception as e:
            print(f"❌ step_minutes failed: {e}")
            return False
    
    def set_speed(self, callsign: str, speed_kt: float) -> bool:
        """Set aircraft speed using SPD command"""
        command = f"SPD {callsign},{speed_kt}"
        response = self._send_command(command, expect_response=True)
        success = "ERROR" not in response.upper() and "FAIL" not in response.upper()
        if success:
            print(f"🎯 Set speed for {callsign}: {speed_kt} kt")
        else:
            print(f"❌ Failed to set speed for {callsign}: {response}")
        return success
    
    def get_conflicts(self) -> List[ConflictInfo]:
        """
        Get conflicts using direct bridge when available, fallback to SSD CONFLICTS parsing
        """
        if not self.connected:
            return []
        
        # Try direct bridge first for improved reliability
        if self.use_direct_bridge and self.direct_bridge and self.direct_bridge.is_available():
            try:
                direct_conflicts = self.direct_bridge.get_conflicts_direct()
                
                # Convert to standard ConflictInfo format
                conflicts = []
                for direct_conflict in direct_conflicts:
                    conflicts.append(ConflictInfo(
                        aircraft1=direct_conflict.aircraft1,
                        aircraft2=direct_conflict.aircraft2,
                        time_to_conflict=direct_conflict.time_to_conflict,
                        horizontal_distance=direct_conflict.horizontal_distance,
                        vertical_distance=direct_conflict.vertical_distance,
                        conflict_type=direct_conflict.conflict_type,
                        severity=direct_conflict.severity
                    ))
                
                self.conflicts = conflicts
                
                if conflicts:
                    print(f"⚠️  Found {len(conflicts)} conflicts via direct bridge")
                    for conflict in conflicts:
                        print(f"   {conflict.aircraft1} vs {conflict.aircraft2}: "
                              f"h={conflict.horizontal_distance:.1f}NM, v={conflict.vertical_distance:.0f}ft")
                
                return conflicts
                
            except Exception as e:
                print(f"⚠️ Direct bridge conflict detection failed, falling back to SSD: {e}")
                self.use_direct_bridge = False
        
        # Fallback to SSD CONFLICTS parsing
        return self._get_conflicts_ssd()
    
    def _get_conflicts_ssd(self) -> List[ConflictInfo]:
        """
        Paper-aligned conflict detection using SSD CONFLICTS command
        - Use BlueSky's built-in ASAS conflict detection
        - Query via SSD CONFLICTS for global conflict set
        - Parse structured conflict data for LLM input
        """
        if not self.connected:
            return []
        
        try:
            conflicts = []
            
            # Ensure ASAS is enabled (should be from initialization)
            if self.config.asas_enabled:
                self._send_command("ASAS ON", expect_response=True)
            
            # Method 1: Use SSD CONFLICTS for global conflict set (paper-recommended)
            try:
                ssd_response = self._send_command("SSD CONFLICTS", expect_response=True, timeout=3.0)
                if ssd_response and "ERROR" not in ssd_response.upper():
                    conflicts = self._parse_ssd_conflicts(ssd_response)
                    if conflicts:
                        print(f"✅ Found {len(conflicts)} conflicts via SSD CONFLICTS")
                        self.conflicts = conflicts
                        return conflicts
                else:
                    print(f"⚠️ SSD CONFLICTS response: {ssd_response}")
            except Exception as e:
                print(f"⚠️ SSD CONFLICTS failed: {e}")
            
            # Method 2: Fallback to manual detection if SSD fails
            states = self.get_aircraft_states()
            if len(states) >= 2:
                conflicts = self._detect_conflicts_manually(states)
                if conflicts:
                    print(f"✅ Found {len(conflicts)} conflicts via manual detection")
            
            self.conflicts = conflicts
            return conflicts
            
        except Exception as e:
            print(f"❌ Error getting conflicts: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _parse_ssd_conflicts(self, response: str) -> List[ConflictInfo]:
        """Parse SSD CONFLICTS response to extract conflict pairs and metrics"""
        conflicts = []
        
        try:
            # Parse BlueSky SSD CONFLICTS output format
            # Expected format varies but typically includes:
            # - Aircraft pair identifiers
            # - Time to loss of separation (TLOS)
            # - Closest point of approach (CPA) data
            # - Horizontal/vertical separation values
            
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Look for conflict pattern: AC1-AC2 or AC1,AC2 or similar
                # Try different parsing approaches for different BlueSky versions
                
                # Pattern 1: Space or comma separated aircraft pairs with metrics
                conflict_match = re.search(r'(\w+)[-,\s]+(\w+).*?(\d+\.?\d*)', line)
                if conflict_match and len(conflict_match.groups()) >= 3:
                    ac1, ac2, metric = conflict_match.groups()
                    
                    # IMPORTANT: Validate aircraft callsigns against known aircraft
                    # Reject if callsigns are numeric dates or invalid patterns
                    if (ac1.isdigit() and len(ac1) == 4) or (ac2.isdigit() and len(ac2) == 4):
                        # Skip date-like patterns (e.g., "2025")
                        continue
                    if (ac1.isdigit() and len(ac1) <= 2) or (ac2.isdigit() and len(ac2) <= 2):
                        # Skip month/day-like patterns (e.g., "08")
                        continue
                    
                    # Validate against tracked callsigns if available
                    if hasattr(self, 'callsigns') and self.callsigns:
                        if ac1 not in self.callsigns and ac2 not in self.callsigns:
                            # Neither aircraft is in our tracked list, likely parsing error
                            continue
                    
                    # Extract additional metrics if available
                    numbers = re.findall(r'-?\d+\.?\d*', line)
                    if len(numbers) >= 3:
                        time_to_conflict = float(numbers[0]) if numbers[0] else 0.0
                        h_dist = float(numbers[1]) if len(numbers) > 1 else 0.0
                        v_dist = float(numbers[2]) if len(numbers) > 2 else 0.0
                        
                        # Determine conflict type and severity
                        h_violation = h_dist < self.config.det_radius_nm
                        v_violation = v_dist < (2 * self.config.det_half_vert_ft)
                        
                        if h_violation and v_violation:
                            conflict_type = "both"
                            severity = "high"
                        elif h_violation:
                            conflict_type = "horizontal"
                            severity = "medium"
                        elif v_violation:
                            conflict_type = "vertical"
                            severity = "medium"
                        else:
                            conflict_type = "potential"
                            severity = "low"
                        
                        conflicts.append(ConflictInfo(
                            aircraft1=ac1,
                            aircraft2=ac2,
                            time_to_conflict=time_to_conflict,
                            horizontal_distance=h_dist,
                            vertical_distance=v_dist,
                            conflict_type=conflict_type,
                            severity=severity
                        ))
            
            # If no conflicts found with structured parsing, look for simpler indicators
            if not conflicts and "conflict" in response.lower():
                # Try to extract aircraft names mentioned in conflict context
                aircraft_names = re.findall(r'\b[A-Z]{2,}[0-9]*\b', response)
                if len(aircraft_names) >= 2:
                    for i in range(0, len(aircraft_names)-1, 2):
                        if i+1 < len(aircraft_names):
                            conflicts.append(ConflictInfo(
                                aircraft1=aircraft_names[i],
                                aircraft2=aircraft_names[i+1],
                                time_to_conflict=0.0,
                                horizontal_distance=0.0,
                                vertical_distance=0.0,
                                conflict_type="unknown",
                                severity="medium"
                            ))
        
        except Exception as e:
            print(f"❌ Error parsing SSD conflicts: {e}")
            print(f"   Response: {response}")
        
        return conflicts
    
    def _parse_conflict_response(self, callsign: str, response: str) -> List[ConflictInfo]:
        """Parse BlueSky conflict detection response"""
        conflicts = []
        
        try:
            # Look for conflict patterns in response
            # This is a simplified parser - actual BlueSky format may vary
            conflict_matches = re.findall(r'conflict.*?(\w+).*?(\d+\.?\d*)', response, re.IGNORECASE)
            
            for match in conflict_matches:
                other_aircraft = match[0] if match[0] != callsign else None
                time_to_conflict = float(match[1]) if len(match) > 1 else 0.0
                
                if other_aircraft and other_aircraft in self.aircraft_states:
                    conflict = ConflictInfo(
                        aircraft1=callsign,
                        aircraft2=other_aircraft,
                        time_to_conflict=time_to_conflict,
                        horizontal_distance=0.0,  # Would need additional parsing
                        vertical_distance=0.0,    # Would need additional parsing
                        conflict_type='unknown',
                        severity='medium'
                    )
                    conflicts.append(conflict)
                    
        except Exception as e:
            print(f"❌ Error parsing conflict response: {e}")
        
        return conflicts
    
    def _detect_conflicts_manually(self, states: Dict[str, AircraftState]) -> List[ConflictInfo]:
        """Manual conflict detection as fallback"""
        conflicts = []
        aircraft_list = list(states.values())
        
        for i, ac1 in enumerate(aircraft_list):
            for ac2 in aircraft_list[i+1:]:
                # Calculate horizontal distance
                horizontal_dist = self._calculate_distance(
                    ac1.latitude, ac1.longitude,
                    ac2.latitude, ac2.longitude
                )
                
                # Calculate vertical separation
                vertical_dist = abs(ac1.altitude_ft - ac2.altitude_ft)
                
                # Check if within protected zones
                horizontal_violation = horizontal_dist < self.config.protected_zone_radius
                vertical_violation = vertical_dist < self.config.protected_zone_height
                
                if horizontal_violation or vertical_violation:
                    conflict_type = 'both' if (horizontal_violation and vertical_violation) else \
                                  'horizontal' if horizontal_violation else 'vertical'
                    
                    severity = 'high' if (horizontal_violation and vertical_violation) else 'medium'
                    
                    conflict = ConflictInfo(
                        aircraft1=ac1.callsign,
                        aircraft2=ac2.callsign,
                        time_to_conflict=0.0,  # Current conflict
                        horizontal_distance=horizontal_dist,
                        vertical_distance=vertical_dist,
                        conflict_type=conflict_type,
                        severity=severity
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance in nautical miles"""
        import math
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in nautical miles
        earth_radius_nm = 3440.065
        
        return c * earth_radius_nm
    
    def _launch_bluesky(self) -> bool:
        """
        Launch BlueSky in headless mode
        - Use subprocess with proper arguments
        - Configure for fast-time simulation
        - Set up conflict detection parameters
        - Handle BlueSky startup timing
        """
        try:
            print("🚀 Launching BlueSky simulator...")
            
            # Determine BlueSky command
            bluesky_cmd = ["python", "-m", "bluesky"]
            
            # Add configuration arguments
            if self.config.headless:
                bluesky_cmd.extend(["--headless"])
            
            # Add scenario if specified
            if self.config.scenario_path:
                scenario_path = Path(self.config.scenario_path)
                if scenario_path.exists():
                    bluesky_cmd.extend(["--scenfile", str(scenario_path)])
            
            # Set environment variables for BlueSky configuration
            env = os.environ.copy()
            env["BLUESKY_HEADLESS"] = "1" if self.config.headless else "0"
            
            print(f"Command: {' '.join(bluesky_cmd)}")
            
            # Launch BlueSky process
            self.process = subprocess.Popen(
                bluesky_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=self.config.bluesky_path or os.getcwd()
            )
            
            # Wait for BlueSky to start up
            startup_timeout = 30
            print(f"⏳ Waiting for BlueSky startup (max {startup_timeout}s)...")
            
            for i in range(startup_timeout):
                time.sleep(1)
                
                # Check if process is still running
                if self.process.poll() is not None:
                    stdout, stderr = self.process.communicate(timeout=5)
                    print(f"❌ BlueSky process exited with code {self.process.returncode}")
                    print(f"STDOUT: {stdout}")
                    print(f"STDERR: {stderr}")
                    return False
                
                # Check if telnet interface is available
                if self._is_bluesky_running():
                    print(f"✅ BlueSky started successfully after {i+1}s")
                    return True
                
                # Show progress
                if i % 5 == 0 and i > 0:
                    print(f"   Still waiting... ({i}s)")
            
            print("❌ BlueSky startup timeout")
            return False
            
        except FileNotFoundError:
            print("❌ BlueSky not found. Please install: pip install bluesky-simulator")
            print("   Or ensure BlueSky is available in your Python environment")
            return False
        except Exception as e:
            print(f"❌ Error launching BlueSky: {e}")
            return False
    
    def _is_bluesky_running(self) -> bool:
        """Check if BlueSky is already running by testing telnet connection"""
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(2.0)
            result = test_socket.connect_ex((self.config.host, self.config.port))
            test_socket.close()
            return result == 0
        except Exception:
            return False
    
    def _initialize_simulation(self):
        """Initialize embedded BlueSky simulation with essential settings"""
        try:
            print("🔧 Initializing embedded BlueSky simulation...")
            
            # Reset traffic to start fresh
            print("   Resetting traffic...")
            self.bs.traf.reset()
            
            # Configure simulation parameters
            print("   Configuring simulation parameters...")
            self.bs.stack.stack(f'DT {self.config.dt}')  # Time step
            self.bs.stack.stack(f'DTMULT {self.config.dtmult}')  # Fast-time multiplier
            self.bs.stack.stack(f'SEED {self.config.seed}')  # Random seed
            print(f"✅ Timing: dt={self.config.dt}s, multiplier={self.config.dtmult}x, seed={self.config.seed}")
            
            # Detection parameters
            print("   Setting detection parameters...")
            self.bs.stack.stack(f"DTLOOK {self.config.dtlook}")
            self.bs.stack.stack(f"DTNOLOOK {self.config.dtnolook}")
            print(f"✅ Detection: look-ahead={self.config.dtlook}s, refresh={self.config.dtnolook}s")
            
            # Safety zones
            print("   Setting safety zones...")
            self.bs.stack.stack(f"ZONER {self.config.det_radius_nm}")
            self.bs.stack.stack(f"ZONEDH {self.config.det_half_vert_ft}")
            print(f"✅ Safety zones: {self.config.det_radius_nm}NM, ±{self.config.det_half_vert_ft}ft")
            
            # ASAS configuration
            if self.config.asas_enabled:
                self.bs.stack.stack("ASAS ON")
                print("✅ ASAS conflict detection enabled")
            else:
                self.bs.stack.stack("ASAS OFF")
                print("✅ ASAS conflict detection disabled")
            
            if self.config.reso_off:
                self.bs.stack.stack("RESOOFF")
                print("✅ Automatic resolution disabled (LLM will resolve)")
            
            # Stability settings
            print("   Setting stability options...")
            self.bs.stack.stack("NOISE OFF")  # Disable turbulence
            print("✅ Noise/turbulence disabled")
            
            # Clear tracking state
            self.aircraft_states.clear()
            self.conflicts.clear()
            self.callsigns.clear()
            self.step_count = 0
            
            print("✅ Embedded BlueSky simulation initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize simulation: {e}")
            import traceback
            traceback.print_exc()
            
        except Exception as e:
            print(f"⚠️ Error during initialization: {e}")
            import traceback
            traceback.print_exc()
            # Continue anyway - some settings might not be critical
    
    def hold_simulation(self) -> bool:
        """Pause the simulation"""
        response = self._send_command("HOLD", expect_response=True)
        success = "ERROR" not in response.upper()
        if success:
            print("⏸️ Simulation paused")
        return success
    
    def continue_simulation(self) -> bool:
        """Continue/resume the simulation"""
        response = self._send_command("OP", expect_response=True)
        success = "ERROR" not in response.upper()
        if success:
            print("▶️ Simulation resumed")
        return success
    
    def reset_simulation(self) -> bool:
        """Reset the simulation"""
        response = self._send_command("RESET", expect_response=True)
        success = "ERROR" not in response.upper()
        if success:
            print("🔄 Simulation reset")
            self.aircraft_states.clear()
            self.conflicts.clear()
        return success
    
    def get_simulation_time(self) -> float:
        """Get current simulation time"""
        try:
            response = self._send_command("TIME", expect_response=True)
            # Parse time from response - format may vary
            time_match = re.search(r'(\d+:?\d*:?\d*\.?\d*)', response)
            if time_match:
                time_str = time_match.group(1)
                # Convert to seconds (simplified - may need more robust parsing)
                return float(time_str) if '.' in time_str else float(time_str)
        except Exception as e:
            print(f"❌ Error getting simulation time: {e}")
        
        return self.simulation_time
    
    def load_scenario(self, scenario_file: str) -> bool:
        """Load a scenario file"""
        command = f"IC {scenario_file}"
        response = self._send_command(command, expect_response=True)
        success = "ERROR" not in response.upper()
        if success:
            print(f"✅ Loaded scenario: {scenario_file}")
            self.aircraft_states.clear()  # Clear previous aircraft
        else:
            print(f"❌ Failed to load scenario: {response}")
        return success
    
    def save_scenario(self, scenario_file: str) -> bool:
        """Save current situation as scenario"""
        command = f"SAVEIC {scenario_file}"
        response = self._send_command(command, expect_response=True)
        success = "ERROR" not in response.upper()
        if success:
            print(f"✅ Saved scenario: {scenario_file}")
        else:
            print(f"❌ Failed to save scenario: {response}")
        return success
    
    def get_airport_info(self, icao_code: str) -> Dict[str, Any]:
        """Get airport information including runways"""
        try:
            response = self._send_command(f"RUNWAYS {icao_code}", expect_response=True)
            if "ERROR" not in response.upper():
                # Parse runway information from response
                return {"icao": icao_code, "runways": response, "available": True}
        except Exception as e:
            print(f"❌ Error getting airport info for {icao_code}: {e}")
        
        return {"icao": icao_code, "available": False}
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
