"""BlueSky simulator client with command interface"""

import socket
import subprocess
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


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


@dataclass
class BlueSkyConfig:
    """BlueSky configuration"""
    host: str = "127.0.0.1"
    port: int = 8888
    headless: bool = True
    fast_time_factor: float = 1.0
    conflict_detection: bool = True
    lookahead_time: float = 600.0


class BlueSkyClient:
    """BlueSky simulator client with command interface"""
    
    def __init__(self, config: BlueSkyConfig):
        self.config = config
        self.socket: Optional[socket.socket] = None
        self.process: Optional[subprocess.Popen] = None
        self.connected = False
        self.aircraft_states: Dict[str, AircraftState] = {}
    
    def connect(self, timeout: float = 30.0) -> bool:
        """Connect to BlueSky simulator"""
        try:
            # Launch BlueSky if not already running
            if not self._is_bluesky_running():
                self._launch_bluesky()
            
            # Connect to telnet interface
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.socket.connect((self.config.host, self.config.port))
                    self.connected = True
                    self._initialize_settings()
                    return True
                except ConnectionRefusedError:
                    time.sleep(1)
                    continue
            
            return False
            
        except Exception as e:
            print(f"Failed to connect to BlueSky: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from BlueSky"""
        if self.socket:
            self.socket.close()
            self.socket = None
        
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=10)
            self.process = None
        
        self.connected = False
    
    def stack_command(self, command: str) -> str:
        """Send command to BlueSky stack"""
        if not self.connected:
            raise RuntimeError("Not connected to BlueSky")
        
        try:
            command_bytes = (command + '\n').encode('utf-8')
            self.socket.send(command_bytes)
            response = self.socket.recv(4096).decode('utf-8').strip()
            return response
        except Exception as e:
            print(f"Command failed: {command}, Error: {e}")
            return f"ERROR: {e}"
    
    def create_aircraft(self, callsign: str, aircraft_type: str, 
                       lat: float, lon: float, heading: float,
                       altitude_ft: float, speed_kt: float) -> bool:
        """Create aircraft in simulation"""
        command = f"CRE {callsign},{aircraft_type},{lat},{lon},{heading},{altitude_ft},{speed_kt}"
        response = self.stack_command(command)
        return "ERROR" not in response.upper()
    
    def delete_aircraft(self, callsign: str) -> bool:
        """Delete aircraft from simulation"""
        command = f"DEL {callsign}"
        response = self.stack_command(command)
        return "ERROR" not in response.upper()
    
    def heading_command(self, callsign: str, heading_deg: float) -> bool:
        """Issue heading command"""
        command = f"HDG {callsign},{heading_deg}"
        response = self.stack_command(command)
        return "ERROR" not in response.upper()
    
    def altitude_command(self, callsign: str, altitude_ft: float, 
                        vertical_speed_fpm: Optional[float] = None) -> bool:
        """Issue altitude command"""
        if vertical_speed_fpm:
            command = f"ALT {callsign},{altitude_ft},{vertical_speed_fpm}"
        else:
            command = f"ALT {callsign},{altitude_ft}"
        response = self.stack_command(command)
        return "ERROR" not in response.upper()
    
    def speed_command(self, callsign: str, speed_kt: float) -> bool:
        """Issue speed command"""
        command = f"SPD {callsign},{speed_kt}"
        response = self.stack_command(command)
        return "ERROR" not in response.upper()
    
    def get_aircraft_states(self) -> Dict[str, AircraftState]:
        """Get current states of all aircraft"""
        # Mock implementation - in real system would parse BlueSky output
        states = {}
        
        for callsign in self.aircraft_states.keys():
            # Mock aircraft state
            states[callsign] = AircraftState(
                callsign=callsign,
                latitude=41.978,
                longitude=-87.904,
                altitude_ft=37000,
                heading_deg=270,
                speed_kt=450,
                vertical_speed_fpm=0,
                timestamp=time.time()
            )
        
        self.aircraft_states.update(states)
        return states
    
    def _launch_bluesky(self):
        """Launch BlueSky process"""
        cmd = ["python", "-m", "bluesky"]
        if self.config.headless:
            cmd.extend(["--headless"])
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            time.sleep(5)  # Wait for BlueSky to start
        except FileNotFoundError:
            print("BlueSky not found. Please install: pip install bluesky-simulator")
    
    def _is_bluesky_running(self) -> bool:
        """Check if BlueSky is already running"""
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(1)
            result = test_socket.connect_ex((self.config.host, self.config.port))
            test_socket.close()
            return result == 0
        except:
            return False
    
    def _initialize_settings(self):
        """Initialize BlueSky simulation settings"""
        self.stack_command(f"DTLOOK {self.config.lookahead_time}")
        self.stack_command("ASAS OFF")  # Disable ASAS for LLM-only resolution
        
        if self.config.fast_time_factor != 1.0:
            self.stack_command(f"DTMULT {self.config.fast_time_factor}")
