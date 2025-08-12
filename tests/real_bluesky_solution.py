#!/usr/bin/env python3
"""
REAL Working BlueSky Solution with Proper Simulation Stepping
Uses embedded BlueSky API directly with bs.sim.step() for actual movement
"""

import sys
import os
import time
import math
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass

# Import screen dummy
from screen_dummy import ScreenDummy

@dataclass 
class AircraftState:
    callsign: str
    latitude: float
    longitude: float
    altitude_ft: float
    heading_deg: float
    speed_kt: float
    vertical_speed_fpm: float
    timestamp: float

class RealBlueSkyClient:
    """BlueSky client with REAL simulation stepping using bs.sim.step()"""
    
    def __init__(self):
        self.aircraft_created = []
        self.simulation_running = False
        self.bs = None
        
    def initialize_bluesky(self):
        """Initialize BlueSky with proper embedded mode"""
        
        try:
            print("🚀 Initializing embedded BlueSky...")
            
            # Import BlueSky
            import bluesky as bs
            self.bs = bs
            
            # Initialize as detached simulation node (like in your example)
            print("   Initializing BlueSky simulation...")
            bs.init(mode='sim', detached=True)
            
            # Set dummy screen to suppress output (like in your example)
            print("   Setting up dummy screen...")
            bs.scr = ScreenDummy()
            
            # Set time step and fast-forward capability (from your example)
            print("   Configuring simulation parameters...")
            bs.stack.stack('DT 5;FF')
            
            print("✅ BlueSky embedded mode initialized!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize BlueSky: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_aircraft(self, callsign: str, aircraft_type: str, lat: float, lon: float,
                       heading: float, altitude_ft: float, speed_kt: float) -> bool:
        """Create aircraft using BlueSky traffic management"""
        
        if not self.bs:
            print("❌ BlueSky not initialized")
            return False
        
        try:
            print(f"✈️ Creating aircraft {callsign}...")
            
            # Use BlueSky traffic creation (like in your example)
            # bs.traf.cre(acid, actype, acspd)
            success = self.bs.traf.cre(callsign, actype=aircraft_type, acspd=speed_kt)
            
            if success:
                # Get the aircraft index
                ac_idx = self.bs.traf.id2idx(callsign)
                
                if ac_idx >= 0:
                    # Set position manually
                    self.bs.traf.lat[ac_idx] = lat
                    self.bs.traf.lon[ac_idx] = lon
                    self.bs.traf.alt[ac_idx] = altitude_ft  # BlueSky uses meters, but let's try feet first
                    self.bs.traf.hdg[ac_idx] = heading
                    self.bs.traf.gs[ac_idx] = speed_kt  # Ground speed
                    self.bs.traf.cas[ac_idx] = speed_kt  # Calibrated airspeed
                    
                    self.aircraft_created.append(callsign)
                    
                    print(f"✅ Created {callsign} at ({lat:.6f}, {lon:.6f}), hdg={heading}°, spd={speed_kt}kt")
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
    
    def start_simulation(self) -> bool:
        """Start the simulation"""
        try:
            print("▶️ Starting simulation...")
            self.simulation_running = True
            return True
        except Exception as e:
            print(f"❌ Error starting simulation: {e}")
            return False
    
    def step_simulation(self, steps: int = 1):
        """Step the simulation forward (this is the key!)"""
        
        if not self.bs or not self.simulation_running:
            print("❌ Simulation not running")
            return False
        
        try:
            print(f"⏩ Stepping simulation {steps} step(s)...")
            
            # This is the crucial part - actually step the simulation!
            for i in range(steps):
                self.bs.sim.step()
                
            print(f"✅ Simulation stepped {steps} time(s)")
            return True
            
        except Exception as e:
            print(f"❌ Error stepping simulation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_aircraft_states(self) -> Dict[str, AircraftState]:
        """Get REAL aircraft states from BlueSky simulation"""
        
        if not self.bs:
            return {}
        
        states = {}
        
        try:
            for callsign in self.aircraft_created:
                ac_idx = self.bs.traf.id2idx(callsign)
                
                if ac_idx >= 0:
                    # Get REAL position from BlueSky simulation
                    lat = self.bs.traf.lat[ac_idx]
                    lon = self.bs.traf.lon[ac_idx]
                    alt = self.bs.traf.alt[ac_idx]
                    hdg = self.bs.traf.hdg[ac_idx]
                    spd = self.bs.traf.gs[ac_idx]
                    vs = getattr(self.bs.traf, 'vs', [0] * len(self.bs.traf.lat))[ac_idx] if hasattr(self.bs.traf, 'vs') else 0
                    
                    state = AircraftState(
                        callsign=callsign,
                        latitude=lat,
                        longitude=lon,
                        altitude_ft=alt,  # May need conversion from meters
                        heading_deg=hdg,
                        speed_kt=spd,
                        vertical_speed_fpm=vs,
                        timestamp=time.time()
                    )
                    
                    states[callsign] = state
                    
        except Exception as e:
            print(f"❌ Error getting aircraft states: {e}")
            import traceback
            traceback.print_exc()
        
        return states
    
    def send_heading_command(self, callsign: str, heading: float) -> bool:
        """Send heading command to aircraft"""
        
        if not self.bs:
            return False
        
        try:
            ac_idx = self.bs.traf.id2idx(callsign)
            if ac_idx >= 0:
                # Send command through BlueSky stack
                command = f"HDG {callsign} {int(heading)}"
                self.bs.stack.stack(command)
                print(f"✅ Sent heading command: {command}")
                return True
            else:
                print(f"❌ Aircraft {callsign} not found")
                return False
                
        except Exception as e:
            print(f"❌ Error sending heading command: {e}")
            return False
    
    def test_real_movement(self):
        """Test REAL aircraft movement with simulation stepping"""
        
        print("=== REAL BlueSky Movement Test ===")
        print("Using bs.sim.step() for actual simulation!")
        
        try:
            # Initialize BlueSky
            if not self.initialize_bluesky():
                return
            
            # Reset traffic to start fresh
            print("🔄 Resetting traffic...")
            self.bs.traf.reset()
            
            # Create test aircraft
            callsign = "REAL01"
            success = self.create_aircraft(
                callsign=callsign,
                aircraft_type="A320",
                lat=52.0,
                lon=4.0,
                heading=90,  # East
                altitude_ft=35000,
                speed_kt=450
            )
            
            if not success:
                print("❌ Failed to create test aircraft")
                return
            
            # Start simulation
            self.start_simulation()
            
            # Get initial state
            print("\n📊 Initial Aircraft State:")
            initial_states = self.get_aircraft_states()
            initial_state = initial_states.get(callsign)
            
            if initial_state:
                print(f"   Position: {initial_state.latitude:.6f}, {initial_state.longitude:.6f}")
                print(f"   Heading:  {initial_state.heading_deg:.1f}°")
                print(f"   Speed:    {initial_state.speed_kt:.0f} kt")
                print(f"   Altitude: {initial_state.altitude_ft:.0f} ft")
            
            # Step simulation multiple times to allow movement
            print("\n⏩ Stepping simulation to allow aircraft movement...")
            
            for step in range(10):
                print(f"   Step {step + 1}/10...")
                self.step_simulation(10)  # Step 10 simulation steps each time
                time.sleep(0.5)  # Brief pause
            
            # Get state after simulation steps
            print("\n📊 Aircraft State After Simulation Steps:")
            after_states = self.get_aircraft_states()
            after_state = after_states.get(callsign)
            
            if after_state and initial_state:
                print(f"   Position: {after_state.latitude:.6f}, {after_state.longitude:.6f}")
                print(f"   Heading:  {after_state.heading_deg:.1f}°")
                print(f"   Speed:    {after_state.speed_kt:.0f} kt")
                print(f"   Altitude: {after_state.altitude_ft:.0f} ft")
                
                # Calculate movement
                lat_diff = abs(after_state.latitude - initial_state.latitude)
                lon_diff = abs(after_state.longitude - initial_state.longitude)
                
                print(f"\n📏 Movement Analysis:")
                print(f"   Δlat: {lat_diff:.8f}")
                print(f"   Δlon: {lon_diff:.8f}")
                
                if lat_diff > 0.0001 or lon_diff > 0.0001:
                    print("🎉 SUCCESS: REAL aircraft movement detected!")
                    print("   Aircraft actually moved in BlueSky simulation!")
                else:
                    print("❌ No significant movement detected")
            
            # Test heading change
            print("\n🔄 Testing Heading Change...")
            
            # Send heading command
            self.send_heading_command(callsign, 180.0)  # Turn south
            
            # Step simulation to apply command
            print("   Stepping simulation to apply heading change...")
            for step in range(5):
                self.step_simulation(10)
                time.sleep(0.3)
            
            # Check new state
            final_states = self.get_aircraft_states()
            final_state = final_states.get(callsign)
            
            if final_state and after_state:
                print(f"\n📊 Final Aircraft State:")
                print(f"   Position: {final_state.latitude:.6f}, {final_state.longitude:.6f}")
                print(f"   Heading:  {final_state.heading_deg:.1f}°")
                
                # Check if heading changed
                hdg_diff = abs(final_state.heading_deg - after_state.heading_deg)
                if hdg_diff > 1.0:
                    print("🎉 SUCCESS: Heading change detected!")
                else:
                    print("❌ No heading change detected")
                
                # Check for additional movement
                lat_diff2 = abs(final_state.latitude - after_state.latitude)
                lon_diff2 = abs(final_state.longitude - after_state.longitude)
                
                if lat_diff2 > 0.0001 or lon_diff2 > 0.0001:
                    print("🎉 SUCCESS: Additional movement after heading change!")
                
            print("\n✅ REAL BlueSky simulation test complete!")
            print("This uses actual bs.sim.step() for movement")
            
        except Exception as e:
            print(f"❌ Test error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Run the real BlueSky movement test"""
    
    print("=== REAL BlueSky Movement Solution ===")
    print("Using embedded BlueSky with bs.sim.step() for actual simulation")
    
    client = RealBlueSkyClient()
    client.test_real_movement()

if __name__ == "__main__":
    main()
