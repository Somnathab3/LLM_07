#!/usr/bin/env python3
"""
Diagnostic test to specifically investigate altitude and speed tracking issues.
This test focuses on understanding why aircraft lose altitude and speed rapidly.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add src to path
sys.path.append('.')
sys.path.append('src')

from src.cdr.simulation.bluesky_client import BlueSkyClient, BlueSkyConfig, AircraftState

def diagnose_altitude_speed_issue():
    """
    Focused test to diagnose altitude and speed tracking problems
    """
    print("🔍 ALTITUDE & SPEED DIAGNOSTIC TEST")
    print("="*50)
    
    # Configuration - slower simulation for detailed analysis
    config = BlueSkyConfig()
    config.headless = True
    config.asas_enabled = False  # Disable ASAS to avoid interference
    config.reso_off = True       # Disable automatic resolution
    config.dtmult = 0.5         # Slow down simulation for debugging
    config.dt = 0.1             # Small time steps
    
    # Test parameters
    test_duration_seconds = 300  # 5 minutes
    measurement_interval = 5     # Every 5 seconds
    measurements = test_duration_seconds // measurement_interval
    
    # Output
    output_dir = Path("output/altitude_speed_diagnostic")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        print("🔌 Connecting to BlueSky...")
        client = BlueSkyClient(config)
        
        if not client.connect():
            print("❌ Failed to connect to BlueSky")
            return False
        
        print("✅ Connected to BlueSky")
        
        # Create a single test aircraft for detailed monitoring
        callsign = "DIAG001"
        initial_alt = 35000
        initial_spd = 450
        
        print(f"\n✈️ Creating diagnostic aircraft {callsign}...")
        success = client.create_aircraft(
            callsign=callsign,
            aircraft_type="B738",
            lat=42.0,
            lon=-87.0,
            heading=90,
            altitude_ft=initial_alt,
            speed_kt=initial_spd
        )
        
        if not success:
            print(f"❌ Failed to create aircraft {callsign}")
            return False
        
        print(f"✅ Aircraft {callsign} created")
        
        # Wait for aircraft to initialize
        time.sleep(2)
        
        # Issue explicit maintenance commands
        print(f"🎯 Setting aircraft for stable cruise flight...")
        
        # Set altitude with explicit command
        alt_cmd_success = client.altitude_command(callsign, initial_alt)
        print(f"   Altitude command: {'✅' if alt_cmd_success else '❌'}")
        
        # Set speed with explicit command  
        spd_cmd_success = client.speed_command(callsign, initial_spd)
        print(f"   Speed command: {'✅' if spd_cmd_success else '❌'}")
        
        # Set heading to maintain
        hdg_cmd_success = client.heading_command(callsign, 90)
        print(f"   Heading command: {'✅' if hdg_cmd_success else '❌'}")
        
        # Start simulation
        print(f"\n▶️ Starting diagnostic simulation...")
        client.op()
        
        # Data collection
        diagnostic_data = []
        
        print(f"📊 Collecting detailed measurements every {measurement_interval}s for {test_duration_seconds}s...")
        
        for measurement in range(measurements):
            current_time = time.time()
            elapsed = measurement * measurement_interval
            
            # Get aircraft state
            states = client.get_aircraft_states([callsign])
            
            if callsign in states:
                state = states[callsign]
                
                # Detailed logging
                data_point = {
                    'time_s': elapsed,
                    'measurement': measurement,
                    'lat': state.latitude,
                    'lon': state.longitude,
                    'altitude_ft': state.altitude_ft,
                    'heading_deg': state.heading_deg,
                    'speed_kt': state.speed_kt,
                    'vertical_speed_fpm': state.vertical_speed_fpm
                }
                
                diagnostic_data.append(data_point)
                
                # Console output
                print(f"   T+{elapsed:3d}s: ALT={state.altitude_ft:7.1f}ft, SPD={state.speed_kt:6.1f}kt, "
                      f"VS={state.vertical_speed_fpm:6.1f}fpm, HDG={state.heading_deg:5.1f}°")
                
                # Alert on significant changes
                if len(diagnostic_data) > 1:
                    prev = diagnostic_data[-2]
                    alt_change = state.altitude_ft - prev['altitude_ft']
                    spd_change = state.speed_kt - prev['speed_kt']
                    
                    if abs(alt_change) > 500:  # More than 500ft change
                        print(f"     ⚠️ Large altitude change: {alt_change:+.1f}ft")
                    
                    if abs(spd_change) > 25:   # More than 25kt change
                        print(f"     ⚠️ Large speed change: {spd_change:+.1f}kt")
                    
                    if state.vertical_speed_fpm < -1000:  # Rapid descent
                        print(f"     🚨 RAPID DESCENT DETECTED: {state.vertical_speed_fpm:.1f}fpm")
                        
                        # Try to correct with altitude command
                        print(f"     🎯 Issuing corrective altitude command...")
                        client.altitude_command(callsign, initial_alt)
                
                # Issue periodic maintenance commands every 60 seconds
                if elapsed > 0 and elapsed % 60 == 0:
                    print(f"   🔧 Issuing maintenance commands at T+{elapsed}s...")
                    client.altitude_command(callsign, initial_alt)
                    client.speed_command(callsign, initial_spd)
                    
            else:
                print(f"   ❌ Aircraft {callsign} not found at T+{elapsed}s")
                break
            
            # Step simulation forward
            if measurement < measurements - 1:
                client.step_minutes(measurement_interval / 60.0)
        
        print(f"\n📊 Diagnostic data collection completed")
        
        # Save diagnostic data
        data_file = output_dir / "diagnostic_data.json"
        with open(data_file, 'w') as f:
            json.dump(diagnostic_data, f, indent=2)
        
        print(f"💾 Diagnostic data saved to {data_file}")
        
        # Analysis
        print(f"\n🔍 DIAGNOSTIC ANALYSIS:")
        
        if diagnostic_data:
            initial_data = diagnostic_data[0]
            final_data = diagnostic_data[-1]
            
            # Calculate changes
            alt_change = final_data['altitude_ft'] - initial_data['altitude_ft']
            spd_change = final_data['speed_kt'] - initial_data['speed_kt']
            time_span = final_data['time_s']
            
            print(f"   📊 Initial state: ALT={initial_data['altitude_ft']:.1f}ft, SPD={initial_data['speed_kt']:.1f}kt")
            print(f"   📊 Final state:   ALT={final_data['altitude_ft']:.1f}ft, SPD={final_data['speed_kt']:.1f}kt")
            print(f"   📊 Changes over {time_span}s:")
            print(f"     Altitude: {alt_change:+.1f}ft ({alt_change/time_span*60:.1f}ft/min)")
            print(f"     Speed: {spd_change:+.1f}kt")
            
            # Identify issues
            issues = []
            if abs(alt_change) > 1000:
                issues.append(f"Large altitude change: {alt_change:+.1f}ft")
            if abs(spd_change) > 50:
                issues.append(f"Large speed change: {spd_change:+.1f}kt")
            if final_data['altitude_ft'] < 1000:
                issues.append("Aircraft near ground level")
            if final_data['speed_kt'] < 100:
                issues.append("Aircraft very slow")
            
            if issues:
                print(f"\n   ❌ ISSUES DETECTED:")
                for issue in issues:
                    print(f"     • {issue}")
            else:
                print(f"\n   ✅ NO MAJOR ISSUES DETECTED")
            
            # Stability analysis
            altitude_stable = all(abs(d['altitude_ft'] - initial_alt) < 2000 for d in diagnostic_data)
            speed_stable = all(abs(d['speed_kt'] - initial_spd) < 100 for d in diagnostic_data)
            
            print(f"\n   📈 STABILITY ANALYSIS:")
            print(f"     Altitude stable: {'✅' if altitude_stable else '❌'}")
            print(f"     Speed stable: {'✅' if speed_stable else '❌'}")
            
        # Create simple plot data for manual inspection
        plot_data = {
            'times': [d['time_s'] for d in diagnostic_data],
            'altitudes': [d['altitude_ft'] for d in diagnostic_data],
            'speeds': [d['speed_kt'] for d in diagnostic_data],
            'vertical_speeds': [d['vertical_speed_fpm'] for d in diagnostic_data]
        }
        
        plot_file = output_dir / "plot_data.json"
        with open(plot_file, 'w') as f:
            json.dump(plot_data, f, indent=2)
        
        print(f"📈 Plot data saved to {plot_file}")
        
        # Disconnect
        client.disconnect()
        
        return len(diagnostic_data) > 0
        
    except Exception as e:
        print(f"❌ Diagnostic test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = diagnose_altitude_speed_issue()
    exit(0 if success else 1)
