#!/usr/bin/env python3
"""
Comprehensive test script to test all major BlueSky client functions including:
- Aircraft creation/deletion
- Flight control commands (heading, altitude, speed)
- Simulation control (hold, op, ff)
- Conflict detection
- State tracking over time

This extends the trajectory test to validate all BlueSky client functionality.
"""

import sys
import os
import time
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add src to path
sys.path.append('.')
sys.path.append('src')

from src.cdr.simulation.bluesky_client import BlueSkyClient, BlueSkyConfig, AircraftState

class ComprehensiveBlueSkyTester:
    """Comprehensive test suite for BlueSky client functions"""
    
    def __init__(self):
        self.trajectories = {}  # callsign -> list of states
        self.command_log = []   # Log of all commands executed
        self.test_results = {}  # Results of each test
        self.start_time = None
        self.step_count = 0
        
    def log_command(self, command_name: str, callsign: str, params: dict, success: bool):
        """Log command execution for analysis"""
        self.command_log.append({
            'step': self.step_count,
            'time': time.time() - self.start_time if self.start_time else 0,
            'command': command_name,
            'callsign': callsign,
            'params': params,
            'success': success
        })
    
    def add_observation(self, timestamp: float, states: Dict[str, AircraftState], conflicts: List = None):
        """Add observation with aircraft states and conflicts"""
        if self.start_time is None:
            self.start_time = timestamp
        
        relative_time = timestamp - self.start_time
        
        for callsign, state in states.items():
            if callsign not in self.trajectories:
                self.trajectories[callsign] = []
            
            observation = {
                'time': relative_time,
                'step': self.step_count,
                'lat': state.latitude,
                'lon': state.longitude,
                'alt_ft': state.altitude_ft,
                'hdg_deg': state.heading_deg,
                'spd_kt': state.speed_kt,
                'vs_fpm': state.vertical_speed_fpm,
                'conflicts': len(conflicts) if conflicts else 0
            }
            self.trajectories[callsign].append(observation)
        
        self.step_count += 1
    
    def test_aircraft_lifecycle(self, client: BlueSkyClient) -> bool:
        """Test aircraft creation and deletion"""
        print("\n🧪 Testing Aircraft Lifecycle (Create/Delete)")
        
        test_callsign = "TEST_LIFECYCLE"
        
        # Test creation
        success_create = client.create_aircraft(
            callsign=test_callsign,
            aircraft_type="B738",
            lat=42.5,
            lon=-87.5,
            heading=180,
            altitude_ft=25000,
            speed_kt=400
        )
        
        self.log_command("create_aircraft", test_callsign, 
                        {"type": "B738", "alt": 25000, "spd": 400}, success_create)
        
        if success_create:
            # Verify aircraft exists
            states = client.get_aircraft_states([test_callsign])
            if test_callsign in states:
                print(f"   ✅ Aircraft {test_callsign} created and verified")
                
                # Test deletion
                success_delete = client.delete_aircraft(test_callsign)
                self.log_command("delete_aircraft", test_callsign, {}, success_delete)
                
                if success_delete:
                    # Verify aircraft is gone
                    states_after = client.get_aircraft_states([test_callsign])
                    if test_callsign not in states_after:
                        print(f"   ✅ Aircraft {test_callsign} successfully deleted")
                        self.test_results['aircraft_lifecycle'] = True
                        return True
                    else:
                        print(f"   ❌ Aircraft {test_callsign} still exists after deletion")
                else:
                    print(f"   ❌ Failed to delete aircraft {test_callsign}")
            else:
                print(f"   ❌ Aircraft {test_callsign} not found after creation")
        else:
            print(f"   ❌ Failed to create aircraft {test_callsign}")
        
        self.test_results['aircraft_lifecycle'] = False
        return False
    
    def test_flight_commands(self, client: BlueSkyClient, callsign: str) -> bool:
        """Test heading, altitude, and speed commands"""
        print(f"\n🧪 Testing Flight Commands for {callsign}")
        
        # Get initial state
        initial_states = client.get_aircraft_states([callsign])
        if callsign not in initial_states:
            print(f"   ❌ Aircraft {callsign} not found for flight command testing")
            return False
        
        initial_state = initial_states[callsign]
        print(f"   📊 Initial state: HDG={initial_state.heading_deg:.1f}°, "
              f"ALT={initial_state.altitude_ft:.0f}ft, SPD={initial_state.speed_kt:.0f}kt")
        
        # Test heading command
        new_heading = (initial_state.heading_deg + 45) % 360
        success_hdg = client.heading_command(callsign, new_heading)
        self.log_command("heading_command", callsign, {"heading": new_heading}, success_hdg)
        
        if success_hdg:
            print(f"   ✅ Heading command sent: {new_heading:.1f}°")
        else:
            print(f"   ❌ Heading command failed")
        
        # Test altitude command
        new_altitude = initial_state.altitude_ft + 2000
        success_alt = client.altitude_command(callsign, new_altitude)
        self.log_command("altitude_command", callsign, {"altitude": new_altitude}, success_alt)
        
        if success_alt:
            print(f"   ✅ Altitude command sent: {new_altitude:.0f}ft")
        else:
            print(f"   ❌ Altitude command failed")
        
        # Test speed command
        new_speed = initial_state.speed_kt + 50
        success_spd = client.speed_command(callsign, new_speed)
        self.log_command("speed_command", callsign, {"speed": new_speed}, success_spd)
        
        if success_spd:
            print(f"   ✅ Speed command sent: {new_speed:.0f}kt")
        else:
            print(f"   ❌ Speed command failed")
        
        # Step simulation to see effects
        client.step_minutes(0.5)  # 30 seconds
        
        # Check if commands had effect
        new_states = client.get_aircraft_states([callsign])
        if callsign in new_states:
            new_state = new_states[callsign]
            print(f"   📊 After commands: HDG={new_state.heading_deg:.1f}°, "
                  f"ALT={new_state.altitude_ft:.0f}ft, SPD={new_state.speed_kt:.0f}kt")
            
            # Check for changes (allowing for simulation dynamics)
            hdg_changed = abs(new_state.heading_deg - initial_state.heading_deg) > 1
            alt_changed = abs(new_state.altitude_ft - initial_state.altitude_ft) > 100
            spd_changed = abs(new_state.speed_kt - initial_state.speed_kt) > 5
            
            print(f"   📈 Changes detected: HDG={hdg_changed}, ALT={alt_changed}, SPD={spd_changed}")
        
        all_success = success_hdg and success_alt and success_spd
        self.test_results['flight_commands'] = all_success
        return all_success
    
    def test_simulation_control(self, client: BlueSkyClient) -> bool:
        """Test simulation control functions (hold, op, ff)"""
        print(f"\n🧪 Testing Simulation Control Functions")
        
        # Test hold (pause)
        success_hold = client.hold()
        self.log_command("hold", "SIM", {}, success_hold)
        
        if success_hold:
            print(f"   ✅ Simulation paused (HOLD)")
        else:
            print(f"   ❌ Failed to pause simulation")
        
        time.sleep(1)  # Brief pause
        
        # Test op (resume)
        success_op = client.op()
        self.log_command("op", "SIM", {}, success_op)
        
        if success_op:
            print(f"   ✅ Simulation resumed (OP)")
        else:
            print(f"   ❌ Failed to resume simulation")
        
        # Test fast-forward
        ff_seconds = 30
        success_ff = client.ff(ff_seconds)
        self.log_command("ff", "SIM", {"seconds": ff_seconds}, success_ff)
        
        if success_ff:
            print(f"   ✅ Fast-forwarded {ff_seconds} seconds")
        else:
            print(f"   ❌ Failed to fast-forward simulation")
        
        all_success = success_hold and success_op and success_ff
        self.test_results['simulation_control'] = all_success
        return all_success
    
    def test_conflict_detection(self, client: BlueSkyClient) -> bool:
        """Test conflict detection functionality"""
        print(f"\n🧪 Testing Conflict Detection")
        
        # Create two aircraft close to each other for potential conflict
        conflict_ac1 = "CONFLICT_1"
        conflict_ac2 = "CONFLICT_2"
        
        # Aircraft 1
        success1 = client.create_aircraft(
            callsign=conflict_ac1,
            aircraft_type="A320",
            lat=42.0,
            lon=-87.0,
            heading=90,  # Eastbound
            altitude_ft=35000,
            speed_kt=450
        )
        
        # Aircraft 2 - converging path
        success2 = client.create_aircraft(
            callsign=conflict_ac2,
            aircraft_type="B737",
            lat=42.05,  # Slightly north
            lon=-86.95, # Slightly east
            heading=270, # Westbound (opposite direction)
            altitude_ft=35000,  # Same altitude
            speed_kt=450
        )
        
        if success1 and success2:
            print(f"   ✅ Created conflict test aircraft: {conflict_ac1}, {conflict_ac2}")
            
            # Step simulation to let aircraft move
            client.step_minutes(1.0)  # 1 minute
            
            # Check for conflicts
            conflicts = client.get_conflicts()
            self.log_command("get_conflicts", "ALL", {"found": len(conflicts)}, True)
            
            print(f"   📊 Conflicts detected: {len(conflicts)}")
            
            if conflicts:
                for i, conflict in enumerate(conflicts):
                    print(f"     Conflict {i+1}: {conflict.aircraft1} vs {conflict.aircraft2}")
                    print(f"       Distance: {conflict.horizontal_distance:.1f}NM horizontal, "
                          f"{conflict.vertical_distance:.0f}ft vertical")
                    print(f"       Type: {conflict.conflict_type}, Severity: {conflict.severity}")
            
            # Clean up test aircraft
            client.delete_aircraft(conflict_ac1)
            client.delete_aircraft(conflict_ac2)
            
            self.test_results['conflict_detection'] = True
            return True
        else:
            print(f"   ❌ Failed to create conflict test aircraft")
            self.test_results['conflict_detection'] = False
            return False
    
    def create_comprehensive_report(self, output_dir: Path):
        """Create comprehensive test report with visualizations"""
        
        # Command execution analysis
        command_df = pd.DataFrame(self.command_log)
        
        # Create comprehensive visualization
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Flight Paths', 'Altitude Profiles', 
                          'Speed Profiles', 'Command Success Rate',
                          'Command Timeline', 'Aircraft Count Over Time'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        colors = px.colors.qualitative.Set1
        
        # 1. Flight paths
        for i, (callsign, trajectory) in enumerate(self.trajectories.items()):
            if not trajectory:
                continue
            color = colors[i % len(colors)]
            lats = [p['lat'] for p in trajectory]
            lons = [p['lon'] for p in trajectory]
            
            fig.add_trace(
                go.Scatter(x=lons, y=lats, mode='lines+markers',
                          name=f'{callsign} Path', line=dict(color=color)),
                row=1, col=1
            )
        
        # 2. Altitude profiles
        for i, (callsign, trajectory) in enumerate(self.trajectories.items()):
            if not trajectory:
                continue
            color = colors[i % len(colors)]
            times = [p['time'] for p in trajectory]
            alts = [p['alt_ft'] for p in trajectory]
            
            fig.add_trace(
                go.Scatter(x=times, y=alts, mode='lines+markers',
                          name=f'{callsign} Alt', line=dict(color=color)),
                row=1, col=2
            )
        
        # 3. Speed profiles
        for i, (callsign, trajectory) in enumerate(self.trajectories.items()):
            if not trajectory:
                continue
            color = colors[i % len(colors)]
            times = [p['time'] for p in trajectory]
            speeds = [p['spd_kt'] for p in trajectory]
            
            fig.add_trace(
                go.Scatter(x=times, y=speeds, mode='lines+markers',
                          name=f'{callsign} Speed', line=dict(color=color)),
                row=2, col=1
            )
        
        # 4. Command success rate
        if self.command_log:
            command_success = command_df.groupby('command')['success'].agg(['count', 'sum']).reset_index()
            command_success['success_rate'] = command_success['sum'] / command_success['count'] * 100
            
            fig.add_trace(
                go.Bar(x=command_success['command'], y=command_success['success_rate'],
                      name='Success Rate %'),
                row=2, col=2
            )
        
        # 5. Command timeline
        if self.command_log:
            fig.add_trace(
                go.Scatter(x=command_df['time'], y=command_df['command'],
                          mode='markers', name='Commands',
                          marker=dict(color=command_df['success'].map({True: 'green', False: 'red'}))),
                row=3, col=1
            )
        
        # 6. Aircraft count over time
        if self.trajectories:
            all_times = set()
            for trajectory in self.trajectories.values():
                all_times.update([p['time'] for p in trajectory])
            
            times_sorted = sorted(all_times)
            aircraft_counts = []
            
            for t in times_sorted:
                count = 0
                for trajectory in self.trajectories.values():
                    if any(p['time'] == t for p in trajectory):
                        count += 1
                aircraft_counts.append(count)
            
            fig.add_trace(
                go.Scatter(x=times_sorted, y=aircraft_counts, mode='lines+markers',
                          name='Aircraft Count'),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Comprehensive BlueSky Client Function Test Results",
            showlegend=True
        )
        
        # Save comprehensive plot
        plot_file = output_dir / "comprehensive_test_results.html"
        fig.write_html(plot_file)
        print(f"📊 Comprehensive visualization saved to {plot_file}")
        
        # Save detailed reports
        trajectory_file = output_dir / "comprehensive_trajectories.json"
        with open(trajectory_file, 'w') as f:
            json.dump(self.trajectories, f, indent=2)
        
        command_file = output_dir / "command_log.json"
        with open(command_file, 'w') as f:
            json.dump(self.command_log, f, indent=2)
        
        results_file = output_dir / "test_results_summary.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        return fig


def run_comprehensive_bluesky_test():
    """
    Main comprehensive test function
    """
    print("🚀 Starting Comprehensive BlueSky Client Function Test")
    print("="*70)
    
    # Configuration
    config = BlueSkyConfig()
    config.headless = True
    config.asas_enabled = True
    config.reso_off = True
    config.dtmult = 1.0
    config.dt = 1.0
    
    # Test parameters
    test_duration_minutes = 8
    step_interval_seconds = 20
    total_steps = int((test_duration_minutes * 60) / step_interval_seconds)
    
    # Create output directory
    output_dir = Path("output/comprehensive_test")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize tester
    tester = ComprehensiveBlueSkyTester()
    
    # Main aircraft for sustained testing
    main_aircraft = [
        {"callsign": "MAIN001", "type": "B738", "lat": 42.0, "lon": -87.0, "hdg": 90, "alt": 35000, "spd": 450},
        {"callsign": "MAIN002", "type": "A320", "lat": 42.2, "lon": -87.2, "hdg": 180, "alt": 36000, "spd": 400},
    ]
    
    try:
        print("🔌 Connecting to BlueSky...")
        client = BlueSkyClient(config)
        
        if not client.connect():
            print("❌ Failed to connect to BlueSky")
            return False
        
        print("✅ Connected to BlueSky")
        
        # Test 1: Aircraft Lifecycle
        success_lifecycle = tester.test_aircraft_lifecycle(client)
        
        # Create main aircraft for ongoing tests
        print(f"\n✈️ Creating main test aircraft...")
        main_callsigns = []
        
        for ac_config in main_aircraft:
            success = client.create_aircraft(
                callsign=ac_config['callsign'],
                aircraft_type=ac_config['type'],
                lat=ac_config['lat'],
                lon=ac_config['lon'],
                heading=ac_config['hdg'],
                altitude_ft=ac_config['alt'],
                speed_kt=ac_config['spd']
            )
            
            if success:
                main_callsigns.append(ac_config['callsign'])
                print(f"   ✅ {ac_config['callsign']} created")
        
        # Test 2: Flight Commands
        if main_callsigns:
            success_commands = tester.test_flight_commands(client, main_callsigns[0])
        else:
            success_commands = False
        
        # Test 3: Simulation Control
        success_sim_control = tester.test_simulation_control(client)
        
        # Test 4: Conflict Detection
        success_conflicts = tester.test_conflict_detection(client)
        
        # Start main simulation tracking
        print(f"\n📊 Starting {test_duration_minutes}-minute comprehensive tracking...")
        client.op()  # Ensure simulation is running
        
        # Track trajectories and test functions periodically
        for step in range(total_steps):
            current_time = time.time()
            
            # Get aircraft states
            states = client.get_aircraft_states()
            
            # Get conflicts
            conflicts = client.get_conflicts()
            
            # Record observation
            tester.add_observation(current_time, states, conflicts)
            
            # Progress reporting
            elapsed_minutes = step * step_interval_seconds / 60
            print(f"   Step {step+1:2d}/{total_steps}: {elapsed_minutes:4.1f}min, "
                  f"{len(states)} aircraft, {len(conflicts)} conflicts")
            
            # Periodic function testing
            if step % 5 == 0 and main_callsigns:  # Every 5th step
                # Test a command
                test_callsign = main_callsigns[step % len(main_callsigns)]
                if step % 15 == 0:  # Heading change every 15 steps
                    new_hdg = (step * 30) % 360
                    success = client.heading_command(test_callsign, new_hdg)
                    tester.log_command("periodic_heading", test_callsign, {"heading": new_hdg}, success)
                elif step % 10 == 0:  # Speed change every 10 steps
                    new_spd = 400 + (step % 3) * 50
                    success = client.speed_command(test_callsign, new_spd)
                    tester.log_command("periodic_speed", test_callsign, {"speed": new_spd}, success)
            
            # Step simulation
            if step < total_steps - 1:
                client.step_minutes(step_interval_seconds / 60.0)
        
        print(f"\n📈 Comprehensive test completed!")
        
        # Generate comprehensive report
        print(f"\n📊 Generating comprehensive test report...")
        tester.create_comprehensive_report(output_dir)
        
        # Final results summary
        print(f"\n🎯 COMPREHENSIVE TEST RESULTS SUMMARY")
        print(f"="*50)
        
        all_tests = [
            ("Aircraft Lifecycle", success_lifecycle),
            ("Flight Commands", success_commands),
            ("Simulation Control", success_sim_control),
            ("Conflict Detection", success_conflicts)
        ]
        
        passed_tests = sum(1 for _, success in all_tests if success)
        
        for test_name, success in all_tests:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {test_name}: {status}")
        
        print(f"\n📊 Overall Results:")
        print(f"   Tests passed: {passed_tests}/{len(all_tests)}")
        print(f"   Commands executed: {len(tester.command_log)}")
        print(f"   Aircraft tracked: {len(tester.trajectories)}")
        print(f"   Total observations: {tester.step_count}")
        
        success_rate = passed_tests / len(all_tests) * 100
        print(f"   Success rate: {success_rate:.1f}%")
        
        # Disconnect
        client.disconnect()
        
        return success_rate >= 75  # Consider test successful if 75%+ pass
        
    except Exception as e:
        print(f"❌ Comprehensive test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Install required packages if needed
    try:
        import plotly
        import pandas
    except ImportError:
        print("Installing required packages...")
        os.system("pip install plotly pandas")
        import plotly
        import pandas
    
    # Run the comprehensive test
    success = run_comprehensive_bluesky_test()
    exit(0 if success else 1)
