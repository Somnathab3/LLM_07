#!/usr/bin/env python3
"""
Test script to track aircraft trajectories and detect if aircraft are being reset/disappearing.
Creates multiple aircraft and tracks their positions over time to visualize the persistence issue.
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add src to path
sys.path.append('.')
sys.path.append('src')

from src.cdr.simulation.bluesky_client import BlueSkyClient, BlueSkyConfig, AircraftState

class TrajectoryTracker:
    """Track aircraft trajectories over time to detect persistence issues"""
    
    def __init__(self):
        self.trajectories = {}  # callsign -> list of (time, state) tuples
        self.start_time = None
        self.total_steps = 0
        
    def add_observation(self, timestamp: float, states: Dict[str, AircraftState]):
        """Add aircraft states observation at given timestamp"""
        if self.start_time is None:
            self.start_time = timestamp
        
        relative_time = timestamp - self.start_time
        
        for callsign, state in states.items():
            if callsign not in self.trajectories:
                self.trajectories[callsign] = []
            
            self.trajectories[callsign].append({
                'time': relative_time,
                'step': self.total_steps,
                'lat': state.latitude,
                'lon': state.longitude,
                'alt_ft': state.altitude_ft,
                'hdg_deg': state.heading_deg,
                'spd_kt': state.speed_kt,
                'vs_fpm': state.vertical_speed_fpm
            })
        
        self.total_steps += 1
        
    def get_missing_aircraft_report(self) -> Dict:
        """Analyze which aircraft went missing and when"""
        report = {
            'total_aircraft': len(self.trajectories),
            'aircraft_details': {},
            'persistence_issues': []
        }
        
        for callsign, trajectory in self.trajectories.items():
            first_seen = trajectory[0]['step'] if trajectory else None
            last_seen = trajectory[-1]['step'] if trajectory else None
            total_observations = len(trajectory)
            
            # Check for gaps in observations (aircraft disappearing)
            expected_observations = self.total_steps - first_seen if first_seen is not None else 0
            missing_observations = expected_observations - total_observations
            
            aircraft_info = {
                'first_seen_step': first_seen,
                'last_seen_step': last_seen,
                'total_observations': total_observations,
                'expected_observations': expected_observations,
                'missing_observations': missing_observations,
                'persistence_rate': total_observations / expected_observations if expected_observations > 0 else 0.0
            }
            
            report['aircraft_details'][callsign] = aircraft_info
            
            # Flag aircraft with persistence issues
            if missing_observations > 0:
                report['persistence_issues'].append({
                    'callsign': callsign,
                    'missing_steps': missing_observations,
                    'persistence_rate': aircraft_info['persistence_rate']
                })
        
        return report
    
    def create_trajectory_plot(self, output_file: str = None):
        """Create interactive plotly visualization of aircraft trajectories"""
        if not self.trajectories:
            print("❌ No trajectory data to plot")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Flight Paths (Lat/Lon)', 'Altitude vs Time', 
                          'Speed vs Time', 'Aircraft Presence Over Time'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Color palette for different aircraft
        colors = px.colors.qualitative.Set1
        
        # Track aircraft presence for persistence analysis
        presence_data = []
        
        for i, (callsign, trajectory) in enumerate(self.trajectories.items()):
            if not trajectory:
                continue
                
            color = colors[i % len(colors)]
            
            # Extract data
            times = [point['time'] for point in trajectory]
            lats = [point['lat'] for point in trajectory]
            lons = [point['lon'] for point in trajectory]
            alts = [point['alt_ft'] for point in trajectory]
            speeds = [point['spd_kt'] for point in trajectory]
            steps = [point['step'] for point in trajectory]
            
            # Flight path (lat/lon)
            fig.add_trace(
                go.Scatter(x=lons, y=lats, mode='lines+markers',
                          name=f'{callsign} Path', line=dict(color=color),
                          hovertemplate=f'{callsign}<br>Lat: %{{y:.6f}}<br>Lon: %{{x:.6f}}<extra></extra>'),
                row=1, col=1
            )
            
            # Altitude vs time
            fig.add_trace(
                go.Scatter(x=times, y=alts, mode='lines+markers',
                          name=f'{callsign} Alt', line=dict(color=color),
                          hovertemplate=f'{callsign}<br>Time: %{{x:.1f}}s<br>Alt: %{{y:.0f}}ft<extra></extra>'),
                row=1, col=2
            )
            
            # Speed vs time
            fig.add_trace(
                go.Scatter(x=times, y=speeds, mode='lines+markers',
                          name=f'{callsign} Speed', line=dict(color=color),
                          hovertemplate=f'{callsign}<br>Time: %{{x:.1f}}s<br>Speed: %{{y:.0f}}kt<extra></extra>'),
                row=2, col=1
            )
            
            # Aircraft presence (binary: 1 if present, 0 if missing)
            all_steps = list(range(self.total_steps))
            presence = [1 if step in steps else 0 for step in all_steps]
            presence_times = [step * 30 for step in all_steps]  # Assuming 30s steps
            
            fig.add_trace(
                go.Scatter(x=presence_times, y=[i + presence[j] for j in range(len(presence))],
                          mode='lines', name=f'{callsign} Presence', line=dict(color=color),
                          hovertemplate=f'{callsign}<br>Time: %{{x:.1f}}s<br>Present: %{{y}}<extra></extra>'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Aircraft Trajectory Analysis - Persistence Testing",
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Longitude", row=1, col=1)
        fig.update_yaxes(title_text="Latitude", row=1, col=1)
        
        fig.update_xaxes(title_text="Time (seconds)", row=1, col=2)
        fig.update_yaxes(title_text="Altitude (feet)", row=1, col=2)
        
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Speed (knots)", row=2, col=1)
        
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=2)
        fig.update_yaxes(title_text="Aircraft Index + Presence", row=2, col=2)
        
        # Save plot
        if output_file:
            fig.write_html(output_file)
            print(f"✅ Trajectory plot saved to {output_file}")
        
        # Show plot
        fig.show()
        
        return fig


def test_aircraft_trajectories():
    """
    Main test function to create aircraft and track their trajectories
    """
    print("🚀 Starting Aircraft Trajectory Persistence Test")
    print("="*60)
    
    # Configuration
    config = BlueSkyConfig()
    config.headless = True
    config.asas_enabled = True
    config.reso_off = True  # Disable auto-resolution
    config.dtmult = 1.0     # Real-time for detailed tracking
    config.dt = 1.0         # 1 second steps
    
    # Test parameters
    test_duration_minutes = 10
    step_interval_seconds = 30  # Take measurement every 30 seconds
    total_steps = int((test_duration_minutes * 60) / step_interval_seconds)
    
    # Create output directory
    output_dir = Path("output/trajectory_test")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize tracker
    tracker = TrajectoryTracker()
    
    # Aircraft to create - with proper cruise configuration
    aircraft_config = [
        # Ownship
        {"callsign": "OWN001", "type": "B738", "lat": 42.0, "lon": -87.0, "hdg": 90, "alt": 35000, "spd": 450},
        # Intruders - creating converging scenario
        {"callsign": "INT001", "type": "A320", "lat": 42.2, "lon": -87.5, "hdg": 180, "alt": 35000, "spd": 400},
        {"callsign": "INT002", "type": "B777", "lat": 41.8, "lon": -86.5, "hdg": 270, "alt": 36000, "spd": 480},
        {"callsign": "INT003", "type": "A330", "lat": 42.1, "lon": -87.2, "hdg": 45, "alt": 34000, "spd": 420},
        {"callsign": "INT004", "type": "B767", "lat": 41.9, "lon": -86.8, "hdg": 135, "alt": 37000, "spd": 460},
    ]
    
    try:
        # Initialize BlueSky client
        print("🔌 Connecting to BlueSky...")
        client = BlueSkyClient(config)
        
        if not client.connect():
            print("❌ Failed to connect to BlueSky")
            return False
        
        print("✅ Connected to BlueSky")
        
        # Create all aircraft
        print(f"\n✈️ Creating {len(aircraft_config)} aircraft...")
        created_aircraft = []
        
        for ac_config in aircraft_config:
            print(f"   Creating {ac_config['callsign']}...")
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
                created_aircraft.append(ac_config['callsign'])
                print(f"   ✅ {ac_config['callsign']} created")
                
                # IMPORTANT: Set aircraft to maintain altitude (level flight)
                # This prevents automatic descent that causes crashes
                print(f"   🎯 Setting {ac_config['callsign']} to maintain level flight...")
                
                # Set altitude command to maintain current altitude
                alt_success = client.altitude_command(ac_config['callsign'], ac_config['alt'])
                
                # Set speed to maintain current speed  
                spd_success = client.speed_command(ac_config['callsign'], ac_config['spd'])
                
                # Set heading to maintain current heading
                hdg_success = client.heading_command(ac_config['callsign'], ac_config['hdg'])
                
                if alt_success and spd_success and hdg_success:
                    print(f"   ✅ {ac_config['callsign']} configured for stable cruise flight")
                else:
                    print(f"   ⚠️ Warning: Some commands failed for {ac_config['callsign']}")
                    
            else:
                print(f"   ❌ Failed to create {ac_config['callsign']}")
        
        print(f"✅ Successfully created {len(created_aircraft)} aircraft: {created_aircraft}")
        
        # Give aircraft time to stabilize after commands
        print("   ⏱️ Allowing aircraft to stabilize...")
        client.step_minutes(0.5)  # 30 seconds to stabilize
        
        # Start simulation
        print(f"\n▶️ Starting simulation...")
        client.op()  # Start simulation
        
        print(f"📊 Tracking trajectories for {test_duration_minutes} minutes ({total_steps} steps)...")
        print(f"   Taking measurements every {step_interval_seconds} seconds")
        
        # Track trajectories
        for step in range(total_steps):
            current_time = time.time()
            
            # Get aircraft states
            states = client.get_aircraft_states()
            
            # Record observation
            tracker.add_observation(current_time, states)
            
            # Progress reporting
            elapsed_minutes = step * step_interval_seconds / 60
            aircraft_count = len(states)
            
            print(f"   Step {step+1:2d}/{total_steps}: {elapsed_minutes:4.1f}min elapsed, "
                  f"{aircraft_count} aircraft detected: {list(states.keys())}")
            
            # Detailed state monitoring every 5th step
            if step % 5 == 0:  
                for callsign, state in states.items():
                    print(f"     {callsign}: lat={state.latitude:.4f}, lon={state.longitude:.4f}, "
                          f"alt={state.altitude_ft:.0f}ft, hdg={state.heading_deg:.1f}°, spd={state.speed_kt:.0f}kt, vs={state.vertical_speed_fpm:.0f}fpm")
                    
                    # ALTITUDE SAFETY CHECK: If aircraft is descending too fast, issue level-off command
                    if state.altitude_ft < 10000 and state.vertical_speed_fpm < -500:
                        print(f"     ⚠️ {callsign} descending rapidly! Issuing level-off command...")
                        # Command aircraft to level off at current altitude
                        client.altitude_command(callsign, max(state.altitude_ft, 15000))
                        # Reduce vertical speed
                        client.speed_command(callsign, max(state.speed_kt, 250))
                        
                    # SPEED SAFETY CHECK: If aircraft is too slow, increase speed
                    elif state.speed_kt < 200:
                        print(f"     ⚠️ {callsign} too slow! Increasing speed...")
                        client.speed_command(callsign, 350)
            
            # Check for missing aircraft
            missing_aircraft = set(created_aircraft) - set(states.keys())
            if missing_aircraft:
                print(f"   ⚠️  Missing aircraft detected: {missing_aircraft}")
            
            # Periodic altitude maintenance commands (every 10 steps = 5 minutes)
            if step % 10 == 0 and step > 0:
                print(f"   🎯 Issuing altitude maintenance commands...")
                for callsign in created_aircraft:
                    if callsign in states:
                        state = states[callsign]
                        target_alt = 35000 if "OWN" in callsign else (34000 + (hash(callsign) % 4) * 1000)
                        client.altitude_command(callsign, target_alt)
            
            # Step simulation forward
            if step < total_steps - 1:  # Don't step after last measurement
                client.step_minutes(step_interval_seconds / 60.0)
        
        print(f"\n📈 Trajectory tracking completed!")
        
        # Analyze results
        print(f"\n🔍 Analyzing trajectory data...")
        report = tracker.get_missing_aircraft_report()
        
        print(f"📊 TRAJECTORY ANALYSIS REPORT")
        print(f"="*50)
        print(f"Total aircraft created: {len(created_aircraft)}")
        print(f"Total aircraft tracked: {report['total_aircraft']}")
        print(f"Total simulation steps: {tracker.total_steps}")
        
        print(f"\n📋 Aircraft Details:")
        for callsign, details in report['aircraft_details'].items():
            persistence = details['persistence_rate'] * 100
            print(f"  {callsign}:")
            print(f"    First seen: step {details['first_seen_step']}")
            print(f"    Last seen: step {details['last_seen_step']}")
            print(f"    Observations: {details['total_observations']}/{details['expected_observations']}")
            print(f"    Persistence rate: {persistence:.1f}%")
            
            if details['missing_observations'] > 0:
                print(f"    ⚠️  MISSING {details['missing_observations']} observations!")
        
        if report['persistence_issues']:
            print(f"\n❌ PERSISTENCE ISSUES DETECTED:")
            for issue in report['persistence_issues']:
                print(f"  {issue['callsign']}: missing {issue['missing_steps']} steps "
                      f"(persistence: {issue['persistence_rate']*100:.1f}%)")
        else:
            print(f"\n✅ NO PERSISTENCE ISSUES - All aircraft maintained throughout simulation")
        
        # Save raw trajectory data
        trajectory_file = output_dir / "trajectory_data.json"
        with open(trajectory_file, 'w') as f:
            json.dump(tracker.trajectories, f, indent=2)
        print(f"💾 Raw trajectory data saved to {trajectory_file}")
        
        # Save analysis report
        report_file = output_dir / "analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Analysis report saved to {report_file}")
        
        # Create trajectory plot
        print(f"\n📊 Creating trajectory visualization...")
        plot_file = output_dir / "trajectory_plot.html"
        tracker.create_trajectory_plot(str(plot_file))
        
        # Disconnect
        client.disconnect()
        
        # Final summary
        print(f"\n🎯 TEST SUMMARY:")
        if report['persistence_issues']:
            print(f"❌ FAILED: Aircraft persistence issues detected")
            print(f"   {len(report['persistence_issues'])} aircraft had missing observations")
            return False
        else:
            print(f"✅ PASSED: All aircraft persisted throughout simulation")
            return True
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
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
    
    # Run the test
    success = test_aircraft_trajectories()
    exit(0 if success else 1)
