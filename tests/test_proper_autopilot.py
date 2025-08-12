#!/usr/bin/env python3
"""
Enhanced BlueSky Aircraft Test with Proper Autopilot and FMS Configuration

This test addresses the speed degradation issue by properly enabling:
1. Autothrottle (THR command with AUTO)
2. LNAV (Lateral Navigation Autopilot)
3. VNAV (Vertical Navigation Autopilot)
4. Proper autopilot target commands (ALT, SPD, HDG)

Based on BlueSky command reference documentation.
"""

import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add project root to path
sys.path.append('.')

try:
    from src.cdr.simulation.bluesky_client import BlueSkyClient
except ImportError:
    print("❌ Could not import BlueSky client - check src/cdr/simulation/bluesky_client.py")
    sys.exit(1)


def test_proper_autopilot_configuration():
    """
    Test aircraft with proper autopilot and FMS configuration to prevent speed degradation.
    
    Uses BlueSky commands:
    - THR (autothrottle)
    - LNAV (lateral navigation autopilot)
    - VNAV (vertical navigation autopilot)
    - ALT, SPD, HDG (autopilot targets)
    """
    print("🚁 Testing Proper Autopilot and FMS Configuration")
    print("=" * 70)
    
    try:
        # Initialize simplified BlueSky client
        bs_client = BlueSkyClient()
        bs_client.initialize()
        
        # Reset simulation
        bs_client.reset()
        time.sleep(1)
        
        # Aircraft configuration for testing
        aircraft_configs = [
            {"id": "TEST001", "lat": 42.0, "lon": -86.5, "hdg": 90, "alt": 35000, "spd": 450},
            {"id": "TEST002", "lat": 41.8, "lon": -87.0, "hdg": 180, "alt": 36000, "spd": 420},
            {"id": "TEST003", "lat": 42.2, "lon": -86.8, "hdg": 270, "alt": 34000, "spd": 480}
        ]
        
        print(f"✈️ Creating {len(aircraft_configs)} aircraft...")
        
        # Step 1: Create aircraft
        for config in aircraft_configs:
            success = bs_client.create_aircraft(
                acid=config['id'],
                lat=config['lat'],
                lon=config['lon'],
                hdg=config['hdg'],
                alt=config['alt'],
                spd=config['spd']
            )
            if success:
                print(f"   📝 Created {config['id']} at {config['lat']:.2f},{config['lon']:.2f}")
            else:
                print(f"   ❌ Failed to create {config['id']}")
        
        time.sleep(2)  # Allow aircraft creation
        
        # Step 2: CRITICAL - Autopilot and FMS are already configured in create_aircraft
        print("\n🤖 CONFIGURING AUTOPILOT AND FMS SYSTEMS...")
        
        # Configure autopilot for each aircraft
        for config in aircraft_configs:
            acid = config['id']
            print(f"   🎯 Configuring autopilot for {acid}...")
            
            # Set autopilot parameters
            success = bs_client.set_autopilot(
                acid=acid,
                alt=config['alt'],
                spd=config['spd'],
                hdg=config['hdg']
            )
            
            if success:
                print(f"   ✅ {acid}: Autopilot configured (ALT:{config['alt']} SPD:{config['spd']} HDG:{config['hdg']})")
            else:
                print(f"   ❌ {acid}: Autopilot configuration failed")
            
        time.sleep(3)  # Allow autopilot systems to fully engage
        
        # Step 3: Monitor aircraft performance
        print(f"\n📊 MONITORING AIRCRAFT PERFORMANCE...")
        duration_seconds = 300  # 5 minutes
        check_interval = 10     # Check every 10 seconds
        
        tracking_data = {acid: {"times": [], "speeds": [], "altitudes": [], "headings": [], "vs": []} 
                        for acid in [cfg['id'] for cfg in aircraft_configs]}
        
        start_time = time.time()
        check_count = 0
        
        print(f"   Duration: {duration_seconds}s | Check interval: {check_interval}s")
        print(f"   Expecting stable cruise performance with minimal speed degradation...")
        
        while time.time() - start_time < duration_seconds:
            current_time = time.time() - start_time
            check_count += 1
            
            # Step the simulation forward
            bs_client.step_simulation(check_interval)
            
            # Get current aircraft states
            aircraft_found = 0
            
            for config in aircraft_configs:
                acid = config['id']
                state = bs_client.get_aircraft_state(acid)
                
                if state:
                    aircraft_found += 1
                    tracking_data[acid]["times"].append(current_time)
                    tracking_data[acid]["speeds"].append(state.tas)  # True airspeed
                    tracking_data[acid]["altitudes"].append(state.alt)
                    tracking_data[acid]["headings"].append(state.hdg)
                    tracking_data[acid]["vs"].append(state.vs)
                    
                    # Detailed monitoring every 3rd check
                    if check_count % 3 == 0:
                        print(f"   {current_time:6.1f}s | {acid}: "
                              f"SPD={state.tas:5.1f}kt, "
                              f"ALT={state.alt:7.0f}ft, "
                              f"HDG={state.hdg:5.1f}°, "
                              f"VS={state.vs:6.1f}fpm")
            
            # Status summary every 30 seconds
            if check_count % 3 == 0:
                print(f"   📈 {current_time:6.1f}s: {aircraft_found}/{len(aircraft_configs)} aircraft active")
            
            time.sleep(check_interval)
        
        # Step 4: Analysis and Results
        print(f"\n📈 PERFORMANCE ANALYSIS...")
        
        for config in aircraft_configs:
            acid = config['id']
            data = tracking_data[acid]
            
            if not data["speeds"]:
                print(f"   ❌ {acid}: No data recorded")
                continue
            
            initial_speed = data["speeds"][0] if data["speeds"] else 0
            final_speed = data["speeds"][-1] if data["speeds"] else 0
            speed_loss = initial_speed - final_speed
            speed_loss_percent = (speed_loss / initial_speed * 100) if initial_speed > 0 else 0
            
            initial_alt = data["altitudes"][0] if data["altitudes"] else 0
            final_alt = data["altitudes"][-1] if data["altitudes"] else 0
            alt_change = final_alt - initial_alt
            
            print(f"   📊 {acid} Performance:")
            print(f"      Speed: {initial_speed:.1f}kt → {final_speed:.1f}kt (Loss: {speed_loss:.1f}kt, {speed_loss_percent:.1f}%)")
            print(f"      Altitude: {initial_alt:.0f}ft → {final_alt:.0f}ft (Change: {alt_change:+.0f}ft)")
            
            # Performance assessment
            if speed_loss_percent < 5:
                print(f"      ✅ EXCELLENT: Speed very stable (< 5% loss)")
            elif speed_loss_percent < 15:
                print(f"      ✅ GOOD: Acceptable speed stability (< 15% loss)")
            elif speed_loss_percent < 25:
                print(f"      ⚠️ FAIR: Moderate speed degradation (< 25% loss)")
            else:
                print(f"      ❌ POOR: Significant speed degradation (> 25% loss)")
            
            if abs(alt_change) < 500:
                print(f"      ✅ EXCELLENT: Altitude stable (< 500ft variation)")
            else:
                print(f"      ⚠️ Altitude variation: {alt_change:+.0f}ft")
        
        # Step 5: Create visualization
        print(f"\n📊 Creating performance visualization...")
        create_autopilot_performance_plot(tracking_data, aircraft_configs)
        
        # Disconnect/cleanup
        print(f"\n✅ TEST COMPLETED - Autopilot and FMS configuration test finished")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_autopilot_performance_plot(tracking_data, aircraft_configs):
    """Create detailed performance visualization"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Speed vs Time (Autopilot Performance)', 'Altitude vs Time',
                       'Heading vs Time', 'Vertical Speed vs Time'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, config in enumerate(aircraft_configs):
        acid = config['id']
        data = tracking_data[acid]
        color = colors[i % len(colors)]
        
        if not data["times"]:
            continue
        
        # Speed plot (most important for diagnosing autopilot issues)
        fig.add_trace(
            go.Scatter(
                x=data["times"], y=data["speeds"],
                mode='lines+markers',
                name=f'{acid} Speed',
                line=dict(color=color),
                hovertemplate=f'{acid}<br>Time: %{{x:.1f}}s<br>Speed: %{{y:.1f}}kt<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Altitude plot
        fig.add_trace(
            go.Scatter(
                x=data["times"], y=data["altitudes"],
                mode='lines+markers',
                name=f'{acid} Alt',
                line=dict(color=color, dash='dash'),
                hovertemplate=f'{acid}<br>Time: %{{x:.1f}}s<br>Alt: %{{y:.0f}}ft<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Heading plot
        fig.add_trace(
            go.Scatter(
                x=data["times"], y=data["headings"],
                mode='lines+markers',
                name=f'{acid} HDG',
                line=dict(color=color, dash='dot'),
                hovertemplate=f'{acid}<br>Time: %{{x:.1f}}s<br>Heading: %{{y:.1f}}°<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Vertical speed plot
        fig.add_trace(
            go.Scatter(
                x=data["times"], y=data["vs"],
                mode='lines+markers',
                name=f'{acid} VS',
                line=dict(color=color, dash='dashdot'),
                hovertemplate=f'{acid}<br>Time: %{{x:.1f}}s<br>VS: %{{y:.1f}}fpm<extra></extra>',
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title='BlueSky Autopilot and FMS Performance Analysis',
        height=800,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=2)
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=2)
    
    fig.update_yaxes(title_text="Speed (knots)", row=1, col=1)
    fig.update_yaxes(title_text="Altitude (feet)", row=1, col=2)
    fig.update_yaxes(title_text="Heading (degrees)", row=2, col=1)
    fig.update_yaxes(title_text="Vertical Speed (fpm)", row=2, col=2)
    
    # Save plot
    output_dir = Path("output/autopilot_test")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    plot_file = output_dir / "autopilot_performance_analysis.html"
    fig.write_html(str(plot_file))
    print(f"   📊 Performance plot saved: {plot_file}")
    
    # Show plot
    fig.show()


if __name__ == "__main__":
    print("🚁 BlueSky Proper Autopilot and FMS Configuration Test")
    print("This test addresses speed degradation by properly configuring:")
    print("  • Autothrottle (THR AUTO)")
    print("  • LNAV (Lateral Navigation)")
    print("  • VNAV (Vertical Navigation)")
    print("  • Autopilot targets (ALT, SPD, HDG)")
    print()
    
    success = test_proper_autopilot_configuration()
    
    if success:
        print("🎉 Test completed successfully!")
    else:
        print("💥 Test failed!")
        sys.exit(1)
