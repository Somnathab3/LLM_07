#!/usr/bin/env python3
"""
Try to access BlueSky aircraft data through embedded mode
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.cdr.simulation.bluesky_client import BlueSkyClient, create_thesis_config
import time

def test_embedded_access():
    """Test accessing aircraft data through BlueSky's embedded mode"""
    
    print("=== Testing Embedded BlueSky Access ===")
    
    # Create BlueSky client
    bluesky_config = create_thesis_config()
    bluesky_client = BlueSkyClient(bluesky_config)
    
    try:
        # Connect to BlueSky
        print("Connecting to BlueSky...")
        if not bluesky_client.connect():
            print("❌ Failed to connect to BlueSky")
            return
        
        print("✅ Connected to BlueSky")
        
        # Check if embedded mode is available
        print("\n=== Checking Embedded Mode ===")
        print(f"Has embedded_sim attribute: {hasattr(bluesky_client, 'embedded_sim')}")
        
        if hasattr(bluesky_client, 'embedded_sim'):
            embedded_sim = bluesky_client.embedded_sim
            print(f"Embedded sim object: {embedded_sim}")
            print(f"Embedded sim type: {type(embedded_sim)}")
            
            if embedded_sim and hasattr(embedded_sim, 'traffic'):
                traffic = embedded_sim.traffic
                print(f"Traffic object: {traffic}")
                print(f"Traffic type: {type(traffic)}")
                print(f"Traffic attributes: {dir(traffic)[:10]}...")
                
                # Check if there are aircraft initially
                if hasattr(traffic, 'id'):
                    print(f"Initial aircraft count: {len(traffic.id) if traffic.id else 0}")
                    if traffic.id:
                        print(f"Initial aircraft IDs: {list(traffic.id)}")
                else:
                    print("Traffic has no 'id' attribute")
        
        # Reset and create aircraft
        bluesky_client._send_command("RESET")
        time.sleep(1)
        
        # Create test aircraft
        success = bluesky_client.create_aircraft(
            callsign="TEST",
            aircraft_type="B738",
            lat=42.0,
            lon=-87.0,
            heading=90,
            altitude_ft=35000,
            speed_kt=450
        )
        
        if not success:
            print("❌ Failed to create aircraft")
            return
        
        print("✅ Created aircraft TEST at (42.0, -87.0)")
        
        # Start simulation
        bluesky_client.op()
        time.sleep(2)
        
        # Try to access aircraft data through embedded mode
        print("\n=== Accessing Aircraft Data ===")
        
        if hasattr(bluesky_client, 'embedded_sim') and bluesky_client.embedded_sim:
            embedded_sim = bluesky_client.embedded_sim
            
            if hasattr(embedded_sim, 'traffic'):
                traffic = embedded_sim.traffic
                
                print(f"Aircraft count after creation: {len(traffic.id) if hasattr(traffic, 'id') and traffic.id else 0}")
                
                if hasattr(traffic, 'id') and traffic.id:
                    aircraft_ids = list(traffic.id)
                    print(f"Aircraft IDs: {aircraft_ids}")
                    
                    if "TEST" in aircraft_ids:
                        idx = aircraft_ids.index("TEST")
                        print(f"Found TEST aircraft at index {idx}")
                        
                        # Try to get position data
                        position_attrs = ['lat', 'lon', 'alt', 'tas', 'cas', 'gs', 'hdg', 'vs']
                        aircraft_data = {}
                        
                        for attr in position_attrs:
                            if hasattr(traffic, attr):
                                attr_val = getattr(traffic, attr)
                                if attr_val is not None and idx < len(attr_val):
                                    aircraft_data[attr] = attr_val[idx]
                                    print(f"   {attr}: {attr_val[idx]}")
                                else:
                                    print(f"   {attr}: not available or index out of range")
                            else:
                                print(f"   {attr}: attribute not found")
                        
                        # Store initial position
                        initial_position = aircraft_data.copy()
                        
                        # Let simulation run
                        print("\n=== Letting Simulation Run ===")
                        print("Running for 10 seconds...")
                        time.sleep(10)
                        
                        # Check position again
                        print("Checking position after 10 seconds:")
                        new_aircraft_data = {}
                        
                        for attr in position_attrs:
                            if hasattr(traffic, attr):
                                attr_val = getattr(traffic, attr)
                                if attr_val is not None and idx < len(attr_val):
                                    new_aircraft_data[attr] = attr_val[idx]
                                    print(f"   {attr}: {attr_val[idx]}")
                        
                        # Compare positions
                        if 'lat' in initial_position and 'lat' in new_aircraft_data:
                            lat_diff = abs(new_aircraft_data['lat'] - initial_position['lat'])
                            lon_diff = abs(new_aircraft_data['lon'] - initial_position['lon']) if 'lon' in initial_position and 'lon' in new_aircraft_data else 0
                            
                            print(f"\n📊 Movement Analysis (Embedded Mode):")
                            print(f"   Initial: lat={initial_position.get('lat', 'N/A'):.6f}, lon={initial_position.get('lon', 'N/A'):.6f}")
                            print(f"   After 10s: lat={new_aircraft_data.get('lat', 'N/A'):.6f}, lon={new_aircraft_data.get('lon', 'N/A'):.6f}")
                            print(f"   Change:  Δlat={lat_diff:.6f}, Δlon={lon_diff:.6f}")
                            
                            if lat_diff > 0.001 or lon_diff > 0.001:
                                print("🎉 SUCCESS: Aircraft is moving (embedded mode)!")
                            else:
                                print("❌ No movement detected (embedded mode)")
                        
                        # Test with fast-forward
                        print("\n=== Testing Fast-Forward ===")
                        bluesky_client.hold()
                        bluesky_client.ff(120.0)  # 2 minutes
                        
                        print("Checking position after fast-forward:")
                        ff_aircraft_data = {}
                        
                        for attr in position_attrs:
                            if hasattr(traffic, attr):
                                attr_val = getattr(traffic, attr)
                                if attr_val is not None and idx < len(attr_val):
                                    ff_aircraft_data[attr] = attr_val[idx]
                                    print(f"   {attr}: {attr_val[idx]}")
                        
                        # Compare with initial
                        if 'lat' in initial_position and 'lat' in ff_aircraft_data:
                            lat_diff_ff = abs(ff_aircraft_data['lat'] - initial_position['lat'])
                            lon_diff_ff = abs(ff_aircraft_data['lon'] - initial_position['lon']) if 'lon' in initial_position and 'lon' in ff_aircraft_data else 0
                            
                            print(f"\n📊 Fast-Forward Movement Analysis:")
                            print(f"   Initial: lat={initial_position.get('lat', 'N/A'):.6f}, lon={initial_position.get('lon', 'N/A'):.6f}")
                            print(f"   After FF: lat={ff_aircraft_data.get('lat', 'N/A'):.6f}, lon={ff_aircraft_data.get('lon', 'N/A'):.6f}")
                            print(f"   Change:  Δlat={lat_diff_ff:.6f}, Δlon={lon_diff_ff:.6f}")
                            
                            if lat_diff_ff > 0.01 or lon_diff_ff > 0.01:
                                print("🎉 SUCCESS: Aircraft moved significantly with fast-forward!")
                                print("   The aircraft IS moving - the issue is in POS command parsing")
                            else:
                                print("❌ Still no significant movement with fast-forward")
                    
                    else:
                        print("❌ TEST aircraft not found in traffic IDs")
                else:
                    print("❌ No aircraft IDs found in traffic")
            else:
                print("❌ Traffic attribute not found in embedded sim")
        else:
            print("❌ Embedded simulation not available")
    
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        bluesky_client.disconnect()
        print("\nDisconnected from BlueSky")

if __name__ == "__main__":
    test_embedded_access()
