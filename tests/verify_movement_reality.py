#!/usr/bin/env python3
"""
Verify if the aircraft movement is real or fake by comparing:
1. Our dead reckoning calculations
2. What BlueSky actually reports (even if broken)
3. Alternative BlueSky commands to verify real simulation state
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import math
from src.cdr.simulation.bluesky_client import BlueSkyClient, create_thesis_config

class MovementVerifier:
    """Verify if movement is real by comparing multiple data sources"""
    
    def __init__(self, client):
        self.client = client
        self.aircraft_created = {}
        self.simulation_start_time = None
    
    def verify_aircraft_movement(self):
        """Complete verification of aircraft movement reality"""
        
        print("=== Movement Reality Verification ===")
        print("Testing if aircraft movement is real or just calculated...")
        
        try:
            # Create aircraft with known parameters
            callsign = "VERIFY01"
            initial_lat = 42.0
            initial_lon = -87.0
            heading = 90  # East
            speed_kt = 450
            
            print(f"\n1️⃣ Creating aircraft {callsign}:")
            print(f"   Position: {initial_lat}, {initial_lon}")
            print(f"   Heading: {heading}° (East)")
            print(f"   Speed: {speed_kt} knots")
            
            success = self.client.create_aircraft(
                callsign=callsign,
                aircraft_type="B738",
                lat=initial_lat,
                lon=initial_lon,
                heading=heading,
                altitude_ft=35000,
                speed_kt=speed_kt
            )
            
            if not success:
                print("❌ Failed to create aircraft")
                return
            
            # Store creation data
            self.aircraft_created[callsign] = {
                'lat': initial_lat,
                'lon': initial_lon,
                'heading': heading,
                'speed': speed_kt,
                'creation_time': time.time()
            }
            
            # Start simulation
            print("\n2️⃣ Starting simulation...")
            self.client.op()
            self.simulation_start_time = time.time()
            
            # Test 1: Check what broken POS command returns
            print("\n3️⃣ Testing broken POS command...")
            time.sleep(2)  # Let simulation run briefly
            
            broken_states = self.client.get_aircraft_states([callsign])
            broken_pos = broken_states.get(callsign)
            
            if broken_pos:
                print(f"   Broken POS returns: lat={broken_pos.latitude:.6f}, lon={broken_pos.longitude:.6f}")
                
                # Check if this matches creation values
                if (abs(broken_pos.latitude - initial_lat) < 0.0001 and 
                    abs(broken_pos.longitude - initial_lon) < 0.0001):
                    print("   ❌ BROKEN: Returns exact creation values (cached)")
                else:
                    print("   ✅ POS might be working (different from creation)")
            
            # Test 2: Calculate expected position using physics
            print("\n4️⃣ Calculating expected position using physics...")
            simulation_time = 10.0  # 10 seconds
            time.sleep(simulation_time)
            
            expected_lat, expected_lon = self.calculate_expected_position(
                initial_lat, initial_lon, heading, speed_kt, simulation_time
            )
            
            print(f"   Expected after {simulation_time}s: lat={expected_lat:.6f}, lon={expected_lon:.6f}")
            
            # Test 3: Check if BlueSky reports different values over time
            print("\n5️⃣ Testing if BlueSky reports change over time...")
            
            # Get position at different times
            positions = []
            for i in range(3):
                current_states = self.client.get_aircraft_states([callsign])
                current_pos = current_states.get(callsign)
                if current_pos:
                    positions.append({
                        'time': time.time() - self.simulation_start_time,
                        'lat': current_pos.latitude,
                        'lon': current_pos.longitude
                    })
                    print(f"   T+{positions[-1]['time']:.1f}s: lat={current_pos.latitude:.6f}, lon={current_pos.longitude:.6f}")
                time.sleep(3)
            
            # Analyze if positions are changing
            if len(positions) >= 2:
                pos_changes = []
                for i in range(1, len(positions)):
                    lat_change = abs(positions[i]['lat'] - positions[i-1]['lat'])
                    lon_change = abs(positions[i]['lon'] - positions[i-1]['lon'])
                    pos_changes.append(lat_change + lon_change)
                
                total_change = sum(pos_changes)
                print(f"   Total position change reported by BlueSky: {total_change:.8f}")
                
                if total_change < 0.000001:
                    print("   ❌ BROKEN: BlueSky reports no position change (static values)")
                else:
                    print("   ✅ BlueSky reports position changes (might be real)")
            
            # Test 4: Use alternative BlueSky commands
            print("\n6️⃣ Testing alternative BlueSky commands...")
            
            # Try LISTROUTE command
            print("   Testing LISTROUTE command...")
            route_response = self.client._send_command(f"LISTROUTE {callsign}")
            print(f"   LISTROUTE response: {route_response[:100] if route_response else 'None'}...")
            
            # Try INFO command
            print("   Testing INFO command...")
            info_response = self.client._send_command(f"INFO {callsign}")
            print(f"   INFO response: {info_response[:100] if info_response else 'None'}...")
            
            # Try TRAIL command to see if aircraft has moved
            print("   Testing TRAIL command...")
            trail_response = self.client._send_command(f"TRAIL {callsign} ON")
            print(f"   TRAIL response: {trail_response[:50] if trail_response else 'None'}...")
            
            # Test 5: Fast-forward and check for dramatic changes
            print("\n7️⃣ Testing fast-forward for dramatic movement...")
            
            self.client.hold()
            print("   Paused simulation")
            
            # Get position before fast-forward
            pre_ff_states = self.client.get_aircraft_states([callsign])
            pre_ff_pos = pre_ff_states.get(callsign)
            
            # Fast-forward 2 minutes (120 seconds)
            ff_time = 120.0
            self.client.ff(ff_time)
            print(f"   Fast-forwarded {ff_time} seconds")
            
            # Get position after fast-forward
            post_ff_states = self.client.get_aircraft_states([callsign])
            post_ff_pos = post_ff_states.get(callsign)
            
            if pre_ff_pos and post_ff_pos:
                ff_lat_change = abs(post_ff_pos.latitude - pre_ff_pos.latitude)
                ff_lon_change = abs(post_ff_pos.longitude - pre_ff_pos.longitude)
                
                print(f"   Before FF: lat={pre_ff_pos.latitude:.6f}, lon={pre_ff_pos.longitude:.6f}")
                print(f"   After FF:  lat={post_ff_pos.latitude:.6f}, lon={post_ff_pos.longitude:.6f}")
                print(f"   Change: Δlat={ff_lat_change:.6f}, Δlon={ff_lon_change:.6f}")
                
                # Calculate what change should be expected
                expected_ff_lat, expected_ff_lon = self.calculate_expected_position(
                    initial_lat, initial_lon, heading, speed_kt, ff_time + simulation_time
                )
                
                expected_ff_lat_change = abs(expected_ff_lat - initial_lat)
                expected_ff_lon_change = abs(expected_ff_lon - initial_lon)
                
                print(f"   Expected total change: Δlat={expected_ff_lat_change:.6f}, Δlon={expected_ff_lon_change:.6f}")
                
                # Compare actual vs expected
                lat_accuracy = abs(ff_lat_change - expected_ff_lat_change) / expected_ff_lat_change if expected_ff_lat_change > 0 else 1
                lon_accuracy = abs(ff_lon_change - expected_ff_lon_change) / expected_ff_lon_change if expected_ff_lon_change > 0 else 1
                
                print(f"   Accuracy: lat={lat_accuracy:.3f}, lon={lon_accuracy:.3f} (0=perfect, <0.1=good)")
                
                if ff_lat_change < 0.001 and ff_lon_change < 0.001:
                    print("   ❌ FAKE: No significant movement after fast-forward")
                elif lat_accuracy < 0.1 and lon_accuracy < 0.1:
                    print("   ✅ REAL: Movement matches physics expectations")
                else:
                    print("   ⚠️ UNKNOWN: Movement detected but doesn't match expectations")
            
            # Test 6: Create second aircraft and compare
            print("\n8️⃣ Testing with second aircraft for comparison...")
            
            callsign2 = "VERIFY02"
            success2 = self.client.create_aircraft(
                callsign=callsign2,
                aircraft_type="A320",
                lat=41.0,
                lon=-88.0,
                heading=180,  # South
                altitude_ft=30000,
                speed_kt=350
            )
            
            if success2:
                print("   Created second aircraft for comparison")
                
                # Get both aircraft positions
                both_states = self.client.get_aircraft_states([callsign, callsign2])
                pos1 = both_states.get(callsign)
                pos2 = both_states.get(callsign2)
                
                if pos1 and pos2:
                    print(f"   Aircraft 1: lat={pos1.latitude:.6f}, lon={pos1.longitude:.6f}")
                    print(f"   Aircraft 2: lat={pos2.latitude:.6f}, lon={pos2.longitude:.6f}")
                    
                    # Check if they report different positions
                    if (abs(pos1.latitude - pos2.latitude) > 0.001 or 
                        abs(pos1.longitude - pos2.longitude) > 0.001):
                        print("   ✅ Different aircraft report different positions")
                    else:
                        print("   ❌ All aircraft report same position (likely broken)")
        
        except Exception as e:
            print(f"❌ Error during verification: {e}")
            import traceback
            traceback.print_exc()
    
    def calculate_expected_position(self, initial_lat, initial_lon, heading, speed_kt, time_seconds):
        """Calculate expected position using dead reckoning"""
        
        heading_rad = math.radians(heading)
        time_hours = time_seconds / 3600.0
        distance_nm = speed_kt * time_hours
        
        # Convert to lat/lon changes
        lat_change = distance_nm * math.cos(heading_rad) / 60.0
        lon_change = distance_nm * math.sin(heading_rad) / (60.0 * math.cos(math.radians(initial_lat)))
        
        return initial_lat + lat_change, initial_lon + lon_change

def main():
    """Run movement verification test"""
    
    print("=== Aircraft Movement Reality Verification ===")
    print("This test determines if movement is real or just calculated")
    
    # Create BlueSky client
    config = create_thesis_config()
    client = BlueSkyClient(config)
    
    try:
        # Connect to BlueSky
        print("Connecting to BlueSky...")
        if not client.connect():
            print("❌ Failed to connect to BlueSky")
            return
        
        print("✅ Connected to BlueSky")
        
        # Reset simulation
        client._send_command("RESET")
        time.sleep(1)
        
        # Run verification
        verifier = MovementVerifier(client)
        verifier.verify_aircraft_movement()
        
        # Summary
        print("\n" + "="*50)
        print("VERIFICATION SUMMARY:")
        print("Check the results above to determine:")
        print("✅ REAL = BlueSky simulation actually moving aircraft")
        print("❌ FAKE = Only our calculations, BlueSky static")
        print("⚠️ MIXED = Partial working, needs investigation")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect()
        print("\nDisconnected from BlueSky")

if __name__ == "__main__":
    main()
