#!/usr/bin/env python3
"""Test script for enhanced SCAT adapter with full ASTERIX I062 support"""

import json
from pathlib import Path
from datetime import datetime
import tempfile
import os

from src.cdr.adapters.scat_adapter import SCATAdapter
from src.cdr.schemas.scat_schemas import SCATData, TrackPoint


def create_test_scat_data() -> dict:
    """Create test SCAT data in the new format"""
    return {
        "flight_id": "TEST001",
        "callsign": "TST123",
        "aircraft_type": "B738",
        "departure": "ESSA",
        "arrival": "EGLL",
        "fpl_plan_update": {
            "callsign": "TST123",
            "route": "ESSA DCT BEROL DCT LASIN DCT EGLL",
            "rfl": 390,
            "tas": 450,
            "departure_time": "2023-07-15T08:30:00Z",
            "waypoints": [
                {"name": "BEROL", "lat": 59.1234, "lon": 18.5678, "altitude": 15000},
                {"name": "LASIN", "lat": 57.9876, "lon": 14.3210, "altitude": 39000}
            ]
        },
        "fpl_clearance": {
            "clearance_limit": "EGLL",
            "cleared_flight_level": 390,
            "updates": [
                {"time": 1689408600, "type": "altitude", "value": 350, "reason": "traffic"},
                {"time": 1689409200, "type": "direct", "value": "LASIN", "reason": "separation"}
            ]
        },
        "plots": [
            {
                "time_of_track": 1689408000,
                "source_id": "radar_station_1",
                "I062/010": {"sac": 101, "sic": 15},
                "I062/015": {"service_identification": 3},
                "I062/070": {"time_of_track": 45123.5},
                "I062/105": {"latitude": 59.651944, "longitude": 17.918611},
                "I062/100": {"x_coordinate": 123456.78, "y_coordinate": 654321.09},
                "I062/185": {"vx": 145.6, "vy": 78.2},
                "I062/200": {"track_status": "confirmed"},
                "I062/295": {"track_data_ages": {}},
                "I062/136": {"measured_flight_level": 390},
                "I062/130": {"altitude": 39000},
                "I062/220": {"rocd": -500},
                "I062/380": {
                    "aircraft_address": "4CA123",
                    "aircraft_identification": "TST123",
                    "magnetic_heading": 245.5,
                    "indicated_airspeed": 445,
                    "true_airspeed": 465,
                    "mach_number": 0.78,
                    "barometric_pressure_setting": 1013.25
                },
                "I062/040": {"track_number": 12345},
                "I062/080": {"mode_3a_code": "2000"},
                "I062/060": {"mode_c_code": 390},
                "I062/245": {"target_identification": "TST123"}
            },
            {
                "time_of_track": 1689408030,
                "source_id": "radar_station_1",
                "I062/105": {"latitude": 59.651800, "longitude": 17.918500},
                "I062/136": {"measured_flight_level": 385},
                "I062/185": {"vx": 140.2, "vy": 75.8},
                "I062/220": {"rocd": -800},
                "I062/380": {
                    "magnetic_heading": 245.0,
                    "indicated_airspeed": 440,
                    "true_airspeed": 460,
                    "mach_number": 0.77
                }
            }
        ],
        "performance_data": {
            "aircraft_type": "B738",
            "mtow": 79015.8,
            "oew": 41413.0,
            "max_fuel": 26020.0,
            "wing_area": 124.6,
            "wing_span": 35.79
        }
    }


def create_legacy_test_data() -> dict:
    """Create test data in legacy format (like sample_scat.json)"""
    return {
        "centre_ctrl": [
            {
                "centre_id": 1,
                "start_time": "2016-10-20T11:37:51.764000"
            }
        ],
        "fpl": {
            "fpl_base": [
                {
                    "callsign": "LEG456",
                    "aircraft_type": "A320",
                    "adep": "EKCH",
                    "ades": "LKPR",
                    "flight_rules": "I",
                    "wtc": "M",
                    "time_stamp": "2016-10-20T08:46:02.056000"
                }
            ]
        },
        "id": 100000,
        "plots": [
            {
                "I062/105": {
                    "lat": 55.34817159175873,
                    "lon": 13.029962182044983
                },
                "I062/136": {
                    "measured_flight_level": 146.25
                },
                "I062/185": {
                    "vx": 78.5,
                    "vy": -164.5
                },
                "I062/200": {
                    "adf": False,
                    "long": 0,
                    "trans": 0,
                    "vert": 1
                },
                "I062/220": {
                    "rocd": 3150.0
                },
                "I062/380": {
                    "subitem3": {
                        "mag_hdg": 149.765625
                    },
                    "subitem26": {
                        "ias": 307
                    },
                    "subitem27": {
                        "mach": 0.6
                    },
                    "subitem6": {
                        "altitude": 25000,
                        "sas": False,
                        "source": 0
                    }
                },
                "time_of_track": "2016-10-20T11:40:02.898437"
            }
        ]
    }


def test_new_format_parsing():
    """Test parsing of new SCAT format"""
    print("🧪 Testing New Format Parsing...")
    
    # Create temporary file
    test_data = create_test_scat_data()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f, indent=2)
        temp_file = Path(f.name)
    
    try:
        # Test adapter
        adapter = SCATAdapter(temp_file)
        raw_data = adapter.load_file()
        
        print(f"✅ File loaded successfully")
        print(f"   Callsign: {adapter.get_callsign()}")
        print(f"   Aircraft Type: {adapter.get_aircraft_type()}")
        
        # Test track parsing
        track_points = list(adapter.ownship_track())
        print(f"✅ Track points extracted: {len(track_points)}")
        
        if track_points:
            first_point = track_points[0]
            print(f"   First point: lat={first_point.latitude:.4f}, lon={first_point.longitude:.4f}")
            print(f"   Altitude: {first_point.altitude_ft} ft, FL: {first_point.flight_level}")
            print(f"   Speed: {first_point.speed_kt} kt, Heading: {first_point.heading_deg}°")
            print(f"   Mach: {first_point.mach}, ROCD: {first_point.vertical_rate_fpm} fpm")
        
        # Test flight plan parsing
        flight_plan = adapter.flight_plan()
        print(f"✅ Flight plan extracted:")
        print(f"   Route: {flight_plan.route_string}")
        print(f"   RFL: {flight_plan.requested_flight_level}")
        print(f"   Waypoints: {len(flight_plan.waypoints)}")
        
        # Test clearance extraction
        clearances = adapter.extract_clearances()
        print(f"✅ Clearances extracted: {len(clearances)}")
        
        for clearance in clearances:
            print(f"   {clearance.type}: {clearance.value} (reason: {clearance.reason})")
        
    finally:
        os.unlink(temp_file)


def test_legacy_format_parsing():
    """Test parsing of legacy SCAT format"""
    print("\n🧪 Testing Legacy Format Parsing...")
    
    # Create temporary file
    test_data = create_legacy_test_data()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f, indent=2)
        temp_file = Path(f.name)
    
    try:
        # Test adapter
        adapter = SCATAdapter(temp_file)
        raw_data = adapter.load_file()
        
        print(f"✅ Legacy file loaded successfully")
        print(f"   Callsign: {adapter.get_callsign()}")
        print(f"   Aircraft Type: {adapter.get_aircraft_type()}")
        
        # Test track parsing
        track_points = list(adapter.ownship_track())
        print(f"✅ Track points extracted: {len(track_points)}")
        
        if track_points:
            first_point = track_points[0]
            print(f"   First point: lat={first_point.latitude:.4f}, lon={first_point.longitude:.4f}")
            print(f"   Altitude: {first_point.altitude_ft} ft")
            print(f"   Speed: {first_point.speed_kt} kt, Heading: {first_point.heading_deg}°")
            print(f"   Mach: {first_point.mach}, ROCD: {first_point.vertical_rate_fpm} fpm")
        
    finally:
        os.unlink(temp_file)


def test_actual_sample_data():
    """Test with actual sample data"""
    print("\n🧪 Testing Actual Sample Data...")
    
    sample_file = Path("data/sample_scat.json")
    if not sample_file.exists():
        print("❌ Sample data file not found")
        return
    
    try:
        adapter = SCATAdapter(sample_file)
        raw_data = adapter.load_file()
        
        print(f"✅ Sample file loaded successfully")
        print(f"   Callsign: {adapter.get_callsign()}")
        print(f"   Aircraft Type: {adapter.get_aircraft_type()}")
        
        # Test track parsing
        track_points = list(adapter.ownship_track())
        print(f"✅ Track points extracted: {len(track_points)}")
        
        if track_points:
            first_point = track_points[0]
            last_point = track_points[-1]
            print(f"   First point: lat={first_point.latitude:.4f}, lon={first_point.longitude:.4f}")
            print(f"   Last point: lat={last_point.latitude:.4f}, lon={last_point.longitude:.4f}")
            print(f"   Duration: {last_point.timestamp - first_point.timestamp:.0f} seconds")
        
        # Test flight plan
        flight_plan = adapter.flight_plan()
        print(f"✅ Flight plan: {flight_plan.callsign} -> {flight_plan.departure_airport} to {flight_plan.arrival_airport}")
        
    except Exception as e:
        print(f"❌ Error testing sample data: {e}")


def test_neighbor_finding():
    """Test neighbor finding functionality"""
    print("\n🧪 Testing Neighbor Finding...")
    
    # Create multiple test files
    test_data_1 = create_test_scat_data()
    test_data_2 = create_test_scat_data()
    
    # Modify second aircraft to be nearby
    test_data_2["flight_id"] = "TEST002"
    test_data_2["callsign"] = "TST456"
    test_data_2["plots"][0]["I062/105"]["latitude"] = 59.652000  # Slightly different
    test_data_2["plots"][0]["I062/105"]["longitude"] = 17.919000
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        file1 = temp_path / "test1.json"
        file2 = temp_path / "test2.json"
        
        with open(file1, 'w') as f:
            json.dump(test_data_1, f)
        
        with open(file2, 'w') as f:
            json.dump(test_data_2, f)
        
        # Test neighbor finding
        adapter1 = SCATAdapter(file1)
        neighbors = adapter1.find_neighbors(temp_path)
        
        print(f"✅ Neighbor search completed")
        print(f"   Found {len(neighbors)} neighbors")
        
        for neighbor in neighbors:
            print(f"   Neighbor: {neighbor.get_callsign()}")


def test_distance_calculation():
    """Test haversine distance calculation"""
    print("\n🧪 Testing Distance Calculation...")
    
    # Test coordinates (Stockholm to Copenhagen approximately)
    lat1, lon1 = 59.3293, 18.0686  # Stockholm
    lat2, lon2 = 55.6761, 12.5683  # Copenhagen
    
    distance_nm = SCATAdapter.haversine_distance(lat1, lon1, lat2, lon2)
    expected_distance = 250  # Approximately 250 NM
    
    print(f"✅ Distance calculation test:")
    print(f"   Stockholm to Copenhagen: {distance_nm:.1f} NM")
    print(f"   Expected ~{expected_distance} NM")
    
    if abs(distance_nm - expected_distance) < 50:  # Within 50 NM tolerance
        print("✅ Distance calculation appears correct")
    else:
        print("❌ Distance calculation may have issues")


if __name__ == "__main__":
    print("🚀 Enhanced SCAT Adapter Test Suite")
    print("=" * 50)
    
    test_new_format_parsing()
    test_legacy_format_parsing()
    test_actual_sample_data()
    test_neighbor_finding()
    test_distance_calculation()
    
    print("\n" + "=" * 50)
    print("✅ Test suite completed!")
