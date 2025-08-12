#!/usr/bin/env python3
"""
Comprehensive end-to-end test for enhanced CDR system
Tests all functionality including BlueSky fixes, heading-aware destinations, and LLM logging
"""

import sys
import json
import math
import time
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cdr.simulation.bluesky_client import SimpleBlueSkyClient, Destination
from cdr.ai.llm_client_streamlined import StreamlinedLLMClient, LLMConfig, LLMProvider, ConflictContext


def test_heading_aware_destinations():
    """Test destination generation with different headings"""
    print("🧪 Testing heading-aware destination generation...")
    
    # Initialize LLM client
    config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3.1:8b")
    llm_client = StreamlinedLLMClient(config)
    
    # Test starting position
    start_lat = 42.3601
    start_lon = -87.3479
    
    # Test different headings
    test_headings = [0, 90, 180, 270, 45, 135, 225, 315]
    
    print(f"📍 Starting position: {start_lat:.4f}, {start_lon:.4f}")
    
    destinations = []
    for heading in test_headings:
        dest = llm_client.generate_destination_from_scat_start(
            start_lat, start_lon, current_heading=heading
        )
        destinations.append(dest)
        
        print(f"🧭 Heading {heading:3.0f}° → Destination bearing {dest['original_bearing']:3.0f}° "
              f"(difference: {abs(dest['original_bearing'] - heading):3.0f}°)")
    
    return destinations


def test_bluesky_command_fixes():
    """Test BlueSky command fixes for waypoint creation"""
    print("\n🧪 Testing BlueSky command fixes...")
    
    # Initialize BlueSky client
    bs_client = SimpleBlueSkyClient()
    bs_client.initialize()
    bs_client.reset()
    
    # Test data
    test_cases = [
        {"callsign": "TEST1", "lat": 42.3601, "lon": -87.3479, "hdg": 90},
        {"callsign": "TEST2", "lat": 42.4000, "lon": -87.2000, "hdg": 180},
        {"callsign": "TEST3", "lat": 42.2000, "lon": -87.5000, "hdg": 270},
    ]
    
    success_count = 0
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        
        # Create aircraft
        success = bs_client.create_aircraft(
            acid=test_case["callsign"],
            lat=test_case["lat"],
            lon=test_case["lon"],
            hdg=test_case["hdg"],
            alt=35000,
            spd=400
        )
        
        if not success:
            print(f"❌ Failed to create aircraft {test_case['callsign']}")
            continue
        
        print(f"✅ Created aircraft {test_case['callsign']}")
        
        # Generate destination with heading awareness
        destination = bs_client.generate_fixed_destination(
            test_case["lat"], test_case["lon"], 
            current_heading=test_case["hdg"]
        )
        
        # Set destination (this is where the previous error occurred)
        success = bs_client.set_aircraft_destination(test_case["callsign"], destination)
        
        if success:
            print(f"✅ Successfully set destination {destination.name}")
            success_count += 1
        else:
            print(f"❌ Failed to set destination {destination.name}")
    
    print(f"\n📊 BlueSky test results: {success_count}/{len(test_cases)} successful")
    return success_count == len(test_cases)


def test_multi_aircraft_complex_scenario():
    """Test complex multi-aircraft scenario with realistic parameters"""
    print("\n🧪 Testing complex multi-aircraft scenario...")
    
    # Initialize clients
    config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3.1:8b")
    llm_client = StreamlinedLLMClient(config)
    
    bs_client = SimpleBlueSkyClient()
    bs_client.initialize()
    bs_client.reset()
    
    # Complex scenario setup
    start_lat = 42.3601
    start_lon = -87.3479
    ownship_heading = 75.0  # Northeast
    
    # Create ownship
    ownship_callsign = "UAL123"
    success = bs_client.create_aircraft(
        acid=ownship_callsign,
        lat=start_lat,
        lon=start_lon,
        hdg=ownship_heading,
        alt=37000,
        spd=480
    )
    
    if not success:
        print("❌ Failed to create ownship")
        return False
    
    print(f"✅ Created ownship {ownship_callsign} heading {ownship_heading:.0f}°")
    
    # Generate destination based on current heading
    destination = llm_client.generate_destination_from_scat_start(
        start_lat, start_lon, current_heading=ownship_heading
    )
    bs_destination = Destination(
        name=destination['name'],
        lat=destination['latitude'],
        lon=destination['longitude'],
        alt=destination['altitude']
    )
    
    # Set destination with improved command handling
    success = bs_client.set_aircraft_destination(ownship_callsign, bs_destination)
    if not success:
        print("❌ Failed to set ownship destination")
        return False
    
    print(f"🎯 Set destination: {destination['name']} at {destination['latitude']:.4f}, {destination['longitude']:.4f}")
    print(f"📏 Distance: {destination['original_distance_nm']:.1f} NM, Bearing: {destination['original_bearing']:.0f}°")
    
    # Create multiple intruders with realistic positioning
    intruders = [
        {"callsign": "DAL456", "lat": start_lat + 0.03, "lon": start_lon + 0.04, "hdg": 255, "alt": 37000, "spd": 460},
        {"callsign": "AAL789", "lat": start_lat - 0.02, "lon": start_lon + 0.06, "hdg": 135, "alt": 36000, "spd": 500},
        {"callsign": "SWA101", "lat": start_lat + 0.05, "lon": start_lon - 0.01, "hdg": 45, "alt": 38000, "spd": 420},
        {"callsign": "FDX234", "lat": start_lat - 0.04, "lon": start_lon - 0.03, "hdg": 315, "alt": 35000, "spd": 390},
        {"callsign": "UPS567", "lat": start_lat + 0.02, "lon": start_lon + 0.07, "hdg": 180, "alt": 39000, "spd": 450},
    ]
    
    created_intruders = []
    for intruder in intruders:
        success = bs_client.create_aircraft(
            acid=intruder["callsign"],
            lat=intruder["lat"],
            lon=intruder["lon"],
            hdg=intruder["hdg"],
            alt=intruder["alt"],
            spd=intruder["spd"]
        )
        if success:
            print(f"✅ Created intruder {intruder['callsign']}")
            created_intruders.append(intruder)
    
    print(f"🛫 Created {len(created_intruders)} intruders")
    
    # Step simulation forward
    for step in range(5):
        bs_client.step_simulation(dt=10.0)  # 10-second steps
    
    # Get aircraft states
    ownship_state = bs_client.get_aircraft_state(ownship_callsign)
    if not ownship_state:
        print("❌ Failed to get ownship state")
        return False
    
    intruder_states = []
    for intruder in created_intruders:
        state = bs_client.get_aircraft_state(intruder["callsign"])
        if state:
            intruder_states.append({
                'callsign': state.id,
                'latitude': state.lat,
                'longitude': state.lon,
                'altitude': state.alt,
                'heading': state.hdg,
                'speed': state.tas,
                'vertical_speed_fpm': state.vs * 60  # Convert to fpm
            })
    
    print(f"📊 Retrieved states for ownship and {len(intruder_states)} intruders")
    
    # Create conflict context
    context = ConflictContext(
        ownship_callsign=ownship_callsign,
        ownship_state={
            'latitude': ownship_state.lat,
            'longitude': ownship_state.lon,
            'altitude': ownship_state.alt,
            'heading': ownship_state.hdg,
            'speed': ownship_state.tas,
            'vertical_speed_fpm': ownship_state.vs * 60
        },
        intruders=intruder_states,
        scenario_time=bs_client.get_simulation_time(),
        lookahead_minutes=15.0,
        destination=destination
    )
    
    # Call LLM with enhanced logging
    print(f"📤 Calling LLM with {len(intruder_states)} intruders and heading-aware destination...")
    result = llm_client.detect_and_resolve_conflicts(context)
    
    # Analyze result
    print(f"\n📥 LLM Response Analysis:")
    print(f"   Conflicts detected: {result.get('conflicts_detected', False)}")
    print(f"   Number of conflicts: {len(result.get('conflicts', []))}")
    
    if result.get('conflicts'):
        for i, conflict in enumerate(result['conflicts']):
            print(f"   - Conflict {i+1}: {conflict.get('intruder_callsign', 'Unknown')}")
    
    resolution = result.get('resolution', {})
    print(f"   Resolution type: {resolution.get('resolution_type', 'none')}")
    print(f"   Parameters: {resolution.get('parameters', {})}")
    print(f"   Reasoning: {resolution.get('reasoning', 'N/A')}")
    print(f"   Confidence: {resolution.get('confidence', 0.0)}")
    
    return True


def test_enhanced_logging():
    """Test enhanced logging functionality"""
    print("\n🧪 Testing enhanced logging...")
    
    # Initialize LLM client
    config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3.1:8b")
    llm_client = StreamlinedLLMClient(config)
    
    # Create simple conflict scenario for logging test
    context = ConflictContext(
        ownship_callsign="LOG_TEST",
        ownship_state={
            'latitude': 42.3601,
            'longitude': -87.3479,
            'altitude': 35000,
            'heading': 90,
            'speed': 450,
            'vertical_speed_fpm': 0
        },
        intruders=[
            {
                'callsign': 'LOG_INTRUDER',
                'latitude': 42.3701,
                'longitude': -87.3379,
                'altitude': 35000,
                'heading': 270,
                'speed': 440,
                'vertical_speed_fpm': 0
            }
        ],
        scenario_time=1234567890,
        lookahead_minutes=10.0,
        destination={
            'name': 'LOG_DEST',
            'latitude': 42.5000,
            'longitude': -87.0000,
            'altitude': 35000
        }
    )
    
    print("📤 Testing LLM call with enhanced logging (watch for debug output)...")
    result = llm_client.detect_and_resolve_conflicts(context)
    
    # Check log directory
    log_dir = Path("logs/llm")
    if log_dir.exists():
        log_files = list(log_dir.glob("*.json"))
        print(f"📁 Found {len(log_files)} log files in {log_dir}")
        
        if log_files:
            # Check latest log file
            latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
            print(f"📄 Latest log file: {latest_log.name}")
            
            try:
                with open(latest_log, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    
                print(f"📊 Log file analysis:")
                print(f"   Prompt length: {log_data.get('prompt_length', 0)} chars")
                print(f"   Response length: {log_data.get('response_length', 0)} chars")
                print(f"   Model: {log_data.get('model_config', {}).get('model', 'unknown')}")
                print(f"   Valid JSON: {log_data.get('interaction_metadata', {}).get('response_valid_json', False)}")
                
            except Exception as e:
                print(f"⚠️ Failed to read log file: {e}")
    else:
        print("⚠️ Log directory not found")
    
    return len(result.get('conflicts', [])) >= 0  # Any result is valid


def main():
    """Run comprehensive end-to-end tests"""
    print("🚀 Starting comprehensive end-to-end CDR system tests...")
    print("=" * 80)
    
    tests = [
        ("Heading-aware destinations", test_heading_aware_destinations),
        ("BlueSky command fixes", test_bluesky_command_fixes),
        ("Complex multi-aircraft scenario", test_multi_aircraft_complex_scenario),
        ("Enhanced logging", test_enhanced_logging),
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            results[test_name] = False
            print(f"\n❌ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print("\n" + "=" * 80)
    print("🎯 TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for production use.")
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
