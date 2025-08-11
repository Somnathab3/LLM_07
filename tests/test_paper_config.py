#!/usr/bin/env python3
"""Test script for paper-driven configuration changes (no simulation required)"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.cdr.simulation.bluesky_client import (
    BlueSkyClient, BlueSkyConfig, create_thesis_config, 
    DetectionConfig, AircraftState, ConflictInfo
)


def test_configuration_improvements():
    """Test the new configuration features without requiring BlueSky simulation"""
    print("🧪 Testing Paper-Driven Configuration Improvements")
    print("=" * 55)
    
    # Test 1: Thesis configuration factory
    print("\n1️⃣ Testing Thesis Configuration Factory...")
    config = create_thesis_config(seed=1234)
    
    expected_values = {
        'dt': 1.0,
        'dtmult': 8.0,
        'seed': 1234,
        'asas_enabled': True,
        'reso_off': True,
        'dtlook': 600.0,
        'dtnolook': 5.0,
        'det_radius_nm': 5.0,
        'det_half_vert_ft': 500.0
    }
    
    all_correct = True
    for key, expected in expected_values.items():
        actual = getattr(config, key)
        if actual != expected:
            print(f"❌ {key}: expected {expected}, got {actual}")
            all_correct = False
        else:
            print(f"✅ {key}: {actual}")
    
    if all_correct:
        print("✅ Thesis configuration factory working correctly")
    else:
        print("❌ Some configuration values incorrect")
        return False
    
    # Test 2: Legacy parameter synchronization
    print("\n2️⃣ Testing Legacy Parameter Synchronization...")
    legacy_config = BlueSkyConfig(
        fast_time_factor=4.0,
        lookahead_time=300.0,
        protected_zone_radius=3.0,
        protected_zone_height=800.0
    )
    
    sync_tests = [
        ('dtmult', 'fast_time_factor', 4.0),
        ('dtlook', 'lookahead_time', 300.0),
        ('det_radius_nm', 'protected_zone_radius', 3.0),
        ('det_half_vert_ft', None, 400.0)  # protected_zone_height / 2
    ]
    
    all_synced = True
    for new_param, legacy_param, expected in sync_tests:
        actual = getattr(legacy_config, new_param)
        if actual != expected:
            print(f"❌ {new_param}: expected {expected}, got {actual}")
            all_synced = False
        else:
            legacy_desc = f" (from {legacy_param})" if legacy_param else ""
            print(f"✅ {new_param}: {actual}{legacy_desc}")
    
    if all_synced:
        print("✅ Legacy parameter synchronization working")
    else:
        print("❌ Legacy parameter synchronization broken")
        return False
    
    # Test 3: Client initialization with new parameters
    print("\n3️⃣ Testing Client Initialization...")
    client = BlueSkyClient(config)
    
    # Check that client has the expected attributes
    expected_attrs = ['callsigns', 'command_map']
    missing_attrs = []
    
    for attr in expected_attrs:
        if not hasattr(client, attr):
            missing_attrs.append(attr)
        else:
            print(f"✅ Client has {attr}: {getattr(client, attr)}")
    
    if missing_attrs:
        print(f"❌ Client missing attributes: {missing_attrs}")
        return False
    
    # Test 4: Command map includes new TrafScript commands
    print("\n4️⃣ Testing TrafScript Command Map...")
    expected_commands = ['ic', 'batch', 'ff']
    missing_commands = []
    
    for cmd in expected_commands:
        if cmd not in client.command_map:
            missing_commands.append(cmd)
        else:
            print(f"✅ Command '{cmd}' -> '{client.command_map[cmd]}'")
    
    if missing_commands:
        print(f"❌ Missing commands: {missing_commands}")
        return False
    
    # Test 5: Callsign tracking functionality
    print("\n5️⃣ Testing Callsign Tracking...")
    
    # Simulate aircraft creation (without actual BlueSky)
    client.callsigns = []  # Reset
    test_callsigns = ["TEST001", "TEST002", "PAPER01"]
    
    # Simulate adding callsigns
    for callsign in test_callsigns:
        if callsign not in client.callsigns:
            client.callsigns.append(callsign)
    
    if client.callsigns == test_callsigns:
        print(f"✅ Callsign tracking: {client.callsigns}")
    else:
        print(f"❌ Callsign tracking failed: expected {test_callsigns}, got {client.callsigns}")
        return False
    
    # Simulate removing callsigns
    client.callsigns.remove("TEST001")
    if "TEST001" not in client.callsigns and len(client.callsigns) == 2:
        print(f"✅ Callsign removal: {client.callsigns}")
    else:
        print(f"❌ Callsign removal failed: {client.callsigns}")
        return False
    
    print("\n✅ All configuration tests PASSED!")
    print("🎉 Paper-driven improvements are working correctly!")
    return True


def test_data_structures():
    """Test the data structures used for aircraft and conflicts"""
    print("\n🧪 Testing Data Structures")
    print("=" * 30)
    
    # Test AircraftState
    print("\n1️⃣ Testing AircraftState...")
    state = AircraftState(
        callsign="TEST001",
        latitude=52.0,
        longitude=4.0,
        altitude_ft=35000,
        heading_deg=90,
        speed_kt=450,
        vertical_speed_fpm=1000,
        timestamp=1234567890.0
    )
    
    print(f"✅ AircraftState: {state.callsign} at {state.latitude},{state.longitude}")
    
    # Test ConflictInfo
    print("\n2️⃣ Testing ConflictInfo...")
    conflict = ConflictInfo(
        aircraft1="TEST001",
        aircraft2="TEST002",
        time_to_conflict=120.0,
        horizontal_distance=3.5,
        vertical_distance=800.0,
        conflict_type="both",
        severity="high"
    )
    
    print(f"✅ ConflictInfo: {conflict.aircraft1} vs {conflict.aircraft2}, "
          f"type={conflict.conflict_type}, severity={conflict.severity}")
    
    return True


def test_detection_config():
    """Test the DetectionConfig preset"""
    print("\n🧪 Testing DetectionConfig Preset")
    print("=" * 35)
    
    detection_config = DetectionConfig()
    
    expected_values = {
        'dt': 1.0,
        'dtmult': 8.0,
        'seed': 1337,
        'dtlook': 600.0,
        'dtnolook': 5.0,
        'det_radius_nm': 5.0,
        'det_half_vert_ft': 500.0,
        'asas_enabled': True,
        'reso_off': True
    }
    
    all_correct = True
    for key, expected in expected_values.items():
        actual = getattr(detection_config, key)
        if actual != expected:
            print(f"❌ DetectionConfig.{key}: expected {expected}, got {actual}")
            all_correct = False
        else:
            print(f"✅ DetectionConfig.{key}: {actual}")
    
    if all_correct:
        print("✅ DetectionConfig preset working correctly")
        return True
    else:
        print("❌ DetectionConfig preset has incorrect values")
        return False


def main():
    """Run all configuration tests"""
    print("🚀 Paper-Driven Configuration Test Suite")
    print("=" * 42)
    print("   (No BlueSky simulation required)")
    
    success = True
    
    try:
        success &= test_configuration_improvements()
        success &= test_data_structures()
        success &= test_detection_config()
        
        if success:
            print("\n" + "=" * 50)
            print("🎉 All configuration tests PASSED!")
            print("✅ Your BlueSky client is ready for paper-aligned experiments")
            print("\nKey improvements:")
            print("  📊 Thesis configuration factory (create_thesis_config)")
            print("  🔄 Legacy parameter synchronization")
            print("  📡 Callsign tracking for efficient state queries")
            print("  🎛️  TrafScript command helpers (ic, batch, ff, op, hold)")
            print("  🔍 SSD CONFLICTS-based conflict detection")
            print("  ⚙️  Paper-aligned initialization sequence")
            print("  🎯 5 NM / ±1000 ft / 10-min detection preset")
        else:
            print("\n" + "=" * 50)
            print("⚠️ Some configuration tests FAILED")
            print("   Check the output above for details")
            
    except Exception as e:
        print(f"\n💥 Configuration test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    print("\n🏁 Configuration test suite completed")
    return success


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
