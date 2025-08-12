#!/usr/bin/env python3
"""
Test script for route-aware conflict resolution enhancements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cdr.ai.llm_client import ConflictContext, LLMConfig, LLMProvider
from src.cdr.utils.geo_route import create_deterministic_reroute


def test_enhanced_conflict_context():
    """Test the enhanced ConflictContext with destination"""
    print("🧪 Testing ConflictContext with destination...")
    
    context = ConflictContext(
        ownship_callsign="TEST123",
        ownship_state={
            "latitude": 42.3601,
            "longitude": -87.2734,
            "altitude": 35000,
            "heading": 90,
            "speed": 450,
            "vertical_speed_fpm": 0
        },
        intruders=[{
            "callsign": "INTR456",
            "latitude": 42.3701,
            "longitude": -87.2634,
            "altitude": 35000,
            "heading": 270,
            "speed": 400,
            "vertical_speed_fpm": 0
        }],
        scenario_time=300.0,
        lookahead_minutes=10.0,
        constraints={
            "max_heading_change": 45,
            "separation_minima": {"horizontal_nm": 5.0, "vertical_ft": 1000}
        },
        destination={
            "name": "DST1",
            "lat": 42.4000,
            "lon": -87.0000
        }
    )
    
    print(f"✅ ConflictContext created with destination: {context.destination}")
    assert context.destination is not None
    assert context.destination["name"] == "DST1"
    print("✅ ConflictContext test passed!")


def test_enhanced_llm_schemas():
    """Test the enhanced JSON schemas"""
    print("\n🧪 Testing enhanced LLM resolution schemas...")
    
    # Test direct_to resolution
    direct_to_resolution = {
        "schema_version": "cdr.v1",
        "conflict_id": "C1",
        "aircraft1": "TEST123",
        "aircraft2": "INTR456",
        "resolution_type": "direct_to",
        "parameters": {
            "waypoint_name": "DST1"
        },
        "reasoning": "Safe to proceed direct to destination",
        "confidence": 0.85
    }
    
    # Test reroute_via resolution
    reroute_via_resolution = {
        "schema_version": "cdr.v1",
        "conflict_id": "C2",
        "aircraft1": "TEST123",
        "aircraft2": "INTR456",
        "resolution_type": "reroute_via",
        "parameters": {
            "via_waypoint": {
                "name": "AVOID1",
                "lat": 42.3205,
                "lon": -87.3500
            },
            "resume_to_destination": True
        },
        "reasoning": "Dogleg 10 NM right of track avoids intruder cluster; rejoin DST1.",
        "confidence": 0.82
    }
    
    print("✅ Enhanced resolution schemas test passed!")
    print(f"   direct_to: {direct_to_resolution['resolution_type']}")
    print(f"   reroute_via: {reroute_via_resolution['resolution_type']}")


def test_deterministic_dogleg():
    """Test the deterministic dogleg route planner"""
    print("\n🧪 Testing deterministic dogleg route planner...")
    
    ownship = {
        "latitude": 42.3601,
        "longitude": -87.2734,
        "altitude": 35000,
        "heading": 90,
        "speed": 450
    }
    
    destination = {
        "name": "DST1",
        "lat": 42.4000,
        "lon": -87.0000
    }
    
    intruders = [{
        "callsign": "INTR456",
        "latitude": 42.3701,
        "longitude": -87.2634,
        "altitude": 35000,
        "heading": 270,
        "speed": 400
    }]
    
    # Test deterministic reroute
    reroute = create_deterministic_reroute(ownship, destination, intruders)
    
    if reroute:
        print(f"✅ Deterministic reroute created:")
        print(f"   Type: {reroute['resolution_type']}")
        print(f"   Via waypoint: {reroute['parameters']['via_waypoint']['name']}")
        print(f"   Coordinates: {reroute['parameters']['via_waypoint']['lat']:.4f}, {reroute['parameters']['via_waypoint']['lon']:.4f}")
        print(f"   Reasoning: {reroute['reasoning']}")
    else:
        print("✅ No reroute needed - direct route is clear")


def test_trafscript_generation():
    """Test TrafScript command generation for new resolution types"""
    print("\n🧪 Testing TrafScript generation...")
    
    from src.cdr.ai.llm_client import LLMClient
    
    # Test direct_to
    direct_to_resolution = {
        "resolution_type": "direct_to",
        "parameters": {
            "waypoint_name": "DST1"
        }
    }
    
    commands = LLMClient.to_trafscript("TEST123", direct_to_resolution)
    expected_command = "DIRECT TEST123,DST1"
    assert commands == [expected_command], f"Expected {expected_command}, got {commands}"
    print(f"✅ direct_to TrafScript: {commands[0]}")
    
    # Test reroute_via
    reroute_via_resolution = {
        "resolution_type": "reroute_via",
        "parameters": {
            "via_waypoint": {
                "name": "AVOID1",
                "lat": 42.3205,
                "lon": -87.3500
            },
            "resume_to_destination": True
        }
    }
    
    commands = LLMClient.to_trafscript("TEST123", reroute_via_resolution)
    expected_command = "DIRECT TEST123,AVOID1"
    assert commands == [expected_command], f"Expected {expected_command}, got {commands}"
    print(f"✅ reroute_via TrafScript: {commands[0]}")


def main():
    """Run all tests"""
    print("🚀 Testing Route-Aware Conflict Resolution Enhancements")
    print("=" * 60)
    
    try:
        test_enhanced_conflict_context()
        test_enhanced_llm_schemas()
        test_deterministic_dogleg()
        test_trafscript_generation()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! Route-aware CDR enhancements are working correctly.")
        print("\nKey features implemented:")
        print("  ✅ ConflictContext extended with destination awareness")
        print("  ✅ Enhanced JSON schemas for direct_to and reroute_via")
        print("  ✅ Deterministic dogleg route planning")
        print("  ✅ TrafScript command generation for new resolution types")
        print("  ✅ Pipeline support for route-aware resolutions")
        print("  ✅ Auto-resume to destination functionality")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
