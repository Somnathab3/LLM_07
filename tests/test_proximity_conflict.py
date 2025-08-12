#!/usr/bin/env python3
"""
Simple Aircraft Proximity Test with LLM Resolution

This test creates a controlled scenario with two aircraft in close proximity
to guarantee a conflict and test LLM-based resolution.
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append('.')

from src.cdr.pipeline.cdr_pipeline import CDRPipeline, PipelineConfig
from src.cdr.simulation.bluesky_client import BlueSkyClient, AircraftState, BlueSkyConfig
from src.cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider


class TestCDRPipeline(CDRPipeline):
    """CDR Pipeline modified for proximity testing"""
    
    def _get_aircraft_destination(self, callsign: str) -> Optional[Dict[str, Any]]:
        """Override destination to be much farther away for testing"""
        # Place destination much farther to prevent premature test completion
        return {
            "name": "FAR_DST",
            "lat": 40.0,  # Much farther south
            "lon": -85.0  # Much farther east
        }
    
    def _complete_test_if_destination_reached(self) -> bool:
        """Disable destination-based completion for proximity test"""
        return False  # Never complete based on destination


class ProximityTestScenario:
    """Simple test scenario with guaranteed aircraft proximity"""
    
    def __init__(self):
        self.scenario_id = "proximity_test"
        
    def to_simple_scenario(self) -> Dict[str, Any]:
        """Create a simple scenario with two aircraft on collision course"""
        return {
            'ownship': {
                'callsign': 'TEST001',
                'aircraft_type': 'B738',
                'latitude': 41.9786,    # Chicago area
                'longitude': -87.9048,
                'altitude_ft': 37000,
                'heading_deg': 90,      # Flying East
                'speed_kt': 450
            },
            'initial_traffic': [],
            'pending_intruders': [
                {
                    'callsign': 'TEST002',
                    'aircraft_type': 'A320',
                    'latitude': 41.9786,        # Same latitude (collision course)
                    'longitude': -87.8848,      # Only 1.2 NM East of ownship (closer!)
                    'altitude_ft': 37000,       # Same altitude
                    'heading_deg': 270,         # Flying West (head-on toward ownship)
                    'speed_kt': 420,
                    'injection_time_minutes': 0.1  # Inject after 6 seconds
                }
            ]
        }


def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_test_config() -> PipelineConfig:
    """Create test configuration with sensitive detection parameters"""
    config = PipelineConfig()
    
    # Make detection very sensitive
    config.separation_min_nm = 1.0        # 1 NM minimum separation
    config.separation_min_ft = 300         # 300 ft minimum separation
    config.detection_range_nm = 20.0       # 20 NM detection range
    config.lookahead_minutes = 10.0        # 10 minute lookahead
    
    # Short cycles for quick testing
    config.cycle_interval_seconds = 5.0    # 5 second cycles (faster)
    config.max_simulation_time_minutes = 5.0  # 5 minute max test (shorter)
    
    # Enable LLM resolution
    config.llm_enabled = True
    
    # Resolution constraints
    config.max_heading_change_deg = 45.0
    config.max_altitude_change_ft = 2000
    
    return config


def verify_aircraft_states(bluesky_client: BlueSkyClient, expected_callsigns: list) -> bool:
    """Verify that all expected aircraft are active in BlueSky"""
    try:
        states = bluesky_client.get_aircraft_states()
        if not states:
            print("❌ No aircraft states returned from BlueSky")
            return False
        
        found_callsigns = set(states.keys())
        expected_set = set(expected_callsigns)
        
        print(f"✅ Found aircraft: {found_callsigns}")
        print(f"Expected aircraft: {expected_set}")
        
        if not expected_set.issubset(found_callsigns):
            missing = expected_set - found_callsigns
            print(f"❌ Missing aircraft: {missing}")
            return False
        
        # Print aircraft positions
        for callsign, state in states.items():
            if callsign in expected_callsigns:
                print(f"📍 {callsign}: {state.latitude:.4f}, {state.longitude:.4f}, "
                      f"{state.altitude_ft}ft, {state.heading_deg}°, {state.speed_kt}kt")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying aircraft states: {e}")
        return False


def calculate_separation(state1: AircraftState, state2: AircraftState) -> tuple:
    """Calculate horizontal and vertical separation between aircraft"""
    import math
    
    # Horizontal separation (rough calculation in NM)
    dlat = state2.latitude - state1.latitude
    dlon = state2.longitude - state1.longitude
    horizontal_nm = math.sqrt(dlat**2 + dlon**2) * 60  # Rough conversion to NM
    
    # Vertical separation
    vertical_ft = abs(state2.altitude_ft - state1.altitude_ft)
    
    return horizontal_nm, vertical_ft


def check_for_conflicts(bluesky_client: BlueSkyClient) -> Optional[tuple]:
    """Check for conflicts between aircraft"""
    try:
        states = bluesky_client.get_aircraft_states()
        if len(states) < 2:
            return None
        
        callsigns = list(states.keys())
        state1 = states[callsigns[0]]
        state2 = states[callsigns[1]]
        
        horizontal_nm, vertical_ft = calculate_separation(state1, state2)
        
        print(f"🔍 Separation: {horizontal_nm:.2f} NM horizontal, {vertical_ft:.0f} ft vertical")
        
        # Check if within conflict thresholds
        if horizontal_nm < 3.0 and vertical_ft < 1000:  # Standard separation minima
            print(f"⚠️  CONFLICT DETECTED: {callsigns[0]} vs {callsigns[1]}")
            return (callsigns[0], callsigns[1], horizontal_nm, vertical_ft)
        
        return None
        
    except Exception as e:
        print(f"❌ Error checking conflicts: {e}")
        return None


def test_bluesky_connection() -> Optional[BlueSkyClient]:
    """Test BlueSky connection and basic functionality"""
    print("🔗 Testing BlueSky connection...")
    
    try:
        # Create BlueSky config
        bs_config = BlueSkyConfig()
        client = BlueSkyClient(bs_config)
        
        if not client.connect():
            print("❌ Failed to connect to BlueSky")
            return None
        
        print("✅ Connected to BlueSky")
        print(f"   Direct bridge: {'Available' if client.use_direct_bridge else 'Unavailable'}")
        
        # Test basic commands
        client._send_command("RESET", expect_response=True)
        client._initialize_simulation()
        
        print("✅ BlueSky initialized")
        return client
        
    except Exception as e:
        print(f"❌ BlueSky connection error: {e}")
        return None


def run_proximity_test():
    """Run the aircraft proximity test with LLM resolution"""
    logger = setup_logging()
    print("🚁 Starting Aircraft Proximity Test with LLM Resolution")
    print("=" * 60)
    
    # Test BlueSky connection first
    bluesky_client = test_bluesky_connection()
    if not bluesky_client:
        print("❌ Test failed - BlueSky connection issue")
        return False
    
    try:
        # Create test configuration
        config = create_test_config()
        
        # Create LLM client
        print("🤖 Initializing LLM client...")
        llm_config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model_name="llama3.1:8b",
            base_url="http://localhost:11434",
            enable_verifier=False,
            enable_agree_on_two=False,
            enable_reprompt_on_failure=False,
            temperature=0.2,
            seed=1337
        )
        llm_client = LLMClient(llm_config)
        
        # Create CDR pipeline
        pipeline = TestCDRPipeline(
            config=config,
            bluesky_client=bluesky_client,
            llm_client=llm_client
        )
        
        # Create test scenario
        scenario = ProximityTestScenario()
        
        # Setup output directory
        output_dir = Path("output/proximity_test")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print("📋 Test Configuration:")
        print(f"   Separation minima: {config.separation_min_nm} NM / {config.separation_min_ft} ft")
        print(f"   Cycle interval: {config.cycle_interval_seconds} seconds")
        print(f"   Max test time: {config.max_simulation_time_minutes} minutes")
        print(f"   LLM enabled: {config.llm_enabled}")
        
        # Run the test scenario
        print("\n🚀 Starting proximity test scenario...")
        result = pipeline.run_scenario(scenario, output_dir)
        
        print(f"\n📊 Test Results:")
        print(f"   Success: {result.success}")
        print(f"   Execution time: {result.execution_time_seconds:.2f} seconds")
        print(f"   Total conflicts: {result.total_conflicts}")
        print(f"   Successful resolutions: {result.successful_resolutions}")
        
        if result.error_message:
            print(f"   Error: {result.error_message}")
        
        # Verify aircraft states after test
        print(f"\n🔍 Final aircraft verification:")
        final_states = bluesky_client.get_aircraft_states()
        if final_states:
            for callsign, state in final_states.items():
                print(f"   📍 {callsign}: {state.latitude:.4f}, {state.longitude:.4f}, "
                      f"{state.altitude_ft}ft, {state.heading_deg}°")
        else:
            print("   ❌ No final aircraft states available")
        
        # Check if conflicts were resolved
        if result.total_conflicts > 0 and result.successful_resolutions > 0:
            print("✅ SUCCESS: Conflicts detected and resolved by LLM!")
            return True
        elif result.total_conflicts > 0:
            print("⚠️  PARTIAL: Conflicts detected but not resolved")
            return False
        else:
            print("❌ FAILURE: No conflicts detected (proximity test failed)")
            return False
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        # Cleanup
        if bluesky_client:
            try:
                bluesky_client.disconnect()
                print("🔌 BlueSky disconnected")
            except:
                pass


def manual_conflict_test():
    """Manual test to verify conflict detection without full pipeline"""
    print("\n🔧 Running manual conflict detection test...")
    
    # Test BlueSky connection
    client = test_bluesky_connection()
    if not client:
        return False
    
    try:
        # Create two aircraft manually
        print("✈️  Creating test aircraft...")
        
        # Aircraft 1
        success1 = client.create_aircraft(
            callsign="MANUAL01",
            aircraft_type="B738",
            lat=41.9786,
            lon=-87.9048,
            heading=90,
            altitude_ft=37000,
            speed_kt=450
        )
        
        # Aircraft 2 (close proximity)
        success2 = client.create_aircraft(
            callsign="MANUAL02", 
            aircraft_type="A320",
            lat=41.9786,        # Same latitude
            lon=-87.8898,       # About 1 NM apart
            heading=270,        # Opposite direction
            altitude_ft=37000,  # Same altitude
            speed_kt=420
        )
        
        if not (success1 and success2):
            print("❌ Failed to create aircraft")
            return False
        
        print("✅ Aircraft created successfully")
        
        # Start simulation
        client.op()
        time.sleep(2)  # Let aircraft move
        
        # Verify aircraft states
        if not verify_aircraft_states(client, ["MANUAL01", "MANUAL02"]):
            return False
        
        # Check for conflicts
        conflict = check_for_conflicts(client)
        if conflict:
            print(f"✅ SUCCESS: Manual conflict test detected proximity!")
            return True
        else:
            print("❌ No conflicts detected in manual test")
            return False
        
    except Exception as e:
        print(f"❌ Manual test error: {e}")
        return False
    
    finally:
        try:
            client.disconnect()
        except:
            pass


if __name__ == "__main__":
    print("🧪 Aircraft Proximity Test Suite")
    print("=" * 50)
    
    # First run manual test to verify basic functionality
    manual_success = manual_conflict_test()
    
    print("\n" + "=" * 50)
    
    # Then run full LLM test
    if manual_success:
        print("✅ Manual test passed, proceeding with LLM test...")
        llm_success = run_proximity_test()
        
        if llm_success:
            print("\n🎉 COMPLETE SUCCESS: Both manual and LLM tests passed!")
        else:
            print("\n⚠️  PARTIAL SUCCESS: Manual test passed but LLM test failed")
    else:
        print("❌ Manual test failed, skipping LLM test")
        print("💡 Check BlueSky connection and aircraft creation")
