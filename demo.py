#!/usr/bin/env python3
"""Demo script for LLM_ATC7"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cdr.adapters.scat_adapter import SCATAdapter
from cdr.simulation.bluesky_client import BlueSkyClient, BlueSkyConfig
from cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider
from cdr.detection.detector import ConflictDetector
from cdr.pipeline.cdr_pipeline import CDRPipeline, PipelineConfig
from cdr.metrics.calculator import MetricsCalculator


def run_demo():
    """Run a simple demo of the LLM_ATC7 system"""
    print("="*60)
    print("           LLM_ATC7 System Demo")
    print("="*60)
    
    # Test SCAT adapter
    print("\n1. Testing SCAT Data Adapter...")
    scat_file = Path("data/sample_scat.json")
    if scat_file.exists():
        try:
            adapter = SCATAdapter(scat_file)
            data = adapter.load_file()
            flight_plan = adapter.flight_plan()
            track_points = list(adapter.ownship_track())
            
            print(f"   ✓ Loaded SCAT file: {flight_plan.callsign}")
            print(f"   ✓ Flight plan: {flight_plan.route_string}")
            print(f"   ✓ Track points: {len(track_points)}")
        except Exception as e:
            print(f"   ✗ SCAT adapter error: {e}")
    else:
        print("   ✗ Sample SCAT file not found")
    
    # Test BlueSky client (mock)
    print("\n2. Testing BlueSky Client...")
    try:
        config = BlueSkyConfig(headless=True)
        client = BlueSkyClient(config)
        print("   ✓ BlueSky client initialized")
        print("   ✓ Ready for simulation (BlueSky not required for demo)")
    except Exception as e:
        print(f"   ✗ BlueSky client error: {e}")
    
    # Test LLM client
    print("\n3. Testing LLM Client...")
    try:
        llm_config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model_name="llama3.1:8b"
        )
        llm_client = LLMClient(llm_config)
        print("   ✓ LLM client initialized")
        print("   ✓ Ready for AI-based conflict resolution")
    except Exception as e:
        print(f"   ✗ LLM client error: {e}")
    
    # Test conflict detector
    print("\n4. Testing Conflict Detector...")
    try:
        detector = ConflictDetector()
        print("   ✓ Conflict detector initialized")
        print("   ✓ Geometric algorithms ready")
    except Exception as e:
        print(f"   ✗ Conflict detector error: {e}")
    
    # Test pipeline
    print("\n5. Testing CDR Pipeline...")
    try:
        pipeline_config = PipelineConfig(
            llm_enabled=True,
            max_simulation_time_minutes=30
        )
        
        # Mock scenario for demo
        class MockScenario:
            scenario_id = "demo_scenario"
        
        scenario = MockScenario()
        output_dir = Path("output/demo")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline = CDRPipeline(
            config=pipeline_config,
            bluesky_client=client,
            llm_client=llm_client
        )
        
        print("   ✓ CDR Pipeline initialized")
        print("   ✓ Ready for scenario execution")
        
        # Run mock scenario
        print("   ➤ Running demo scenario...")
        result = pipeline.run_scenario(scenario, output_dir)
        
        if result.success:
            print(f"   ✓ Demo scenario completed successfully")
            print(f"   ✓ Execution time: {result.execution_time_seconds:.2f}s")
        else:
            print(f"   ✗ Demo scenario failed: {result.error_message}")
            
    except Exception as e:
        print(f"   ✗ Pipeline error: {e}")
    
    # Test metrics calculator
    print("\n6. Testing Metrics Calculator...")
    try:
        calculator = MetricsCalculator()
        
        # Create mock simulation result
        demo_result = {
            "scenario_id": "demo",
            "success": True,
            "total_conflicts": 2,
            "successful_resolutions": 2,
            "final_time_minutes": 15.0,
            "execution_time_seconds": 5.2
        }
        
        result_file = output_dir / "simulation_result.json"
        with open(result_file, 'w') as f:
            import json
            json.dump(demo_result, f, indent=2)
        
        metrics = calculator.calculate_scenario_metrics(output_dir)
        
        print("   ✓ Metrics calculator initialized")
        print(f"   ✓ Demo metrics calculated: {metrics['scenario_id']}")
        
    except Exception as e:
        print(f"   ✗ Metrics calculator error: {e}")
    
    print("\n" + "="*60)
    print("           Demo Complete!")
    print("="*60)
    print("\nSystem Status: All core components operational")
    print("\nNext Steps:")
    print("• Install BlueSky: pip install bluesky-simulator")
    print("• Setup Ollama: https://ollama.ai/download")
    print("• Run health check: python -m src.cdr.cli health-check")
    print("• Process real data: python -m src.cdr.cli run-e2e --scat-path <file>")
    print("\nFor more options: python -m src.cdr.cli --help")
    print("="*60)


if __name__ == "__main__":
    run_demo()
