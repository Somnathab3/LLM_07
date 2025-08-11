#!/usr/bin/env python3
"""
Test script for CLI fixes
"""

import sys
import tempfile
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_cli_imports():
    """Test that all CLI imports work"""
    print("Testing CLI imports...")
    
    try:
        from src.cdr.cli import LLMATCCLIConsole
        print("✓ CLI imports successful")
        return True
    except Exception as e:
        print(f"✗ CLI import failed: {e}")
        return False

def test_conflict_context():
    """Test ConflictContext creation"""
    print("Testing ConflictContext creation...")
    
    try:
        from src.cdr.ai.llm_client import ConflictContext
        
        # Test creating ConflictContext with correct parameters
        context = ConflictContext(
            ownship_callsign="TEST1",
            ownship_state={'latitude': 0, 'longitude': 0, 'altitude_ft': 35000},
            intruders=[{'latitude': 1, 'longitude': 1, 'altitude_ft': 35000}],
            scenario_time=0.0,
            lookahead_minutes=10.0,
            constraints={},
            nearby_traffic=[]
        )
        print("✓ ConflictContext creation successful")
        return True
    except Exception as e:
        print(f"✗ ConflictContext creation failed: {e}")
        return False

def test_simulation_result():
    """Test SimulationResult serialization"""
    print("Testing SimulationResult serialization...")
    
    try:
        from src.cdr.pipeline.cdr_pipeline import SimulationResult
        
        # Create a SimulationResult instance
        result = SimulationResult(
            scenario_id="test",
            success=True,
            total_conflicts=5,
            successful_resolutions=3
        )
        
        # Test to_dict conversion
        result_dict = result.to_dict()
        
        # Test JSON serialization
        json_str = json.dumps(result_dict, indent=2)
        print("✓ SimulationResult serialization successful")
        return True
    except Exception as e:
        print(f"✗ SimulationResult serialization failed: {e}")
        return False

def test_visualization_imports():
    """Test visualization module imports"""
    print("Testing visualization imports...")
    
    try:
        from src.cdr.visualization.visualization import create_visualizer
        from src.cdr.visualization.config import DEFAULT_CONFIG
        from src.cdr.visualization.models import Aircraft, Position
        
        # Try to create a visualizer
        visualizer = create_visualizer()
        print("✓ Visualization imports successful")
        return True
    except Exception as e:
        print(f"✗ Visualization imports failed: {e}")
        return False

def test_cli_health_check():
    """Test CLI health check command"""
    print("Testing CLI health check...")
    
    try:
        from src.cdr.cli import LLMATCCLIConsole
        
        cli = LLMATCCLIConsole()
        result = cli.run(['health-check'])
        
        print(f"✓ CLI health check completed with exit code: {result}")
        return True
    except Exception as e:
        print(f"✗ CLI health check failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("CLI FIXES VALIDATION")
    print("="*60)
    
    tests = [
        test_cli_imports,
        test_conflict_context, 
        test_simulation_result,
        test_visualization_imports,
        test_cli_health_check
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
        print()
    
    print("="*60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All CLI fixes validated successfully!")
        return 0
    else:
        print("❌ Some issues remain")
        return 1

if __name__ == "__main__":
    sys.exit(main())
