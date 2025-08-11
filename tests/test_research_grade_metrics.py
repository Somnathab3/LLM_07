#!/usr/bin/env python3
"""Test script for the complete research-grade metrics calculator"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cdr.metrics.calculator import MetricsCalculator, SafetyMetrics, PerformanceMetrics, LLMHallucinationMetrics


def create_mock_simulation_result() -> dict:
    """Create mock simulation result data for testing"""
    return {
        'scenario_id': 'test_scenario_001',
        'success': True,
        'total_conflicts': 3,
        'successful_resolutions': 2,
        'final_time_minutes': 45.0,
        'execution_time_seconds': 125.5,
        'avg_detection_time_ms': 45.0,
        'avg_resolution_time_ms': 180.0,
        'peak_memory_mb': 85.0,
        'avg_cpu_pct': 12.5,
        'false_positive_conflicts': 1,
        'false_negative_conflicts': 0,
        'actual_conflicts': 3,
        'deterministic_resolutions': 1,
        'conflicts': [
            {
                'conflict_id': 'CONFLICT_001',
                'timestamp': 300.0,
                'aircraft1': 'OWNSHIP',
                'aircraft2': 'INTRUDER1',
                'time_to_conflict': 480.0,
                'horizontal_distance': 4.2,
                'vertical_distance': 800.0,
                'severity': 'high',
                'resolved': True,
                'resolution_timestamp': 315.0
            },
            {
                'conflict_id': 'CONFLICT_002',
                'timestamp': 1200.0,
                'aircraft1': 'OWNSHIP',
                'aircraft2': 'INTRUDER2',
                'time_to_conflict': 360.0,
                'horizontal_distance': 6.8,
                'vertical_distance': 1200.0,
                'severity': 'medium',
                'resolved': True,
                'resolution_timestamp': 1225.0
            },
            {
                'conflict_id': 'CONFLICT_003',
                'timestamp': 2100.0,
                'aircraft1': 'INTRUDER1',
                'aircraft2': 'INTRUDER2',
                'time_to_conflict': 180.0,
                'horizontal_distance': 3.5,
                'vertical_distance': 500.0,
                'severity': 'high',
                'resolved': False
            }
        ],
        'resolution_history': [
            {
                'conflict_id': 'CONFLICT_001',
                'timestamp': 315.0,
                'detection_timestamp': 300.0,
                'resolution_time_s': 15.0,
                'method': 'llm',
                'success': True
            },
            {
                'conflict_id': 'CONFLICT_002',
                'timestamp': 1225.0,
                'detection_timestamp': 1200.0,
                'resolution_time_s': 25.0,
                'method': 'geometric',
                'success': True
            }
        ]
    }


def create_mock_trajectory_data() -> list:
    """Create mock trajectory data for testing"""
    trajectory_data = []
    
    # Create trajectory snapshots over time
    for t in range(0, 3000, 60):  # Every minute for 50 minutes
        snapshot = {
            'timestamp': t,
            'time_minutes': t / 60.0,
            'aircraft': {
                'OWNSHIP': {
                    'callsign': 'OWNSHIP',
                    'latitude': 41.978 + (t / 10000.0),  # Slight movement
                    'longitude': -87.904 + (t / 15000.0),
                    'altitude_ft': 37000 + np.random.normal(0, 100),
                    'heading_deg': 270 + np.random.normal(0, 2),
                    'speed_kt': 450 + np.random.normal(0, 10)
                },
                'INTRUDER1': {
                    'callsign': 'INTRUDER1',
                    'latitude': 42.0 + (t / 12000.0),
                    'longitude': -87.9 - (t / 18000.0),
                    'altitude_ft': 37000 + np.random.normal(0, 150),
                    'heading_deg': 90 + np.random.normal(0, 3),
                    'speed_kt': 420 + np.random.normal(0, 15)
                },
                'INTRUDER2': {
                    'callsign': 'INTRUDER2',
                    'latitude': 41.95 - (t / 20000.0),
                    'longitude': -87.95 + (t / 25000.0),
                    'altitude_ft': 36500 + np.random.normal(0, 200),
                    'heading_deg': 180 + np.random.normal(0, 4),
                    'speed_kt': 380 + np.random.normal(0, 20)
                }
            }
        }
        trajectory_data.append(snapshot)
    
    return trajectory_data


def create_mock_llm_responses() -> list:
    """Create mock LLM responses for hallucination testing"""
    return [
        {
            'scenario_id': 'test_scenario_001',
            'timestamp': 300.0,
            'resolution_type': 'heading',
            'new_heading': 285.0,
            'confidence': 0.85,
            'reasoning': 'Turn right 15 degrees to avoid conflict'
        },
        {
            'scenario_id': 'test_scenario_001', 
            'timestamp': 1200.0,
            'resolution_type': 'altitude',
            'new_altitude': 38000.0,
            'confidence': 0.75,
            'reasoning': 'Climb 1000 feet for vertical separation'
        },
        {
            'scenario_id': 'test_scenario_001',
            'timestamp': 2100.0,
            'resolution_type': 'heading',
            'new_heading': 320.0,  # This will be a hallucination
            'confidence': 0.90,    # Overconfident
            'reasoning': 'Sharp left turn to avoid traffic'
        }
    ]


def create_mock_ground_truth() -> list:
    """Create mock ground truth data for comparison"""
    return [
        {
            'scenario_id': 'test_scenario_001',
            'timestamp': 300.0,
            'resolution_type': 'heading',
            'new_heading': 290.0,  # Slightly different from LLM
            'confidence': 0.95,
            'optimal_solution': True
        },
        {
            'scenario_id': 'test_scenario_001',
            'timestamp': 1200.0,
            'resolution_type': 'altitude',
            'new_altitude': 38000.0,  # Same as LLM
            'confidence': 0.90,
            'optimal_solution': True
        },
        {
            'scenario_id': 'test_scenario_001',
            'timestamp': 2100.0,
            'resolution_type': 'heading',
            'new_heading': 280.0,  # Very different from LLM (hallucination)
            'confidence': 0.85,
            'optimal_solution': True
        }
    ]


def test_safety_metrics():
    """Test comprehensive safety metrics calculation"""
    print("Testing Safety Metrics Calculation...")
    
    calculator = MetricsCalculator(separation_min_nm=5.0, separation_min_ft=1000.0)
    simulation_result = create_mock_simulation_result()
    trajectory_data = create_mock_trajectory_data()
    
    safety_metrics = calculator._calculate_safety_metrics(simulation_result, trajectory_data)
    
    print(f"✅ Total conflicts: {safety_metrics.total_conflicts}")
    print(f"✅ Resolved conflicts: {safety_metrics.resolved_conflicts}")
    print(f"✅ TBAS score: {safety_metrics.tbas_score:.3f}")
    print(f"✅ LAT score: {safety_metrics.lat_score:.3f}")
    print(f"✅ RAT score: {safety_metrics.rat_score:.3f}")
    print(f"✅ Safety violations: {safety_metrics.safety_violations}")
    print(f"✅ Near misses: {safety_metrics.near_miss_count}")
    print(f"✅ Min separation: {safety_metrics.min_separation_achieved_nm:.2f} NM")
    print(f"✅ Safety margin statistics: {len(safety_metrics.safety_margin_statistics)} metrics")
    
    # Test confidence intervals
    if safety_metrics.confidence_intervals:
        print(f"✅ Confidence intervals calculated: {list(safety_metrics.confidence_intervals.keys())}")
    
    assert isinstance(safety_metrics, SafetyMetrics)
    assert 0.0 <= safety_metrics.tbas_score <= 1.0
    assert 0.0 <= safety_metrics.lat_score <= 1.0
    assert 0.0 <= safety_metrics.rat_score <= 1.0
    
    print("✅ Safety metrics calculation passed!")
    return True


def test_performance_metrics():
    """Test enhanced performance metrics calculation"""
    print("\nTesting Performance Metrics Calculation...")
    
    calculator = MetricsCalculator()
    simulation_result = create_mock_simulation_result()
    trajectory_data = create_mock_trajectory_data()
    
    performance_metrics = calculator._calculate_performance_metrics(simulation_result, trajectory_data)
    
    print(f"✅ Avg resolution time: {performance_metrics.avg_resolution_time_s:.1f}s")
    print(f"✅ LLM success rate: {performance_metrics.llm_success_rate:.3f}")
    print(f"✅ Fallback rate: {performance_metrics.deterministic_fallback_rate:.3f}")
    print(f"✅ Detection accuracy: {performance_metrics.detection_accuracy:.3f}")
    print(f"✅ False positive rate: {performance_metrics.false_positive_rate:.3f}")
    print(f"✅ False negative rate: {performance_metrics.false_negative_rate:.3f}")
    print(f"✅ Computational efficiency metrics: {len(performance_metrics.computational_efficiency)}")
    
    assert isinstance(performance_metrics, PerformanceMetrics)
    assert 0.0 <= performance_metrics.llm_success_rate <= 1.0
    assert 0.0 <= performance_metrics.detection_accuracy <= 1.0
    assert performance_metrics.avg_resolution_time_s > 0
    
    print("✅ Performance metrics calculation passed!")
    return True


def test_hallucination_metrics():
    """Test LLM hallucination quantification"""
    print("\nTesting LLM Hallucination Metrics...")
    
    calculator = MetricsCalculator()
    llm_responses = create_mock_llm_responses()
    ground_truth = create_mock_ground_truth()
    
    hallucination_metrics = calculator.calculate_hallucination_metrics(llm_responses, ground_truth)
    
    print(f"✅ Hallucination rate: {hallucination_metrics.hallucination_rate:.3f}")
    print(f"✅ Geometric deviation score: {hallucination_metrics.geometric_deviation_score:.3f}")
    print(f"✅ Safety impact score: {hallucination_metrics.safety_impact_score:.3f}")
    print(f"✅ Confidence correlation: {hallucination_metrics.confidence_correlation:.3f}")
    print(f"✅ Hallucination categories: {hallucination_metrics.hallucination_categories}")
    print(f"✅ Statistical significance: {len(hallucination_metrics.statistical_significance)} tests")
    
    assert isinstance(hallucination_metrics, LLMHallucinationMetrics)
    assert 0.0 <= hallucination_metrics.hallucination_rate <= 1.0
    assert 0.0 <= hallucination_metrics.geometric_deviation_score <= 1.0
    assert 0.0 <= hallucination_metrics.safety_impact_score <= 1.0
    assert -1.0 <= hallucination_metrics.confidence_correlation <= 1.0
    
    print("✅ Hallucination metrics calculation passed!")
    return True


def test_tbas_score_calculation():
    """Test detailed TBAS score calculation"""
    print("\nTesting TBAS Score Calculation...")
    
    calculator = MetricsCalculator()
    simulation_result = create_mock_simulation_result()
    trajectory_data = create_mock_trajectory_data()
    
    tbas_score = calculator._calculate_tbas_score(simulation_result, trajectory_data)
    
    print(f"✅ TBAS score: {tbas_score:.4f}")
    
    # Test with different scenarios
    # Fast response scenario
    fast_result = simulation_result.copy()
    fast_result['conflicts'][0]['resolution_timestamp'] = fast_result['conflicts'][0]['timestamp'] + 5  # 5s response
    fast_tbas = calculator._calculate_tbas_score(fast_result, trajectory_data)
    
    # Slow response scenario  
    slow_result = simulation_result.copy()
    slow_result['conflicts'][0]['resolution_timestamp'] = slow_result['conflicts'][0]['timestamp'] + 45  # 45s response
    slow_tbas = calculator._calculate_tbas_score(slow_result, trajectory_data)
    
    print(f"✅ Fast response TBAS: {fast_tbas:.4f}")
    print(f"✅ Slow response TBAS: {slow_tbas:.4f}")
    
    assert fast_tbas >= slow_tbas, "Fast response should have higher TBAS score"
    assert 0.0 <= tbas_score <= 1.0
    
    print("✅ TBAS score calculation passed!")
    return True


def test_lat_score_calculation():
    """Test detailed LAT score calculation"""
    print("\nTesting LAT Score Calculation...")
    
    calculator = MetricsCalculator()
    simulation_result = create_mock_simulation_result()
    trajectory_data = create_mock_trajectory_data()
    
    lat_score = calculator._calculate_lat_score(simulation_result, trajectory_data)
    
    print(f"✅ LAT score: {lat_score:.4f}")
    
    assert 0.0 <= lat_score <= 1.0
    
    print("✅ LAT score calculation passed!")
    return True


def test_safety_margins_calculation():
    """Test comprehensive safety margins calculation"""
    print("\nTesting Safety Margins Calculation...")
    
    calculator = MetricsCalculator()
    trajectory_data = create_mock_trajectory_data()
    
    safety_margins = calculator._calculate_safety_margins(trajectory_data)
    
    print(f"✅ Min horizontal separation: {safety_margins['min_horizontal_separation']:.2f} NM")
    print(f"✅ Avg horizontal separation: {safety_margins['avg_horizontal_separation']:.2f} NM")
    print(f"✅ Safety buffer violations: {safety_margins['safety_buffer_violations']}")
    print(f"✅ Critical proximity events: {safety_margins['critical_proximity_events']}")
    print(f"✅ Safety margin efficiency: {safety_margins['safety_margin_efficiency']:.3f}")
    
    assert 'min_horizontal_separation' in safety_margins
    assert 'avg_horizontal_separation' in safety_margins
    assert 'safety_margin_efficiency' in safety_margins
    assert safety_margins['safety_buffer_violations'] >= 0
    
    print("✅ Safety margins calculation passed!")
    return True


def test_complete_scenario_metrics():
    """Test complete scenario metrics calculation"""
    print("\nTesting Complete Scenario Metrics...")
    
    calculator = MetricsCalculator()
    
    # Create test data files
    test_dir = Path("test_metrics_output")
    test_dir.mkdir(exist_ok=True)
    
    simulation_result = create_mock_simulation_result()
    simulation_result['llm_responses'] = create_mock_llm_responses()
    simulation_result['ground_truth'] = create_mock_ground_truth()
    
    trajectory_data = create_mock_trajectory_data()
    
    # Save test files
    with open(test_dir / "simulation_result.json", 'w') as f:
        json.dump(simulation_result, f, indent=2)
    
    with open(test_dir / "trajectories.jsonl", 'w') as f:
        for snapshot in trajectory_data:
            f.write(json.dumps(snapshot) + '\n')
    
    # Calculate complete metrics
    metrics = calculator.calculate_scenario_metrics(test_dir)
    
    print(f"✅ Scenario ID: {metrics['scenario_id']}")
    print(f"✅ Success: {metrics['success']}")
    print(f"✅ Simulation time: {metrics['simulation_time_minutes']:.1f} min")
    print(f"✅ Safety metrics included: {'safety_metrics' in metrics}")
    print(f"✅ Performance metrics included: {'performance_metrics' in metrics}")
    print(f"✅ Hallucination metrics included: {'hallucination_metrics' in metrics}")
    
    if 'hallucination_metrics' in metrics:
        hm = metrics['hallucination_metrics']
        print(f"   - Hallucination rate: {hm['hallucination_rate']:.3f}")
        print(f"   - Safety impact: {hm['safety_impact_score']:.3f}")
    
    # Cleanup test files
    import shutil
    shutil.rmtree(test_dir)
    
    assert metrics['success'] == True
    assert 'safety_metrics' in metrics
    assert 'performance_metrics' in metrics
    
    print("✅ Complete scenario metrics calculation passed!")
    return True


def main():
    """Run all comprehensive metrics tests"""
    print("=" * 70)
    print("RESEARCH-GRADE METRICS CALCULATOR IMPLEMENTATION TEST")
    print("=" * 70)
    
    try:
        # Test individual components
        test_safety_metrics()
        test_performance_metrics()
        test_hallucination_metrics()
        test_tbas_score_calculation()
        test_lat_score_calculation()
        test_safety_margins_calculation()
        test_complete_scenario_metrics()
        
        print("\n" + "=" * 70)
        print("🎉 ALL METRICS TESTS PASSED!")
        print("=" * 70)
        
        print("\nImplemented Research-Grade Features:")
        print("✅ TBAS (Time-Based Avoidance System) scoring")
        print("✅ LAT (Look-Ahead Time) effectiveness analysis")
        print("✅ Comprehensive safety margin statistics")
        print("✅ LLM hallucination quantification & categorization")
        print("✅ Statistical significance testing")
        print("✅ Confidence interval calculations")
        print("✅ Performance and computational efficiency metrics")
        print("✅ False positive/negative rate analysis")
        print("✅ Safety impact assessment")
        print("✅ Geometric deviation scoring")
        
        print("\nKey Research Capabilities:")
        print("• Industry standard benchmarking")
        print("• Statistical confidence analysis")
        print("• Multi-dimensional safety assessment")
        print("• AI/LLM reliability quantification")
        print("• Real-time performance monitoring")
        print("• Research-grade experimental validation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
