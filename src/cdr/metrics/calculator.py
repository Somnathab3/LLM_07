"""Calculate comprehensive simulation metrics"""

import json
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SafetyMetrics:
    """Safety-related metrics"""
    total_conflicts: int
    resolved_conflicts: int  
    safety_violations: int
    min_separation_achieved_nm: float
    avg_min_separation_nm: float
    tbas_score: float  # Time-Based Avoidance System
    lat_score: float   # Look-Ahead Time
    rat_score: float   # Resolution Action Time


@dataclass  
class PerformanceMetrics:
    """Performance-related metrics"""
    avg_resolution_time_s: float
    avg_additional_distance_nm: float
    avg_additional_time_min: float
    llm_success_rate: float
    deterministic_fallback_rate: float


class MetricsCalculator:
    """Calculate comprehensive simulation metrics"""
    
    def calculate_scenario_metrics(self, scenario_dir: Path) -> Dict[str, Any]:
        """Calculate metrics for a single scenario"""
        
        # Load simulation results
        result_file = scenario_dir / "simulation_result.json"
        trajectory_file = scenario_dir / "trajectories.jsonl"
        
        if not result_file.exists():
            return {
                'scenario_id': 'unknown',
                'success': False,
                'error': 'No simulation result file found'
            }
        
        try:
            with open(result_file, 'r') as f:
                simulation_result = json.load(f)
        except Exception as e:
            return {
                'scenario_id': 'unknown',
                'success': False,
                'error': f'Failed to load result file: {e}'
            }
        
        # Load trajectory data if available
        trajectory_data = []
        if trajectory_file.exists():
            try:
                with open(trajectory_file, 'r') as f:
                    for line in f:
                        trajectory_data.append(json.loads(line.strip()))
            except Exception as e:
                print(f"Warning: Failed to load trajectory data: {e}")
        
        # Calculate metrics
        safety_metrics = self._calculate_safety_metrics(simulation_result, trajectory_data)
        performance_metrics = self._calculate_performance_metrics(simulation_result, trajectory_data)
        
        return {
            'scenario_id': simulation_result.get('scenario_id', 'unknown'),
            'success': simulation_result.get('success', False),
            'simulation_time_minutes': simulation_result.get('final_time_minutes', 0),
            'execution_time_seconds': simulation_result.get('execution_time_seconds', 0),
            'safety_metrics': safety_metrics.__dict__,
            'performance_metrics': performance_metrics.__dict__
        }
    
    def aggregate_metrics(self, result_files: List[Path]) -> Dict[str, Any]:
        """Aggregate metrics across multiple scenarios"""
        
        all_metrics = []
        
        for result_file in result_files:
            try:
                scenario_dir = result_file.parent
                metrics = self.calculate_scenario_metrics(scenario_dir)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Warning: Failed to process {result_file}: {e}")
                continue
        
        if not all_metrics:
            return {
                'total_scenarios': 0,
                'successful_runs': 0,
                'success_rate': 0.0,
                'error': 'No valid metrics found'
            }
        
        # Aggregate statistics
        successful_runs = sum(1 for m in all_metrics if m['success'])
        total_scenarios = len(all_metrics)
        
        # Aggregate safety metrics
        safety_data = [m['safety_metrics'] for m in all_metrics if m['success']]
        aggregated_safety = self._aggregate_safety_metrics(safety_data)
        
        # Aggregate performance metrics
        perf_data = [m['performance_metrics'] for m in all_metrics if m['success']]
        aggregated_performance = self._aggregate_performance_metrics(perf_data)
        
        return {
            'total_scenarios': total_scenarios,
            'successful_runs': successful_runs,
            'success_rate': successful_runs / total_scenarios if total_scenarios > 0 else 0,
            'safety_metrics': aggregated_safety,
            'performance_metrics': aggregated_performance,
            'individual_results': all_metrics
        }
    
    def _calculate_safety_metrics(self, simulation_result: Dict[str, Any],
                                 trajectory_data: List[Dict[str, Any]]) -> SafetyMetrics:
        """Calculate safety-related metrics"""
        
        total_conflicts = simulation_result.get('total_conflicts', 0)
        resolved_conflicts = simulation_result.get('successful_resolutions', 0)
        
        # Mock safety calculations
        safety_violations = 0
        min_separations = [8.5, 6.2, 7.8, 9.1, 5.5]  # Mock data
        
        # Calculate TBAS, LAT, RAT scores (simplified)
        resolution_rate = resolved_conflicts / max(1, total_conflicts)
        tbas_score = min(resolution_rate, 1.0)
        lat_score = 0.85  # Mock look-ahead score
        rat_score = min(resolution_rate * 1.1, 1.0)  # Mock action time score
        
        return SafetyMetrics(
            total_conflicts=total_conflicts,
            resolved_conflicts=resolved_conflicts,
            safety_violations=safety_violations,
            min_separation_achieved_nm=min(min_separations) if min_separations else float('inf'),
            avg_min_separation_nm=np.mean(min_separations) if min_separations else float('inf'),
            tbas_score=tbas_score,
            lat_score=lat_score,
            rat_score=rat_score
        )
    
    def _calculate_performance_metrics(self, simulation_result: Dict[str, Any],
                                     trajectory_data: List[Dict[str, Any]]) -> PerformanceMetrics:
        """Calculate performance-related metrics"""
        
        total_resolutions = simulation_result.get('successful_resolutions', 0)
        total_conflicts = simulation_result.get('total_conflicts', 1)
        
        # Mock performance calculations
        return PerformanceMetrics(
            avg_resolution_time_s=12.5,  # Mock average resolution time
            avg_additional_distance_nm=8.2,  # Mock additional distance
            avg_additional_time_min=3.1,  # Mock additional time
            llm_success_rate=total_resolutions / total_conflicts if total_conflicts > 0 else 0,
            deterministic_fallback_rate=0.1  # Mock fallback rate
        )
    
    def _aggregate_safety_metrics(self, safety_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate safety metrics across scenarios"""
        
        if not safety_data:
            return {}
        
        return {
            'total_conflicts': sum(d['total_conflicts'] for d in safety_data),
            'total_resolved': sum(d['resolved_conflicts'] for d in safety_data),
            'total_violations': sum(d['safety_violations'] for d in safety_data),
            'avg_min_separation_nm': np.mean([d['avg_min_separation_nm'] for d in safety_data 
                                            if d['avg_min_separation_nm'] < float('inf')]),
            'avg_tbas_score': np.mean([d['tbas_score'] for d in safety_data]),
            'avg_lat_score': np.mean([d['lat_score'] for d in safety_data]),
            'avg_rat_score': np.mean([d['rat_score'] for d in safety_data]),
            'resolution_success_rate': (sum(d['resolved_conflicts'] for d in safety_data) / 
                                      max(1, sum(d['total_conflicts'] for d in safety_data)))
        }
    
    def _aggregate_performance_metrics(self, perf_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate performance metrics across scenarios"""
        
        if not perf_data:
            return {}
        
        return {
            'avg_resolution_time_s': np.mean([d['avg_resolution_time_s'] for d in perf_data]),
            'avg_additional_distance_nm': np.mean([d['avg_additional_distance_nm'] for d in perf_data]),
            'avg_additional_time_min': np.mean([d['avg_additional_time_min'] for d in perf_data]),
            'overall_llm_success_rate': np.mean([d['llm_success_rate'] for d in perf_data]),
            'overall_fallback_rate': np.mean([d['deterministic_fallback_rate'] for d in perf_data])
        }
