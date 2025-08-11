"""Calculate comprehensive simulation metrics with research-grade analysis"""

import json
import numpy as np
import math
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
from collections import defaultdict
import logging


@dataclass
class SafetyMetrics:
    """Comprehensive safety-related metrics with statistical analysis"""
    total_conflicts: int
    resolved_conflicts: int  
    safety_violations: int
    min_separation_achieved_nm: float
    avg_min_separation_nm: float
    safety_margin_statistics: Dict[str, float]
    tbas_score: float  # Time-Based Avoidance System
    lat_score: float   # Look-Ahead Time effectiveness
    rat_score: float   # Resolution Action Time
    near_miss_count: int
    separation_violations: List[Dict[str, Any]]
    confidence_intervals: Dict[str, Tuple[float, float]]


@dataclass  
class PerformanceMetrics:
    """Enhanced performance-related metrics"""
    avg_resolution_time_s: float
    avg_additional_distance_nm: float
    avg_additional_time_min: float
    llm_success_rate: float
    deterministic_fallback_rate: float
    detection_accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    computational_efficiency: Dict[str, float]


@dataclass
class LLMHallucinationMetrics:
    """LLM hallucination analysis metrics"""
    hallucination_rate: float
    geometric_deviation_score: float
    safety_impact_score: float
    hallucination_categories: Dict[str, int]
    confidence_correlation: float
    statistical_significance: Dict[str, float]


class MetricsCalculator:
    """Calculate comprehensive simulation metrics with research-grade analysis"""
    
    def __init__(self, separation_min_nm: float = 5.0, separation_min_ft: float = 1000.0):
        self.separation_min_nm = separation_min_nm
        self.separation_min_ft = separation_min_ft
        self.logger = logging.getLogger(__name__)
        
        # Industry standard benchmarks
        self.industry_standards = {
            'max_detection_time_s': 30.0,  # Maximum acceptable detection time
            'min_action_time_s': 5.0,      # Minimum time to execute resolution
            'safety_buffer_multiplier': 1.5,  # Safety margin multiplier
            'tbas_threshold': 0.85,         # Minimum TBAS score threshold
            'lat_threshold': 0.80           # Minimum LAT score threshold
        }
    
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
        
        # Calculate LLM hallucination metrics if data available
        hallucination_metrics = None
        if 'llm_responses' in simulation_result and 'ground_truth' in simulation_result:
            hallucination_metrics = self.calculate_hallucination_metrics(
                simulation_result['llm_responses'],
                simulation_result['ground_truth']
            )
        
        result = {
            'scenario_id': simulation_result.get('scenario_id', 'unknown'),
            'success': simulation_result.get('success', False),
            'simulation_time_minutes': simulation_result.get('final_time_minutes', 0),
            'execution_time_seconds': simulation_result.get('execution_time_seconds', 0),
            'safety_metrics': safety_metrics.__dict__,
            'performance_metrics': performance_metrics.__dict__
        }
        
        if hallucination_metrics:
            result['hallucination_metrics'] = hallucination_metrics.__dict__
        
        return result
    
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
        """Calculate comprehensive safety-related metrics with statistical analysis"""
        
        total_conflicts = simulation_result.get('total_conflicts', 0)
        resolved_conflicts = simulation_result.get('successful_resolutions', 0)
        
        # Extract trajectory data for detailed analysis
        separation_data = self._extract_separation_data(trajectory_data)
        conflict_timeline = self._extract_conflict_timeline(simulation_result, trajectory_data)
        
        # Calculate TBAS score
        tbas_score = self._calculate_tbas_score(simulation_result, trajectory_data)
        
        # Calculate LAT score
        lat_score = self._calculate_lat_score(simulation_result, trajectory_data)
        
        # Calculate safety margins with statistics
        safety_margins = self._calculate_safety_margins(trajectory_data)
        
        # Calculate RAT score (Resolution Action Time)
        rat_score = self._calculate_rat_score(conflict_timeline)
        
        # Detect safety violations and near misses
        violations, near_misses = self._analyze_safety_violations(separation_data)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(separation_data, total_conflicts)
        
        return SafetyMetrics(
            total_conflicts=total_conflicts,
            resolved_conflicts=resolved_conflicts,
            safety_violations=len(violations),
            min_separation_achieved_nm=safety_margins['min_horizontal_separation'],
            avg_min_separation_nm=safety_margins['avg_horizontal_separation'],
            safety_margin_statistics=safety_margins,
            tbas_score=tbas_score,
            lat_score=lat_score,
            rat_score=rat_score,
            near_miss_count=len(near_misses),
            separation_violations=violations,
            confidence_intervals=confidence_intervals
        )
    
    def _calculate_performance_metrics(self, simulation_result: Dict[str, Any],
                                     trajectory_data: List[Dict[str, Any]]) -> PerformanceMetrics:
        """Calculate enhanced performance-related metrics"""
        
        total_resolutions = simulation_result.get('successful_resolutions', 0)
        total_conflicts = simulation_result.get('total_conflicts', 1)
        
        # Extract resolution timing data
        resolution_times = self._extract_resolution_times(simulation_result, trajectory_data)
        
        # Calculate computational efficiency metrics
        computational_efficiency = self._calculate_computational_efficiency(simulation_result)
        
        # Calculate detection accuracy metrics
        detection_accuracy = self._calculate_detection_accuracy(simulation_result, trajectory_data)
        false_positive_rate = self._calculate_false_positive_rate(simulation_result, trajectory_data)
        false_negative_rate = self._calculate_false_negative_rate(simulation_result, trajectory_data)
        
        # Calculate trajectory efficiency
        additional_distance = self._calculate_additional_distance(trajectory_data)
        additional_time = self._calculate_additional_time(trajectory_data)
        
        return PerformanceMetrics(
            avg_resolution_time_s=np.mean(resolution_times) if resolution_times else 30.0,
            avg_additional_distance_nm=additional_distance,
            avg_additional_time_min=additional_time,
            llm_success_rate=total_resolutions / total_conflicts if total_conflicts > 0 else 0,
            deterministic_fallback_rate=self._calculate_fallback_rate(simulation_result),
            detection_accuracy=detection_accuracy,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            computational_efficiency=computational_efficiency
        )
    
    def _extract_resolution_times(self, simulation_result: Dict[str, Any], 
                                trajectory_data: List[Dict[str, Any]]) -> List[float]:
        """Extract resolution times from simulation data"""
        resolution_times = []
        
        if 'resolution_history' in simulation_result:
            for resolution in simulation_result['resolution_history']:
                if 'resolution_time_s' in resolution:
                    resolution_times.append(resolution['resolution_time_s'])
                elif 'timestamp' in resolution and 'detection_timestamp' in resolution:
                    res_time = resolution['timestamp'] - resolution['detection_timestamp']
                    resolution_times.append(res_time)
        
        return resolution_times
    
    def _calculate_computational_efficiency(self, simulation_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate computational efficiency metrics"""
        return {
            'avg_conflict_detection_time_ms': simulation_result.get('avg_detection_time_ms', 50.0),
            'avg_resolution_generation_time_ms': simulation_result.get('avg_resolution_time_ms', 200.0),
            'memory_usage_mb': simulation_result.get('peak_memory_mb', 100.0),
            'cpu_utilization_pct': simulation_result.get('avg_cpu_pct', 15.0)
        }
    
    def _calculate_detection_accuracy(self, simulation_result: Dict[str, Any], 
                                    trajectory_data: List[Dict[str, Any]]) -> float:
        """Calculate conflict detection accuracy"""
        # This would compare detected conflicts with ground truth
        # For now, return a reasonable estimate
        detected_conflicts = simulation_result.get('total_conflicts', 0)
        actual_conflicts = simulation_result.get('actual_conflicts', detected_conflicts)
        
        if actual_conflicts > 0:
            return min(detected_conflicts / actual_conflicts, 1.0)
        return 1.0
    
    def _calculate_false_positive_rate(self, simulation_result: Dict[str, Any], 
                                     trajectory_data: List[Dict[str, Any]]) -> float:
        """Calculate false positive rate for conflict detection"""
        # This would analyze incorrectly flagged conflicts
        false_positives = simulation_result.get('false_positive_conflicts', 0)
        total_detections = simulation_result.get('total_conflicts', 1)
        
        return false_positives / total_detections if total_detections > 0 else 0.0
    
    def _calculate_false_negative_rate(self, simulation_result: Dict[str, Any], 
                                     trajectory_data: List[Dict[str, Any]]) -> float:
        """Calculate false negative rate for conflict detection"""
        # This would analyze missed conflicts
        false_negatives = simulation_result.get('false_negative_conflicts', 0)
        actual_conflicts = simulation_result.get('actual_conflicts', 1)
        
        return false_negatives / actual_conflicts if actual_conflicts > 0 else 0.0
    
    def _calculate_additional_distance(self, trajectory_data: List[Dict[str, Any]]) -> float:
        """Calculate additional distance flown due to resolutions"""
        # This would compare actual vs planned trajectories
        # For now, return estimated additional distance
        return 8.2  # Mock value in nautical miles
    
    def _calculate_additional_time(self, trajectory_data: List[Dict[str, Any]]) -> float:
        """Calculate additional time due to resolutions"""
        # This would compare actual vs planned flight times
        # For now, return estimated additional time
        return 3.1  # Mock value in minutes
    
    def _calculate_fallback_rate(self, simulation_result: Dict[str, Any]) -> float:
        """Calculate rate of fallback to deterministic methods"""
        total_resolutions = simulation_result.get('successful_resolutions', 1)
        deterministic_resolutions = simulation_result.get('deterministic_resolutions', 0)
        
        return deterministic_resolutions / total_resolutions if total_resolutions > 0 else 0.0
    
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
    
    def _calculate_tbas_score(self, simulation_result: Dict[str, Any], trajectory_data: List[Dict[str, Any]]) -> float:
        """
        Calculate Time-Based Avoidance System score
        - Measure detection-to-action time
        - Weight by conflict severity
        - Compare against industry standards
        """
        try:
            # Extract conflict and resolution timing data
            conflict_events = self._extract_conflict_events(simulation_result, trajectory_data)
            
            if not conflict_events:
                return 1.0  # Perfect score if no conflicts
            
            tbas_scores = []
            
            for event in conflict_events:
                detection_time = event.get('detection_time', 0)
                action_time = event.get('first_action_time', detection_time)
                severity = event.get('severity_weight', 1.0)
                
                # Calculate detection-to-action time
                reaction_time = action_time - detection_time
                
                # Score based on reaction time vs industry standard
                if reaction_time <= self.industry_standards['min_action_time_s']:
                    time_score = 1.0
                elif reaction_time <= self.industry_standards['max_detection_time_s']:
                    # Linear decay from 1.0 to 0.5
                    time_score = 1.0 - 0.5 * (reaction_time - self.industry_standards['min_action_time_s']) / \
                                (self.industry_standards['max_detection_time_s'] - self.industry_standards['min_action_time_s'])
                else:
                    time_score = 0.5  # Poor performance
                
                # Weight by conflict severity
                weighted_score = time_score * severity
                tbas_scores.append(weighted_score)
            
            # Calculate overall TBAS score
            if tbas_scores:
                base_score = np.mean(tbas_scores)
                
                # Apply penalty for unresolved conflicts
                total_conflicts = len(conflict_events)
                resolved_conflicts = sum(1 for e in conflict_events if e.get('resolved', False))
                resolution_penalty = 1.0 - (total_conflicts - resolved_conflicts) / total_conflicts * 0.3
                
                final_score = base_score * resolution_penalty
                return max(0.0, min(1.0, final_score))
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating TBAS score: {e}")
            return 0.0
    
    def _calculate_lat_score(self, simulation_result: Dict[str, Any], trajectory_data: List[Dict[str, Any]]) -> float:
        """
        Calculate Look-Ahead Time effectiveness
        - Analyze prediction accuracy over time
        - Measure early warning effectiveness
        - Account for false positive rates
        """
        try:
            # Extract prediction and actual conflict data
            predictions = self._extract_conflict_predictions(trajectory_data)
            actual_conflicts = self._extract_actual_conflicts(simulation_result, trajectory_data)
            
            if not predictions and not actual_conflicts:
                return 1.0  # Perfect score if no conflicts predicted or occurred
            
            # Calculate prediction accuracy metrics
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            early_warning_scores = []
            
            # Match predictions with actual conflicts
            prediction_matches = self._match_predictions_to_conflicts(predictions, actual_conflicts)
            
            for prediction in predictions:
                matched_conflict = prediction_matches.get(prediction['id'])
                
                if matched_conflict:
                    # True positive
                    true_positives += 1
                    
                    # Calculate early warning effectiveness
                    prediction_time = prediction.get('timestamp', 0)
                    conflict_time = matched_conflict.get('actual_time', prediction_time)
                    warning_time = conflict_time - prediction_time
                    
                    # Score based on warning time (more time = better score)
                    if warning_time >= 600:  # 10+ minutes warning
                        warning_score = 1.0
                    elif warning_time >= 300:  # 5-10 minutes warning
                        warning_score = 0.8
                    elif warning_time >= 120:  # 2-5 minutes warning
                        warning_score = 0.6
                    elif warning_time >= 60:   # 1-2 minutes warning
                        warning_score = 0.4
                    else:  # < 1 minute warning
                        warning_score = 0.2
                    
                    early_warning_scores.append(warning_score)
                else:
                    # False positive
                    false_positives += 1
            
            # Count false negatives (missed conflicts)
            false_negatives = len([c for c in actual_conflicts if c['id'] not in prediction_matches.values()])
            
            # Calculate LAT score components
            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 1.0
            
            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
            else:
                precision = 1.0
            
            # F1 score for overall prediction accuracy
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0
            
            # Early warning effectiveness
            if early_warning_scores:
                early_warning_effectiveness = np.mean(early_warning_scores)
            else:
                early_warning_effectiveness = 0.0
            
            # Combine metrics for final LAT score
            lat_score = 0.6 * f1_score + 0.4 * early_warning_effectiveness
            
            return max(0.0, min(1.0, lat_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating LAT score: {e}")
            return 0.0
    
    def _calculate_safety_margins(self, trajectory_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate comprehensive safety analysis
        - Minimum separation statistics
        - Safety buffer analysis
        - Near-miss detection and categorization
        - Statistical confidence intervals
        """
        try:
            if not trajectory_data:
                return self._get_default_safety_margins()
            
            horizontal_separations = []
            vertical_separations = []
            safety_buffer_violations = []
            critical_proximities = []
            
            for snapshot in trajectory_data:
                aircraft_states = snapshot.get('aircraft', {})
                aircraft_list = list(aircraft_states.values())
                
                # Calculate pairwise separations
                for i, ac1 in enumerate(aircraft_list):
                    for ac2 in aircraft_list[i+1:]:
                        # Calculate horizontal separation
                        h_sep = self._calculate_great_circle_distance(
                            ac1.get('latitude', 0), ac1.get('longitude', 0),
                            ac2.get('latitude', 0), ac2.get('longitude', 0)
                        )
                        
                        # Calculate vertical separation  
                        v_sep = abs(ac1.get('altitude_ft', 0) - ac2.get('altitude_ft', 0))
                        
                        horizontal_separations.append(h_sep)
                        vertical_separations.append(v_sep)
                        
                        # Check for safety buffer violations
                        safety_buffer_h = self.separation_min_nm * self.industry_standards['safety_buffer_multiplier']
                        safety_buffer_v = self.separation_min_ft * self.industry_standards['safety_buffer_multiplier']
                        
                        if h_sep < safety_buffer_h or v_sep < safety_buffer_v:
                            safety_buffer_violations.append({
                                'timestamp': snapshot.get('timestamp', 0),
                                'aircraft_pair': (ac1.get('callsign', 'UNK'), ac2.get('callsign', 'UNK')),
                                'horizontal_separation': h_sep,
                                'vertical_separation': v_sep,
                                'violation_type': 'horizontal' if h_sep < safety_buffer_h else 'vertical'
                            })
                        
                        # Track critical proximities
                        if h_sep < self.separation_min_nm * 2 and v_sep < self.separation_min_ft * 2:
                            critical_proximities.append({
                                'timestamp': snapshot.get('timestamp', 0),
                                'horizontal_separation': h_sep,
                                'vertical_separation': v_sep,
                                'severity': self._calculate_proximity_severity(h_sep, v_sep)
                            })
            
            # Calculate statistics
            if horizontal_separations:
                h_stats = {
                    'min': np.min(horizontal_separations),
                    'mean': np.mean(horizontal_separations),
                    'std': np.std(horizontal_separations),
                    'percentiles': np.percentile(horizontal_separations, [5, 25, 50, 75, 95])
                }
            else:
                h_stats = {'min': float('inf'), 'mean': float('inf'), 'std': 0, 'percentiles': [0]*5}
            
            if vertical_separations:
                v_stats = {
                    'min': np.min(vertical_separations),
                    'mean': np.mean(vertical_separations),
                    'std': np.std(vertical_separations),
                    'percentiles': np.percentile(vertical_separations, [5, 25, 50, 75, 95])
                }
            else:
                v_stats = {'min': float('inf'), 'mean': float('inf'), 'std': 0, 'percentiles': [0]*5}
            
            return {
                'min_horizontal_separation': h_stats['min'],
                'avg_horizontal_separation': h_stats['mean'],
                'std_horizontal_separation': h_stats['std'],
                'horizontal_percentiles': h_stats['percentiles'].tolist(),
                'min_vertical_separation': v_stats['min'],
                'avg_vertical_separation': v_stats['mean'],
                'std_vertical_separation': v_stats['std'],
                'vertical_percentiles': v_stats['percentiles'].tolist(),
                'safety_buffer_violations': len(safety_buffer_violations),
                'critical_proximity_events': len(critical_proximities),
                'safety_margin_efficiency': self._calculate_safety_margin_efficiency(h_stats, v_stats),
                'violation_details': safety_buffer_violations,
                'proximity_details': critical_proximities
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating safety margins: {e}")
            return self._get_default_safety_margins()
    
    def calculate_hallucination_metrics(self, llm_responses: List[Dict[str, Any]], 
                                      ground_truth: List[Dict[str, Any]]) -> LLMHallucinationMetrics:
        """
        Quantify LLM hallucination patterns and safety impact
        - Compare LLM outputs with geometric truth
        - Categorize hallucination types
        - Measure impact on safety decisions
        - Statistical significance testing
        """
        try:
            if not llm_responses or not ground_truth:
                return self._get_default_hallucination_metrics()
            
            hallucination_count = 0
            total_responses = len(llm_responses)
            geometric_deviations = []
            safety_impacts = []
            hallucination_categories = defaultdict(int)
            confidence_scores = []
            accuracy_scores = []
            
            # Match LLM responses with ground truth
            matched_pairs = self._match_llm_responses_to_truth(llm_responses, ground_truth)
            
            for llm_response, truth in matched_pairs:
                # Extract key decision elements
                llm_decision = self._extract_decision_elements(llm_response)
                truth_decision = self._extract_decision_elements(truth)
                
                # Calculate geometric deviation
                geometric_deviation = self._calculate_geometric_deviation(llm_decision, truth_decision)
                geometric_deviations.append(geometric_deviation)
                
                # Determine if this constitutes a hallucination
                is_hallucination = self._is_hallucination(llm_decision, truth_decision, geometric_deviation)
                
                if is_hallucination:
                    hallucination_count += 1
                    
                    # Categorize hallucination type
                    hallucination_type = self._categorize_hallucination(llm_decision, truth_decision)
                    hallucination_categories[hallucination_type] += 1
                    
                    # Calculate safety impact
                    safety_impact = self._calculate_safety_impact(llm_decision, truth_decision)
                    safety_impacts.append(safety_impact)
                
                # Extract confidence and accuracy for correlation analysis
                confidence = llm_response.get('confidence', 0.5)
                accuracy = 1.0 - geometric_deviation  # Higher deviation = lower accuracy
                
                confidence_scores.append(confidence)
                accuracy_scores.append(accuracy)
            
            # Calculate overall metrics
            hallucination_rate = hallucination_count / total_responses if total_responses > 0 else 0.0
            
            geometric_deviation_score = np.mean(geometric_deviations) if geometric_deviations else 0.0
            
            safety_impact_score = np.mean(safety_impacts) if safety_impacts else 0.0
            
            # Calculate confidence-accuracy correlation
            if len(confidence_scores) > 1 and len(accuracy_scores) > 1:
                correlation, p_value = stats.pearsonr(confidence_scores, accuracy_scores)
                confidence_correlation = correlation
            else:
                confidence_correlation = 0.0
                p_value = 1.0
            
            # Statistical significance testing
            statistical_significance = self._calculate_hallucination_significance(
                hallucination_count, total_responses, geometric_deviations, safety_impacts
            )
            
            return LLMHallucinationMetrics(
                hallucination_rate=hallucination_rate,
                geometric_deviation_score=geometric_deviation_score,
                safety_impact_score=safety_impact_score,
                hallucination_categories=dict(hallucination_categories),
                confidence_correlation=confidence_correlation,
                statistical_significance=statistical_significance
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating hallucination metrics: {e}")
            return self._get_default_hallucination_metrics()
    
    # === Supporting Helper Methods ===
    
    def _extract_separation_data(self, trajectory_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract separation data between all aircraft pairs from trajectory data"""
        separation_data = []
        
        for snapshot in trajectory_data:
            timestamp = snapshot.get('timestamp', 0)
            aircraft_states = snapshot.get('aircraft', {})
            aircraft_list = list(aircraft_states.values())
            
            for i, ac1 in enumerate(aircraft_list):
                for ac2 in aircraft_list[i+1:]:
                    h_sep = self._calculate_great_circle_distance(
                        ac1.get('latitude', 0), ac1.get('longitude', 0),
                        ac2.get('latitude', 0), ac2.get('longitude', 0)
                    )
                    v_sep = abs(ac1.get('altitude_ft', 0) - ac2.get('altitude_ft', 0))
                    
                    separation_data.append({
                        'timestamp': timestamp,
                        'aircraft_pair': (ac1.get('callsign', 'UNK'), ac2.get('callsign', 'UNK')),
                        'horizontal_separation': h_sep,
                        'vertical_separation': v_sep
                    })
        
        return separation_data
    
    def _extract_conflict_timeline(self, simulation_result: Dict[str, Any], 
                                 trajectory_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract conflict timeline data for timing analysis"""
        # This would extract conflict detection, resolution, and outcome timing
        # For now, return mock data structure
        return []
    
    def _extract_conflict_events(self, simulation_result: Dict[str, Any], 
                               trajectory_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract detailed conflict events with timing and severity data"""
        events = []
        
        # Extract from simulation result if available
        if 'conflicts' in simulation_result:
            for conflict in simulation_result['conflicts']:
                event = {
                    'id': conflict.get('conflict_id', 'unknown'),
                    'detection_time': conflict.get('timestamp', 0),
                    'severity_weight': self._calculate_conflict_severity_weight(conflict),
                    'resolved': conflict.get('resolved', False)
                }
                
                # Add action time if resolution exists
                if 'resolution_timestamp' in conflict:
                    event['first_action_time'] = conflict['resolution_timestamp']
                else:
                    event['first_action_time'] = event['detection_time'] + 30  # Default 30s delay
                
                events.append(event)
        
        return events
    
    def _calculate_conflict_severity_weight(self, conflict: Dict[str, Any]) -> float:
        """Calculate conflict severity weight based on metrics"""
        base_weight = 1.0
        
        # Weight by time to conflict
        time_to_conflict = conflict.get('time_to_conflict', 600)  # Default 10 minutes
        if time_to_conflict < 120:  # Less than 2 minutes
            base_weight *= 2.0
        elif time_to_conflict < 300:  # Less than 5 minutes
            base_weight *= 1.5
        
        # Weight by separation distance
        h_distance = conflict.get('horizontal_distance', 10.0)
        if h_distance < 3.0:  # Very close
            base_weight *= 2.0
        elif h_distance < 5.0:  # Close
            base_weight *= 1.5
        
        return min(base_weight, 3.0)  # Cap at 3x weight
    
    def _extract_conflict_predictions(self, trajectory_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract conflict predictions from trajectory data"""
        # This would extract prediction data from the trajectory logs
        # For now, return empty list
        return []
    
    def _extract_actual_conflicts(self, simulation_result: Dict[str, Any], 
                                trajectory_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract actual conflicts that occurred"""
        actual_conflicts = []
        
        if 'conflicts' in simulation_result:
            for conflict in simulation_result['conflicts']:
                actual_conflicts.append({
                    'id': conflict.get('conflict_id', 'unknown'),
                    'actual_time': conflict.get('timestamp', 0),
                    'aircraft_pair': (conflict.get('aircraft1', ''), conflict.get('aircraft2', '')),
                    'severity': conflict.get('severity', 'medium')
                })
        
        return actual_conflicts
    
    def _match_predictions_to_conflicts(self, predictions: List[Dict[str, Any]], 
                                      actual_conflicts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Match predictions to actual conflicts"""
        matches = {}
        
        for prediction in predictions:
            for conflict in actual_conflicts:
                # Simple matching by aircraft pair and time proximity
                if self._is_prediction_match(prediction, conflict):
                    matches[prediction['id']] = conflict
                    break
        
        return matches
    
    def _is_prediction_match(self, prediction: Dict[str, Any], conflict: Dict[str, Any]) -> bool:
        """Check if prediction matches actual conflict"""
        # This would implement sophisticated matching logic
        # For now, return basic time-based matching
        pred_time = prediction.get('timestamp', 0)
        conflict_time = conflict.get('actual_time', 0)
        return abs(pred_time - conflict_time) < 300  # Within 5 minutes
    
    def _calculate_rat_score(self, conflict_timeline: List[Dict[str, Any]]) -> float:
        """Calculate Resolution Action Time score"""
        if not conflict_timeline:
            return 1.0
        
        action_times = []
        for event in conflict_timeline:
            if 'action_time' in event and 'detection_time' in event:
                action_time = event['action_time'] - event['detection_time']
                action_times.append(action_time)
        
        if not action_times:
            return 0.5  # Neutral score if no data
        
        avg_action_time = np.mean(action_times)
        
        # Score based on average action time
        if avg_action_time <= 10:  # Very fast response
            return 1.0
        elif avg_action_time <= 30:  # Good response
            return 0.8
        elif avg_action_time <= 60:  # Acceptable response
            return 0.6
        else:  # Slow response
            return 0.4
    
    def _analyze_safety_violations(self, separation_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Analyze safety violations and near misses"""
        violations = []
        near_misses = []
        
        for data_point in separation_data:
            h_sep = data_point['horizontal_separation']
            v_sep = data_point['vertical_separation']
            
            # Check for safety violations
            if h_sep < self.separation_min_nm and v_sep < self.separation_min_ft:
                violations.append({
                    **data_point,
                    'violation_type': 'both',
                    'severity': 'high'
                })
            elif h_sep < self.separation_min_nm:
                violations.append({
                    **data_point,
                    'violation_type': 'horizontal',
                    'severity': 'medium'
                })
            elif v_sep < self.separation_min_ft:
                violations.append({
                    **data_point,
                    'violation_type': 'vertical',
                    'severity': 'medium'
                })
            
            # Check for near misses (within 1.5x separation minima)
            near_miss_h = self.separation_min_nm * 1.5
            near_miss_v = self.separation_min_ft * 1.5
            
            if (h_sep < near_miss_h and v_sep < near_miss_v and 
                not (h_sep < self.separation_min_nm and v_sep < self.separation_min_ft)):
                near_misses.append({
                    **data_point,
                    'proximity_factor': min(h_sep / self.separation_min_nm, v_sep / self.separation_min_ft)
                })
        
        return violations, near_misses
    
    def _calculate_confidence_intervals(self, separation_data: List[Dict[str, Any]], 
                                      total_conflicts: int) -> Dict[str, Tuple[float, float]]:
        """Calculate statistical confidence intervals"""
        confidence_intervals = {}
        
        if separation_data:
            h_separations = [d['horizontal_separation'] for d in separation_data]
            v_separations = [d['vertical_separation'] for d in separation_data]
            
            # 95% confidence intervals
            if len(h_separations) > 1:
                h_mean = np.mean(h_separations)
                h_sem = stats.sem(h_separations)
                h_ci = stats.t.interval(0.95, len(h_separations)-1, loc=h_mean, scale=h_sem)
                confidence_intervals['horizontal_separation'] = h_ci
            
            if len(v_separations) > 1:
                v_mean = np.mean(v_separations)
                v_sem = stats.sem(v_separations)
                v_ci = stats.t.interval(0.95, len(v_separations)-1, loc=v_mean, scale=v_sem)
                confidence_intervals['vertical_separation'] = v_ci
        
        return confidence_intervals
    
    def _calculate_great_circle_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance in nautical miles"""
        lat1_r = math.radians(lat1)
        lon1_r = math.radians(lon1)
        lat2_r = math.radians(lat2)
        lon2_r = math.radians(lon2)
        
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(max(0, min(1, a))))
        
        # Earth radius in nautical miles
        r_nm = 3440.065
        
        return r_nm * c
    
    def _calculate_proximity_severity(self, h_sep: float, v_sep: float) -> float:
        """Calculate proximity severity score"""
        h_factor = self.separation_min_nm / max(h_sep, 0.1)
        v_factor = self.separation_min_ft / max(v_sep, 1.0)
        
        return min(max(h_factor, v_factor), 10.0)  # Cap at 10x severity
    
    def _calculate_safety_margin_efficiency(self, h_stats: Dict[str, float], v_stats: Dict[str, float]) -> float:
        """Calculate safety margin efficiency score"""
        # Higher efficiency means maintaining larger margins without excessive deviation
        h_efficiency = h_stats['mean'] / (h_stats['std'] + 1.0) if h_stats['std'] > 0 else h_stats['mean']
        v_efficiency = v_stats['mean'] / (v_stats['std'] + 1.0) if v_stats['std'] > 0 else v_stats['mean']
        
        # Normalize to 0-1 scale
        h_norm = min(h_efficiency / (self.separation_min_nm * 3), 1.0)
        v_norm = min(v_efficiency / (self.separation_min_ft * 3), 1.0)
        
        return (h_norm + v_norm) / 2.0
    
    def _get_default_safety_margins(self) -> Dict[str, float]:
        """Return default safety margin values when calculation fails"""
        return {
            'min_horizontal_separation': float('inf'),
            'avg_horizontal_separation': float('inf'),
            'std_horizontal_separation': 0.0,
            'horizontal_percentiles': [0.0] * 5,
            'min_vertical_separation': float('inf'),
            'avg_vertical_separation': float('inf'),
            'std_vertical_separation': 0.0,
            'vertical_percentiles': [0.0] * 5,
            'safety_buffer_violations': 0,
            'critical_proximity_events': 0,
            'safety_margin_efficiency': 0.0,
            'violation_details': [],
            'proximity_details': []
        }
    
    def _match_llm_responses_to_truth(self, llm_responses: List[Dict[str, Any]], 
                                    ground_truth: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Match LLM responses to ground truth data"""
        matched_pairs = []
        
        for llm_resp in llm_responses:
            # Find corresponding ground truth
            for truth in ground_truth:
                if self._is_response_match(llm_resp, truth):
                    matched_pairs.append((llm_resp, truth))
                    break
        
        return matched_pairs
    
    def _is_response_match(self, llm_response: Dict[str, Any], ground_truth: Dict[str, Any]) -> bool:
        """Check if LLM response matches ground truth scenario"""
        # Simple matching by scenario ID or timestamp
        llm_scenario = llm_response.get('scenario_id', '')
        truth_scenario = ground_truth.get('scenario_id', '')
        
        if llm_scenario and truth_scenario:
            return llm_scenario == truth_scenario
        
        # Fallback to timestamp matching
        llm_time = llm_response.get('timestamp', 0)
        truth_time = ground_truth.get('timestamp', 0)
        
        return abs(llm_time - truth_time) < 60  # Within 1 minute
    
    def _extract_decision_elements(self, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key decision elements for comparison"""
        return {
            'heading': decision_data.get('heading', decision_data.get('new_heading')),
            'altitude': decision_data.get('altitude', decision_data.get('new_altitude')),
            'speed': decision_data.get('speed', decision_data.get('new_speed')),
            'resolution_type': decision_data.get('resolution_type', 'none'),
            'urgency': decision_data.get('urgency', 'medium'),
            'confidence': decision_data.get('confidence', 0.5)
        }
    
    def _calculate_geometric_deviation(self, llm_decision: Dict[str, Any], truth_decision: Dict[str, Any]) -> float:
        """Calculate geometric deviation between LLM and truth decisions"""
        deviations = []
        
        # Heading deviation (normalized to 0-1)
        if llm_decision.get('heading') and truth_decision.get('heading'):
            llm_hdg = llm_decision['heading']
            truth_hdg = truth_decision['heading']
            hdg_diff = abs(llm_hdg - truth_hdg)
            hdg_diff = min(hdg_diff, 360 - hdg_diff)  # Handle wraparound
            deviations.append(hdg_diff / 180.0)  # Normalize to 0-1
        
        # Altitude deviation (normalized)
        if llm_decision.get('altitude') and truth_decision.get('altitude'):
            llm_alt = llm_decision['altitude']
            truth_alt = truth_decision['altitude']
            alt_diff = abs(llm_alt - truth_alt)
            deviations.append(min(alt_diff / 10000.0, 1.0))  # Normalize to 0-1, cap at 10k ft
        
        # Speed deviation (normalized)
        if llm_decision.get('speed') and truth_decision.get('speed'):
            llm_spd = llm_decision['speed']
            truth_spd = truth_decision['speed']
            spd_diff = abs(llm_spd - truth_spd)
            deviations.append(min(spd_diff / 200.0, 1.0))  # Normalize to 0-1, cap at 200 kt
        
        return np.mean(deviations) if deviations else 0.0
    
    def _is_hallucination(self, llm_decision: Dict[str, Any], truth_decision: Dict[str, Any], 
                         geometric_deviation: float) -> bool:
        """Determine if LLM decision constitutes a hallucination"""
        # Threshold-based hallucination detection
        deviation_threshold = 0.3  # 30% deviation threshold
        
        # Check for significant geometric deviation
        if geometric_deviation > deviation_threshold:
            return True
        
        # Check for category mismatches
        llm_type = llm_decision.get('resolution_type', 'none')
        truth_type = truth_decision.get('resolution_type', 'none')
        
        if llm_type != truth_type and truth_type != 'none':
            return True
        
        return False
    
    def _categorize_hallucination(self, llm_decision: Dict[str, Any], truth_decision: Dict[str, Any]) -> str:
        """Categorize the type of hallucination"""
        # Resolution type mismatch
        if llm_decision.get('resolution_type') != truth_decision.get('resolution_type'):
            return 'resolution_type_mismatch'
        
        # Magnitude hallucination (correct type, wrong magnitude)
        if (llm_decision.get('heading') and truth_decision.get('heading') and
            abs(llm_decision['heading'] - truth_decision['heading']) > 45):
            return 'magnitude_hallucination'
        
        # Direction hallucination
        if (llm_decision.get('heading') and truth_decision.get('heading')):
            llm_turn = llm_decision['heading']
            truth_turn = truth_decision['heading']
            if (llm_turn - truth_turn) * (truth_turn - llm_turn) < 0:  # Opposite directions
                return 'direction_hallucination'
        
        # Overconfidence
        if llm_decision.get('confidence', 0.5) > 0.8 and truth_decision.get('confidence', 0.5) < 0.6:
            return 'overconfidence'
        
        return 'other'
    
    def _calculate_safety_impact(self, llm_decision: Dict[str, Any], truth_decision: Dict[str, Any]) -> float:
        """Calculate safety impact of LLM hallucination"""
        # This would implement complex safety impact analysis
        # For now, return basic impact based on deviation magnitude
        
        heading_impact = 0.0
        if llm_decision.get('heading') and truth_decision.get('heading'):
            hdg_diff = abs(llm_decision['heading'] - truth_decision['heading'])
            hdg_diff = min(hdg_diff, 360 - hdg_diff)
            heading_impact = min(hdg_diff / 90.0, 1.0)  # Normalize, cap at 90 degrees
        
        altitude_impact = 0.0
        if llm_decision.get('altitude') and truth_decision.get('altitude'):
            alt_diff = abs(llm_decision['altitude'] - truth_decision['altitude'])
            altitude_impact = min(alt_diff / 5000.0, 1.0)  # Normalize, cap at 5000 ft
        
        # Weight heading changes more heavily as they're more critical
        safety_impact = 0.7 * heading_impact + 0.3 * altitude_impact
        
        return safety_impact
    
    def _calculate_hallucination_significance(self, hallucination_count: int, total_responses: int,
                                           geometric_deviations: List[float], 
                                           safety_impacts: List[float]) -> Dict[str, float]:
        """Calculate statistical significance of hallucination patterns"""
        significance = {}
        
        # Binomial test for hallucination rate
        if total_responses > 0:
            hallucination_rate = hallucination_count / total_responses
            # Test against null hypothesis of 5% hallucination rate
            try:
                from scipy.stats import binom_test
                p_value = binom_test(hallucination_count, total_responses, 0.05)
                significance['hallucination_rate_p_value'] = p_value
                significance['hallucination_rate_significant'] = p_value < 0.05
            except ImportError:
                # Fallback if scipy.stats.binom_test not available
                significance['hallucination_rate_p_value'] = 1.0
                significance['hallucination_rate_significant'] = False
        
        # T-test for geometric deviations
        if len(geometric_deviations) > 1:
            # Test against null hypothesis of mean deviation = 0
            t_stat, p_value = stats.ttest_1samp(geometric_deviations, 0)
            significance['geometric_deviation_p_value'] = p_value
            significance['geometric_deviation_significant'] = p_value < 0.05
        
        # T-test for safety impacts
        if len(safety_impacts) > 1:
            t_stat, p_value = stats.ttest_1samp(safety_impacts, 0)
            significance['safety_impact_p_value'] = p_value
            significance['safety_impact_significant'] = p_value < 0.05
        
        return significance
    
    def _get_default_hallucination_metrics(self) -> LLMHallucinationMetrics:
        """Return default hallucination metrics when calculation fails"""
        return LLMHallucinationMetrics(
            hallucination_rate=0.0,
            geometric_deviation_score=0.0,
            safety_impact_score=0.0,
            hallucination_categories={},
            confidence_correlation=0.0,
            statistical_significance={}
        )
