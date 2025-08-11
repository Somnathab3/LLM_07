"""Conflict Detection and Resolution Pipeline"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass
class PipelineConfig:
    """CDR Pipeline configuration"""
    cycle_interval_seconds: float = 60.0
    lookahead_minutes: float = 10.0
    max_simulation_time_minutes: float = 120.0
    separation_min_nm: float = 5.0
    separation_min_ft: float = 1000.0
    detection_range_nm: float = 100.0
    max_heading_change_deg: float = 45.0
    max_altitude_change_ft: float = 2000.0
    llm_enabled: bool = True
    memory_enabled: bool = True
    save_trajectories: bool = True


@dataclass
class SimulationResult:
    """Simulation result summary"""
    scenario_id: str
    success: bool = False
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0
    total_conflicts: int = 0
    successful_resolutions: int = 0
    simulation_cycles: int = 0
    final_time_minutes: float = 0.0


class CDRPipeline:
    """Conflict Detection and Resolution Pipeline"""
    
    def __init__(self, config: PipelineConfig, 
                 bluesky_client,
                 llm_client=None,
                 memory_store=None):
        self.config = config
        self.bluesky_client = bluesky_client
        self.llm_client = llm_client
        self.memory_store = memory_store
        self.logger = logging.getLogger(__name__)
        
        # Runtime state
        self.current_time: float = 0.0
        self.active_aircraft: Dict[str, Dict[str, Any]] = {}
        self.conflict_history: List[Dict[str, Any]] = []
        self.resolution_history: List[Dict[str, Any]] = []
    
    def run_scenario(self, scenario, output_dir: Path) -> SimulationResult:
        """Run complete CDR scenario"""
        self.logger.info(f"Starting CDR scenario: {getattr(scenario, 'scenario_id', 'unknown')}")
        
        start_time = time.time()
        self.current_time = 0.0
        self.active_aircraft.clear()
        self.conflict_history.clear()
        self.resolution_history.clear()
        
        try:
            # Initialize simulation
            self._initialize_simulation(scenario)
            
            # Main simulation loop
            result = self._run_simulation_loop(scenario, output_dir)
            
            # Calculate final metrics
            result.execution_time_seconds = time.time() - start_time
            result.total_conflicts = len(self.conflict_history)
            result.successful_resolutions = sum(
                1 for r in self.resolution_history if r.get('success', False)
            )
            
            self.logger.info(f"Scenario completed in {result.execution_time_seconds:.2f}s")
            self.logger.info(f"Conflicts: {result.total_conflicts}, Resolved: {result.successful_resolutions}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Scenario failed: {e}", exc_info=True)
            return SimulationResult(
                scenario_id=getattr(scenario, 'scenario_id', 'unknown'),
                success=False,
                error_message=str(e),
                execution_time_seconds=time.time() - start_time
            )
    
    def _initialize_simulation(self, scenario):
        """Initialize simulation with scenario data"""
        self.logger.info("Initializing simulation")
        
        # Mock scenario initialization
        # In real implementation, would create aircraft from scenario data
        self.active_aircraft["OWNSHIP"] = {
            'callsign': 'OWNSHIP',
            'type': 'ownship',
            'created_at': 0.0
        }
        
        self.logger.info(f"Initialized simulation with {len(self.active_aircraft)} aircraft")
    
    def _run_simulation_loop(self, scenario, output_dir: Path) -> SimulationResult:
        """Main simulation loop"""
        self.logger.info("Starting simulation loop")
        
        result = SimulationResult(scenario_id=getattr(scenario, 'scenario_id', 'test'))
        cycle_count = 0
        max_cycles = int(self.config.max_simulation_time_minutes * 60 / self.config.cycle_interval_seconds)
        
        while cycle_count < max_cycles:
            # Advance simulation time
            self.current_time += self.config.cycle_interval_seconds
            
            # Mock aircraft states
            current_states = self._get_mock_aircraft_states()
            
            # Process conflicts (simplified)
            conflicts_resolved = self._process_conflicts(current_states, output_dir)
            
            # Save trajectory data if enabled
            if self.config.save_trajectories:
                self._save_trajectory_snapshot(current_states, output_dir)
            
            # Check termination conditions
            if self._should_terminate(current_states):
                self.logger.info("Simulation terminated - completion criteria met")
                break
            
            cycle_count += 1
            
            # Log progress every 10 cycles
            if cycle_count % 10 == 0:
                elapsed_minutes = self.current_time / 60
                self.logger.info(f"Cycle {cycle_count}: {elapsed_minutes:.1f} minutes elapsed")
        
        result.success = True
        result.simulation_cycles = cycle_count
        result.final_time_minutes = self.current_time / 60
        
        return result
    
    def _get_mock_aircraft_states(self) -> Dict[str, Any]:
        """Get mock aircraft states for demonstration"""
        return {
            "OWNSHIP": {
                'callsign': 'OWNSHIP',
                'latitude': 41.978,
                'longitude': -87.904,
                'altitude_ft': 37000,
                'heading_deg': 270,
                'speed_kt': 450,
                'timestamp': self.current_time
            }
        }
    
    def _process_conflicts(self, current_states: Dict[str, Any], output_dir: Path) -> int:
        """Process conflict detection and resolution"""
        conflicts_resolved = 0
        
        # Mock conflict processing
        if self.current_time > 300:  # After 5 minutes, simulate a conflict
            if not any(c.get('resolved', False) for c in self.conflict_history):
                # Create mock conflict
                conflict = {
                    'conflict_id': f"CONFLICT_{int(self.current_time)}",
                    'timestamp': self.current_time,
                    'ownship': 'OWNSHIP',
                    'intruder': 'TRAFFIC1',
                    'resolved': True
                }
                
                self.conflict_history.append(conflict)
                
                # Mock resolution
                resolution = {
                    'conflict_id': conflict['conflict_id'],
                    'timestamp': self.current_time,
                    'resolution_type': 'heading_change',
                    'success': True,
                    'method': 'llm' if self.config.llm_enabled else 'deterministic'
                }
                
                self.resolution_history.append(resolution)
                conflicts_resolved = 1
                
                self.logger.info(f"Mock conflict resolved at time {self.current_time/60:.1f} min")
        
        return conflicts_resolved
    
    def _save_trajectory_snapshot(self, current_states: Dict[str, Any], output_dir: Path):
        """Save current aircraft positions"""
        trajectory_file = output_dir / "trajectories.jsonl"
        
        snapshot = {
            'timestamp': self.current_time,
            'time_minutes': self.current_time / 60,
            'aircraft': current_states
        }
        
        # Append to JSONL file
        with open(trajectory_file, 'a') as f:
            import json
            f.write(json.dumps(snapshot) + '\n')
    
    def _should_terminate(self, current_states: Dict[str, Any]) -> bool:
        """Check if simulation should terminate"""
        # Simple termination criteria
        return self.current_time >= (self.config.max_simulation_time_minutes * 60)
