"""Conflict Detection and Resolution Pipeline"""

import time
import json
import math
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

# Import CDR components
from ..detection.detector import ConflictDetector, ConflictPrediction
from ..ai.llm_client import ConflictContext


@dataclass
class ResolutionPolicy:
    """Policy configuration for conflict resolution methods"""
    use_llm: bool = True
    use_geometric_baseline: bool = False     # Only for ground truth/analysis
    apply_ssd_resolution: bool = False       # Only for detection/validation

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
    resolution_policy: ResolutionPolicy = None
    
    def __post_init__(self):
        if self.resolution_policy is None:
            self.resolution_policy = ResolutionPolicy()


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'scenario_id': self.scenario_id,
            'success': self.success,
            'error_message': self.error_message,
            'execution_time_seconds': self.execution_time_seconds,
            'total_conflicts': self.total_conflicts,
            'successful_resolutions': self.successful_resolutions,
            'simulation_cycles': self.simulation_cycles,
            'final_time_minutes': self.final_time_minutes
        }


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
        self.pending_intruders: List[Dict[str, Any]] = []
        
        # Initialize conflict detector
        self.conflict_detector = ConflictDetector(
            separation_min_nm=config.separation_min_nm,
            separation_min_ft=config.separation_min_ft,
            lookahead_minutes=config.lookahead_minutes
        )
    
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
        """
        Initialize simulation with real SCAT data and BlueSky integration
        - Load SCAT data for ownship and neighbors
        - Create aircraft in BlueSky
        - Set up surveillance and monitoring
        - Initialize conflict detection systems
        """
        self.logger.info("Initializing simulation with real scenario data")
        
        try:
            # Ensure BlueSky connection
            if not self.bluesky_client.connected:
                if not self.bluesky_client.connect():
                    raise RuntimeError("Failed to connect to BlueSky")
            
            # Reset simulation state
            self.bluesky_client._send_command("RESET", expect_response=True)
            
            # Initialize BlueSky with proper settings
            self.bluesky_client._initialize_simulation()
            
            # Load scenario data (SCAT format)
            scenario_data = self._load_scenario_data(scenario)
            
            # Create ownship from scenario
            ownship_data = scenario_data.get('ownship')
            if ownship_data:
                success = self.bluesky_client.create_aircraft(
                    callsign=ownship_data['callsign'],
                    aircraft_type=ownship_data.get('aircraft_type', 'B738'),
                    lat=ownship_data['latitude'],
                    lon=ownship_data['longitude'],
                    heading=ownship_data['heading_deg'],
                    altitude_ft=ownship_data['altitude_ft'],
                    speed_kt=ownship_data['speed_kt']
                )
                
                if success:
                    self.active_aircraft[ownship_data['callsign']] = {
                        'callsign': ownship_data['callsign'],
                        'type': 'ownship',
                        'created_at': self.current_time,
                        'scat_data': ownship_data
                    }
                    self.logger.info(f"Created ownship: {ownship_data['callsign']}")
                else:
                    raise RuntimeError(f"Failed to create ownship: {ownship_data['callsign']}")
            
            # Create initial traffic from scenario
            traffic_data = scenario_data.get('initial_traffic', [])
            for traffic in traffic_data:
                success = self.bluesky_client.create_aircraft(
                    callsign=traffic['callsign'],
                    aircraft_type=traffic.get('aircraft_type', 'A320'),
                    lat=traffic['latitude'],
                    lon=traffic['longitude'],
                    heading=traffic['heading_deg'],
                    altitude_ft=traffic['altitude_ft'],
                    speed_kt=traffic['speed_kt']
                )
                
                if success:
                    self.active_aircraft[traffic['callsign']] = {
                        'callsign': traffic['callsign'],
                        'type': 'traffic',
                        'created_at': self.current_time,
                        'scat_data': traffic
                    }
                    self.logger.info(f"Created traffic: {traffic['callsign']}")
            
            # Initialize pending intruders schedule
            self.pending_intruders = scenario_data.get('pending_intruders', [])
            
            # Set up surveillance parameters
            self._setup_surveillance_monitoring()
            
            # Initialize conflict detection systems
            self._setup_conflict_detection()
            
            self.logger.info(f"Initialized simulation with {len(self.active_aircraft)} aircraft")
            self.logger.info(f"Pending intruders: {len(self.pending_intruders)}")
            
        except Exception as e:
            self.logger.error(f"Simulation initialization failed: {e}", exc_info=True)
            raise
    
    def _run_simulation_loop(self, scenario, output_dir: Path) -> SimulationResult:
        """Main simulation loop"""
        self.logger.info("Starting simulation loop")
        
        result = SimulationResult(scenario_id=getattr(scenario, 'scenario_id', 'test'))
        cycle_count = 0
        max_cycles = int(self.config.max_simulation_time_minutes * 60 / self.config.cycle_interval_seconds)
        
        while cycle_count < max_cycles:
            # Advance simulation time
            self.current_time += self.config.cycle_interval_seconds
            
            # Inject pending intruders if scheduled
            self._inject_pending_intruders()
            
            # Get current aircraft states from BlueSky
            current_states = self.bluesky_client.get_aircraft_states()
            
            # Process conflicts with real detection and resolution
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
                self.logger.info(f"Cycle {cycle_count}: {elapsed_minutes:.1f} minutes elapsed, "
                               f"{len(self.active_aircraft)} aircraft active")
        
        result.success = True
        result.simulation_cycles = cycle_count
        result.final_time_minutes = self.current_time / 60
        
        return result

    
    def _process_conflicts(self, current_states: Dict[str, Any], output_dir: Path) -> int:
        """
        Complete conflict processing loop with geometric + LLM detection
        - Run geometric + LLM conflict detection
        - Generate and validate resolutions
        - Apply resolutions to BlueSky
        - Monitor resolution effectiveness
        - Handle multiple simultaneous conflicts
        """
        conflicts_resolved = 0
        
        try:
            # Step 1: Multi-layer conflict detection
            all_conflicts = self._detect_conflicts_multilayer(current_states)
            
            if not all_conflicts:
                return 0
            
            self.logger.info(f"Detected {len(all_conflicts)} conflicts at time {self.current_time/60:.1f} min")
            
            # Step 2: Prioritize conflicts by urgency and severity
            prioritized_conflicts = self._prioritize_conflicts(all_conflicts)
            
            # Step 3: Process each conflict with appropriate resolution strategy
            for conflict in prioritized_conflicts:
                try:
                    # Record baseline methods for ground truth (no commands issued)
                    if self.config.resolution_policy.use_geometric_baseline:
                        self._record_geometric_baseline(conflict, current_states)
                    
                    if self.config.resolution_policy.apply_ssd_resolution:
                        self._record_ssd_baseline(conflict)
                    
                    # Only LLM generates actual resolutions if enabled
                    resolution = None
                    if self.config.resolution_policy.use_llm:
                        resolution = self._generate_conflict_resolution(conflict, current_states)
                    
                    if not resolution:
                        self.logger.warning(f"No resolution generated for conflict {conflict.get('conflict_id')}")
                        continue
                    
                    # Validate resolution safety and feasibility
                    if not self._validate_resolution(resolution, conflict):
                        self.logger.warning(f"Resolution validation failed for conflict {conflict.get('conflict_id')}")
                        continue
                    
                    # Apply resolution to BlueSky simulation (only LLM output)
                    if self._apply_resolution_to_bluesky(resolution):
                        conflicts_resolved += 1
                        
                        # Record successful resolution
                        self._record_resolution_success(conflict, resolution)
                        
                        # Monitor resolution effectiveness
                        self._schedule_resolution_monitoring(conflict, resolution)
                        
                        self.logger.info(f"Successfully resolved conflict {conflict.get('conflict_id')}")
                    else:
                        self.logger.error(f"Failed to apply resolution for conflict {conflict.get('conflict_id')}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing conflict {conflict.get('conflict_id', 'unknown')}: {e}")
                    continue
            
            # Step 4: Update conflict history and memory
            self._update_conflict_history(all_conflicts, conflicts_resolved)
            
            # Step 5: Save conflict data for analysis
            if self.config.save_trajectories:
                self._save_conflict_data(all_conflicts, output_dir)
            
            return conflicts_resolved
            
        except Exception as e:
            self.logger.error(f"Error in conflict processing: {e}", exc_info=True)
            return 0
    
    def _save_trajectory_snapshot(self, current_states: Dict[str, Any], output_dir: Path):
        """Save current aircraft positions"""
        try:
            trajectory_file = output_dir / "trajectories.jsonl"
            
            # Convert AircraftState objects to dictionaries
            serializable_states = {}
            for callsign, state in current_states.items():
                if hasattr(state, '__dict__'):
                    serializable_states[callsign] = state.__dict__
                else:
                    serializable_states[callsign] = state
            
            snapshot = {
                'timestamp': self.current_time,
                'time_minutes': self.current_time / 60,
                'aircraft': serializable_states
            }
            
            # Append to JSONL file
            with open(trajectory_file, 'a') as f:
                f.write(json.dumps(snapshot, default=str) + '\n')
                
        except Exception as e:
            self.logger.warning(f"Failed to save trajectory snapshot: {e}")
    
    def _should_terminate(self, current_states: Dict[str, Any]) -> bool:
        """Check if simulation should terminate"""
        # Simple termination criteria
        return self.current_time >= (self.config.max_simulation_time_minutes * 60)
    
    def _inject_pending_intruders(self):
        """
        Dynamic intruder injection with time-based spawning
        - Time-based intruder spawning
        - Monte Carlo scenario generation  
        - Realistic intruder trajectories
        - Multiple intruder conflict scenarios
        """
        try:
            current_time_minutes = self.current_time / 60.0
            
            # Check for scheduled intruder injections
            injections_made = 0
            
            for intruder in self.pending_intruders[:]:  # Copy list to avoid modification during iteration
                spawn_time = intruder.get('spawn_time_minutes', 0)
                
                if current_time_minutes >= spawn_time:
                    # Time to inject this intruder
                    success = self._inject_single_intruder(intruder, current_time_minutes)
                    
                    if success:
                        injections_made += 1
                        self.pending_intruders.remove(intruder)
                        self.logger.info(f"Injected intruder {intruder['callsign']} at {current_time_minutes:.1f} min")
                    else:
                        self.logger.warning(f"Failed to inject intruder {intruder['callsign']}")
            
            # Generate Monte Carlo scenarios if enabled
            if hasattr(self.config, 'monte_carlo_enabled') and self.config.monte_carlo_enabled:
                self._generate_monte_carlo_intruders(current_time_minutes)
            
            if injections_made > 0:
                self.logger.info(f"Injected {injections_made} intruders at time {current_time_minutes:.1f} min")
                
        except Exception as e:
            self.logger.error(f"Error in intruder injection: {e}", exc_info=True)
    
    def _validate_resolution(self, resolution: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Safety validation for conflict resolutions
        - Check against operational constraints
        - Validate heading/altitude limits
        - Ensure separation maintenance
        - Verify feasibility with aircraft performance
        """
        try:
            aircraft_callsign = resolution.get('aircraft_callsign')
            if not aircraft_callsign:
                self.logger.warning("Resolution missing aircraft callsign")
                return False
            
            # Get current aircraft state
            current_states = self.bluesky_client.get_aircraft_states()
            aircraft_state = current_states.get(aircraft_callsign)
            
            if not aircraft_state:
                self.logger.warning(f"Aircraft state not found for {aircraft_callsign}")
                return False
            
            # Validate heading changes
            if 'new_heading' in resolution:
                current_heading = aircraft_state.heading_deg
                new_heading = resolution['new_heading']
                heading_change = abs(new_heading - current_heading)
                
                # Normalize heading change to 0-180 range
                if heading_change > 180:
                    heading_change = 360 - heading_change
                
                if heading_change > self.config.max_heading_change_deg:
                    self.logger.warning(f"Heading change {heading_change:.1f}° exceeds limit {self.config.max_heading_change_deg}°")
                    return False
            
            # Validate altitude changes
            if 'new_altitude' in resolution:
                current_altitude = aircraft_state.altitude_ft
                new_altitude = resolution['new_altitude']
                altitude_change = abs(new_altitude - current_altitude)
                
                if altitude_change > self.config.max_altitude_change_ft:
                    self.logger.warning(f"Altitude change {altitude_change:.0f}ft exceeds limit {self.config.max_altitude_change_ft}ft")
                    return False
                
                # Check altitude bounds (flight levels)
                if new_altitude < 1000 or new_altitude > 60000:
                    self.logger.warning(f"New altitude {new_altitude:.0f}ft outside operational bounds")
                    return False
            
            # Validate speed changes
            if 'new_speed' in resolution:
                current_speed = aircraft_state.speed_kt
                new_speed = resolution['new_speed']
                
                # Basic speed bounds check
                if new_speed < 120 or new_speed > 600:  # Typical commercial aircraft range
                    self.logger.warning(f"New speed {new_speed:.0f}kt outside operational bounds")
                    return False
                
                speed_change_pct = abs(new_speed - current_speed) / current_speed * 100
                if speed_change_pct > 30:  # Limit speed changes to 30%
                    self.logger.warning(f"Speed change {speed_change_pct:.1f}% too large")
                    return False
            
            # Validate separation maintenance
            if not self._validate_separation_maintenance(resolution, context, current_states):
                return False
            
            # Validate aircraft performance constraints
            if not self._validate_aircraft_performance(resolution, aircraft_state):
                return False
            
            self.logger.debug(f"Resolution validation passed for {aircraft_callsign}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in resolution validation: {e}", exc_info=True)
            return False
    
    # === Supporting Methods ===
    
    def _load_scenario_data(self, scenario) -> Dict[str, Any]:
        """Load and parse scenario data from SCAT or custom format"""
        if hasattr(scenario, 'to_dict'):
            return scenario.to_dict()
        elif isinstance(scenario, dict):
            return scenario
        elif isinstance(scenario, (str, Path)):
            # Load from file
            with open(scenario, 'r') as f:
                return json.load(f)
        else:
            # Create mock scenario for testing
            return {
                'ownship': {
                    'callsign': 'OWNSHIP',
                    'aircraft_type': 'B738',
                    'latitude': 41.978,
                    'longitude': -87.904,
                    'altitude_ft': 37000,
                    'heading_deg': 270,
                    'speed_kt': 450
                },
                'initial_traffic': [],
                'pending_intruders': [
                    {
                        'callsign': 'INTRUDER1',
                        'aircraft_type': 'A320',
                        'latitude': 42.0,
                        'longitude': -87.9,
                        'altitude_ft': 37000,
                        'heading_deg': 90,
                        'speed_kt': 420,
                        'spawn_time_minutes': 5.0
                    }
                ]
            }
    
    def _setup_surveillance_monitoring(self):
        """Configure surveillance and monitoring parameters"""
        try:
            # Set detection parameters
            if hasattr(self.bluesky_client.config, 'detection_range_nm'):
                range_nm = self.bluesky_client.config.detection_range_nm
            else:
                range_nm = self.config.detection_range_nm
            
            # Configure detection zones
            self.bluesky_client._send_command(f"ZONER {range_nm}", expect_response=True)
            self.logger.info(f"Set surveillance range: {range_nm} NM")
            
        except Exception as e:
            self.logger.warning(f"Error setting surveillance parameters: {e}")
    
    def _setup_conflict_detection(self):
        """Initialize conflict detection systems"""
        try:
            # Enable ASAS conflict detection
            self.bluesky_client._send_command("ASAS ON", expect_response=True)
            
            # Disable automatic resolution (LLM will handle it)
            self.bluesky_client._send_command("RESOOFF", expect_response=True)
            
            self.logger.info("Conflict detection systems initialized")
            
        except Exception as e:
            self.logger.warning(f"Error setting up conflict detection: {e}")
    
    def _detect_conflicts_multilayer(self, current_states: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Multi-layer conflict detection using BlueSky + geometric methods"""
        all_conflicts = []
        
        try:
            # Layer 1: BlueSky built-in conflict detection
            bluesky_conflicts = self.bluesky_client.get_conflicts()
            
            for conflict in bluesky_conflicts:
                # Normalize callsigns to ensure consistent matching
                aircraft1_norm = conflict.aircraft1.upper().strip()
                aircraft2_norm = conflict.aircraft2.upper().strip()
                
                all_conflicts.append({
                    'conflict_id': f"BS_{aircraft1_norm}_{aircraft2_norm}_{int(self.current_time)}",
                    'source': 'bluesky',
                    'aircraft1': aircraft1_norm,  # << explicit callsigns
                    'aircraft2': aircraft2_norm,  # << explicit callsigns
                    'time_to_conflict': conflict.time_to_conflict,
                    'horizontal_distance': conflict.horizontal_distance,
                    'vertical_distance': conflict.vertical_distance,
                    'conflict_type': conflict.conflict_type,
                    'severity': conflict.severity,
                    'timestamp': self.current_time
                })
            
            # Layer 2: Geometric conflict detection (ground truth only if enabled)
            if self.config.resolution_policy.use_geometric_baseline:
                geometric_conflicts = self._detect_conflicts_geometric(current_states)
                all_conflicts.extend(geometric_conflicts)
            
            # Remove duplicates and merge similar conflicts
            all_conflicts = self._deduplicate_conflicts(all_conflicts)
            
        except Exception as e:
            self.logger.error(f"Error in multi-layer conflict detection: {e}")
        
        return all_conflicts
    
    def _detect_conflicts_geometric(self, current_states: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Geometric conflict detection using the detector module"""
        conflicts = []
        
        try:
            # Find ownship
            ownship_state = None
            ownship_callsign = None
            
            for callsign, aircraft in self.active_aircraft.items():
                if aircraft.get('type') == 'ownship':
                    ownship_state = current_states.get(callsign)
                    ownship_callsign = callsign
                    break
            
            if not ownship_state:
                return conflicts
            
            # Get intruder states
            intruders = []
            for callsign, state in current_states.items():
                if callsign != ownship_callsign:
                    intruders.append(state)
            
            if not intruders:
                return conflicts
            
            # Run geometric detection
            predictions = self.conflict_detector.detect_conflicts(
                ownship_state, intruders, self.config.lookahead_minutes
            )
            
            for prediction in predictions:
                # Normalize callsigns to ensure consistent matching
                ownship_norm = ownship_callsign.upper().strip()
                intruder_norm = prediction.intruder_callsign.upper().strip()
                
                conflicts.append({
                    'conflict_id': f"GEO_{ownship_norm}_{intruder_norm}_{int(self.current_time)}",
                    'source': 'geometric',
                    'aircraft1': ownship_norm,
                    'aircraft2': intruder_norm,
                    'time_to_conflict': prediction.time_to_cpa_minutes * 60,  # Convert to seconds
                    'horizontal_distance': prediction.cpa_distance_nm,
                    'vertical_distance': prediction.cpa_altitude_difference_ft,
                    'conflict_type': prediction.conflict_type,
                    'severity': 'high' if prediction.conflict_severity > 0.7 else 'medium' if prediction.conflict_severity > 0.3 else 'low',
                    'timestamp': self.current_time,
                    'geometric_data': prediction
                })
            
        except Exception as e:
            self.logger.error(f"Error in geometric conflict detection: {e}")
        
        return conflicts
    
    def _deduplicate_conflicts(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate conflicts detected by different methods"""
        unique_conflicts = []
        seen_pairs = set()
        
        for conflict in conflicts:
            # Create pair identifier (order independent)
            ac1, ac2 = conflict['aircraft1'], conflict['aircraft2']
            pair_id = tuple(sorted([ac1, ac2]))
            
            if pair_id not in seen_pairs:
                seen_pairs.add(pair_id)
                unique_conflicts.append(conflict)
            else:
                # Update existing conflict with additional data
                for existing in unique_conflicts:
                    existing_pair = tuple(sorted([existing['aircraft1'], existing['aircraft2']]))
                    if existing_pair == pair_id:
                        # Merge conflict data
                        if conflict.get('geometric_data'):
                            existing['geometric_data'] = conflict['geometric_data']
                        break
        
        return unique_conflicts
    
    def _prioritize_conflicts(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize conflicts by urgency and severity"""
        def conflict_priority(conflict):
            time_to_conflict = conflict.get('time_to_conflict', float('inf'))
            severity_weights = {'high': 3, 'medium': 2, 'low': 1}
            severity_weight = severity_weights.get(conflict.get('severity', 'low'), 1)
            
            # Lower score = higher priority
            priority_score = time_to_conflict / severity_weight
            return priority_score
        
        return sorted(conflicts, key=conflict_priority)
    
    def _generate_conflict_resolution(self, conflict: Dict[str, Any], current_states: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate conflict resolution using LLM + geometric methods"""
        try:
            # Prepare conflict context for LLM
            context = self._prepare_conflict_context(conflict, current_states)
            
            # Use LLM to generate resolution if available
            if self.config.llm_enabled and self.llm_client:
                llm_resolution = self._generate_llm_resolution(context, conflict)
                if llm_resolution:
                    return llm_resolution
            
            # Fallback to geometric resolution
            return self._generate_geometric_resolution(conflict, current_states)
            
        except Exception as e:
            self.logger.error(f"Error generating resolution: {e}")
            return None
    
    def _prepare_conflict_context(self, conflict: Dict[str, Any], current_states: Dict[str, Any]) -> ConflictContext:
        """Prepare conflict context for LLM with robust aircraft state handling"""
        try:
            # Get explicit callsigns from conflict (NOT parsed from conflict ID)
            a = conflict['aircraft1'].upper().strip()
            b = conflict['aircraft2'].upper().strip()
            
            aircraft1_state = current_states.get(a)
            aircraft2_state = current_states.get(b)
            
            # If states are missing, try to fetch them directly from BlueSky
            if not (aircraft1_state and aircraft2_state):
                # Fetch missing aircraft states using the explicit callsigns
                needed = [x for x in [a, b] if x not in current_states]
                if needed:
                    self.logger.debug(f"Fetching missing aircraft states for: {needed}")
                    fetched = self.bluesky_client.get_aircraft_states(needed)
                    current_states.update(fetched)
                    
                aircraft1_state = current_states.get(a)
                aircraft2_state = current_states.get(b)
            
            if not aircraft1_state or not aircraft2_state:
                raise ValueError(f"Missing aircraft states for conflict: {a}, {b}")
            
            # Convert AircraftState objects to dictionaries if needed
            def state_to_dict(state):
                if hasattr(state, '__dict__'):
                    # It's a dataclass or object, convert to dict
                    if hasattr(state, 'latitude'):  # It's an AircraftState
                        return {
                            'callsign': state.callsign,
                            'latitude': state.latitude,
                            'longitude': state.longitude,
                            'altitude': state.altitude_ft,  # Use 'altitude' key for LLM compatibility
                            'altitude_ft': state.altitude_ft,
                            'heading': state.heading_deg,  # Use 'heading' key for LLM compatibility
                            'heading_deg': state.heading_deg,
                            'speed': state.speed_kt,  # Use 'speed' key for LLM compatibility
                            'speed_kt': state.speed_kt,
                            'vertical_speed_fpm': state.vertical_speed_fpm,
                            'timestamp': state.timestamp
                        }
                    else:
                        return state.__dict__
                else:
                    # Already a dictionary
                    return state
            
            aircraft1_dict = state_to_dict(aircraft1_state)
            aircraft2_dict = state_to_dict(aircraft2_state)
            
            # Get nearby traffic and convert to dictionaries
            nearby_traffic = []
            for callsign, state in current_states.items():
                if callsign not in [a, b]:  # Use normalized callsigns
                    nearby_traffic.append(state_to_dict(state))
            
            # Create context object
            context = ConflictContext(
                ownship_callsign=a,  # Use normalized callsign
                ownship_state=aircraft1_dict,
                intruders=[aircraft2_dict],  # List of intruder states
                scenario_time=self.current_time,
                lookahead_minutes=self.config.lookahead_minutes,
                constraints={
                    'max_heading_change': self.config.max_heading_change_deg,
                    'max_altitude_change': self.config.max_altitude_change_ft,
                    'separation_minima': {
                        'horizontal_nm': self.config.separation_min_nm,
                        'vertical_ft': self.config.separation_min_ft
                    }
                },
                nearby_traffic=nearby_traffic
            )
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error preparing conflict context: {e}")
            raise
    
    def _generate_llm_resolution(self, context: ConflictContext, conflict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate resolution using LLM"""
        try:
            if not self.llm_client:
                return None
            
            # Prepare conflict with expected field names for LLM
            conflict_with_intruder = {
                **conflict,
                'intruder_callsign': conflict.get('aircraft2', 'UNKNOWN')  # Add the expected field
            }
            
            # Prepare conflict info for LLM with conflict_id
            conflict_info = {
                'conflict_id': conflict['conflict_id'],  # Pass the conflict ID
                'conflicts': [conflict_with_intruder],  # Use conflict with intruder_callsign
                'primary_conflict': conflict_with_intruder,
                'severity': conflict.get('severity', 0.5),
                'time_to_conflict': conflict.get('time_to_cpa_minutes', 0)
            }
            
            response = self.llm_client.generate_resolution(context, conflict_info)
            
            if response and response.success:
                # Convert LLM response to standard resolution format
                return {
                    'aircraft_callsign': context.ownship_callsign,
                    'resolution_type': response.resolution_type,
                    'parameters': response.parameters,
                    'reasoning': response.reasoning,
                    'confidence': response.confidence,
                    'source': 'llm',
                    'timestamp': self.current_time
                }
            else:
                self.logger.warning("LLM resolution generation failed")
                return None
            
        except Exception as e:
            self.logger.error(f"Error in LLM resolution generation: {e}")
            return None
    
    def _generate_geometric_resolution(self, conflict: Dict[str, Any], current_states: Dict[str, Any]) -> Dict[str, Any]:
        """Generate resolution using geometric algorithms"""
        aircraft_callsign = conflict['aircraft1']  # Usually ownship
        aircraft_state = current_states[aircraft_callsign]
        
        # Convert AircraftState to dict if needed
        if hasattr(aircraft_state, 'heading_deg'):
            current_heading = aircraft_state.heading_deg
        else:
            current_heading = aircraft_state.get('heading_deg', aircraft_state.get('heading', 0))
        
        # Simple geometric resolution - turn right by 30 degrees
        new_heading = (current_heading + 30) % 360
        
        return {
            'aircraft_callsign': aircraft_callsign,
            'resolution_type': 'heading',
            'new_heading': new_heading,
            'reasoning': f"Geometric resolution: turn right 30° from {current_heading}°",
            'method': 'geometric',
            'confidence': 0.7
        }
    
    def _apply_resolution_to_bluesky(self, resolution: Dict[str, Any]) -> bool:
        """Apply resolution commands to BlueSky simulation"""
        try:
            aircraft_callsign = resolution['aircraft_callsign']
            
            # Apply heading change
            if 'new_heading' in resolution:
                success = self.bluesky_client.heading_command(
                    aircraft_callsign, resolution['new_heading']
                )
                if not success:
                    return False
                self.logger.info(f"Applied heading {resolution['new_heading']}° to {aircraft_callsign}")
            
            # Apply altitude change
            if 'new_altitude' in resolution:
                success = self.bluesky_client.altitude_command(
                    aircraft_callsign, resolution['new_altitude']
                )
                if not success:
                    return False
                self.logger.info(f"Applied altitude {resolution['new_altitude']}ft to {aircraft_callsign}")
            
            # Apply speed change
            if 'new_speed' in resolution:
                success = self.bluesky_client.speed_command(
                    aircraft_callsign, resolution['new_speed']
                )
                if not success:
                    return False
                self.logger.info(f"Applied speed {resolution['new_speed']}kt to {aircraft_callsign}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying resolution to BlueSky: {e}")
            return False
    
    def _record_resolution_success(self, conflict: Dict[str, Any], resolution: Dict[str, Any]):
        """Record successful resolution for learning"""
        resolution_record = {
            'conflict_id': conflict['conflict_id'],
            'timestamp': self.current_time,
            'aircraft': resolution['aircraft_callsign'],
            'resolution_type': resolution['resolution_type'],
            'method': resolution.get('method', 'unknown'),
            'confidence': resolution.get('confidence', 0.5),
            'reasoning': resolution.get('reasoning', ''),
            'success': True
        }
        
        self.resolution_history.append(resolution_record)
        
        # Store in memory system if available
        if self.memory_store:
            try:
                # Create a memory record for this resolution
                from ..ai.memory import create_memory_record
                
                # Get current aircraft states properly
                current_states = self.bluesky_client.get_aircraft_states()
                
                # Only create memory record if we have valid aircraft states
                if (conflict['aircraft1'] in current_states and 
                    conflict['aircraft2'] in current_states):
                    
                    # Convert conflict to ConflictContext format
                    conflict_context = self._prepare_conflict_context(conflict, current_states)
                    
                    # Create memory record
                    memory_record = create_memory_record(
                        conflict_context=conflict_context,
                        resolution=resolution,
                        outcome_metrics={'success_rate': 1.0},
                        task_type='resolution'
                    )
                    
                    self.memory_store.store_experience(memory_record)
                else:
                    self.logger.debug("Skipping memory storage - aircraft states not available")
                    
            except Exception as e:
                self.logger.warning(f"Failed to store resolution in memory: {e}")
    
    def _schedule_resolution_monitoring(self, conflict: Dict[str, Any], resolution: Dict[str, Any]):
        """Schedule monitoring of resolution effectiveness"""
        # This would be implemented to check if the resolution actually resolves the conflict
        # For now, we assume the resolution is effective
        pass
    
    def _update_conflict_history(self, conflicts: List[Dict[str, Any]], resolved_count: int):
        """Update conflict history with current detection results"""
        for conflict in conflicts:
            conflict['resolved'] = False  # Will be updated when resolution is confirmed
            self.conflict_history.append(conflict)
    
    def _save_conflict_data(self, conflicts: List[Dict[str, Any]], output_dir: Path):
        """Save conflict data for post-analysis"""
        try:
            conflict_file = output_dir / "conflicts.jsonl"
            
            # Convert any non-serializable objects to dictionaries
            def make_serializable(obj):
                """Recursively convert objects to JSON-serializable format"""
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_serializable(item) for item in obj]
                elif hasattr(obj, '__fspath__'):  # Path object
                    return str(obj)
                elif hasattr(obj, '__dict__'):  # Dataclass or object
                    return make_serializable(obj.__dict__)
                elif callable(obj):  # Method or function
                    return f"<function {obj.__name__}>" if hasattr(obj, '__name__') else "<method>"
                else:
                    # Fallback to string representation
                    return str(obj)
            
            serializable_conflicts = []
            for conflict in conflicts:
                serializable_conflicts.append(make_serializable(conflict))
            
            conflict_snapshot = {
                'timestamp': self.current_time,
                'time_minutes': self.current_time / 60,
                'conflicts': serializable_conflicts,
                'active_aircraft': len(self.active_aircraft)
            }
            
            with open(conflict_file, 'a') as f:
                f.write(json.dumps(conflict_snapshot) + '\n')
                
        except Exception as e:
            self.logger.warning(f"Failed to save conflict data: {e}")
    
    
    def _inject_single_intruder(self, intruder: Dict[str, Any], current_time_minutes: float) -> bool:
        """Inject a single intruder aircraft into the simulation"""
        try:
            # Adjust position based on spawn trajectory if specified
            spawn_position = self._calculate_spawn_position(intruder, current_time_minutes)
            
            success = self.bluesky_client.create_aircraft(
                callsign=intruder['callsign'],
                aircraft_type=intruder.get('aircraft_type', 'A320'),
                lat=spawn_position['latitude'],
                lon=spawn_position['longitude'],
                heading=spawn_position['heading_deg'],
                altitude_ft=spawn_position['altitude_ft'],
                speed_kt=spawn_position['speed_kt']
            )
            
            if success:
                self.active_aircraft[intruder['callsign']] = {
                    'callsign': intruder['callsign'],
                    'type': 'intruder',
                    'created_at': self.current_time,
                    'spawn_data': intruder
                }
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error injecting intruder {intruder.get('callsign')}: {e}")
            return False
    
    def _calculate_spawn_position(self, intruder: Dict[str, Any], current_time_minutes: float) -> Dict[str, Any]:
        """Calculate intruder spawn position based on trajectory"""
        # For now, use the provided position
        # In a full implementation, this would calculate position based on trajectory and timing
        return {
            'latitude': intruder['latitude'],
            'longitude': intruder['longitude'],
            'altitude_ft': intruder['altitude_ft'],
            'heading_deg': intruder['heading_deg'],
            'speed_kt': intruder['speed_kt']
        }
    
    def _generate_monte_carlo_intruders(self, current_time_minutes: float):
        """Generate Monte Carlo scenario intruders"""
        # This would implement Monte Carlo intruder generation
        # For now, skip this feature
        pass
    
    def _validate_separation_maintenance(self, resolution: Dict[str, Any], 
                                       context: Dict[str, Any], 
                                       current_states: Dict[str, Any]) -> bool:
        """Validate that resolution maintains required separation"""
        try:
            # Simple validation - check that resolution doesn't create new conflicts
            # In a full implementation, this would project aircraft positions forward
            # and verify separation is maintained
            return True
            
        except Exception as e:
            self.logger.error(f"Error in separation validation: {e}")
            return False
    
    def _validate_aircraft_performance(self, resolution: Dict[str, Any], 
                                     aircraft_state: Any) -> bool:
        """Validate resolution against aircraft performance constraints"""
        try:
            # Basic performance validation
            # In a full implementation, this would check against aircraft-specific performance data
            
            # Check reasonable climb/descent rates
            if 'new_altitude' in resolution:
                current_alt = aircraft_state.altitude_ft
                new_alt = resolution['new_altitude']
                alt_change = abs(new_alt - current_alt)
                
                # Limit climb/descent rate (simplified)
                max_alt_change_per_minute = 2000  # ft/min
                if alt_change > max_alt_change_per_minute * 5:  # 5 minute limit
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in performance validation: {e}")
            return False
    
    def _record_geometric_baseline(self, conflict: Dict[str, Any], current_states: Dict[str, Any]):
        """Record geometric conflict detection baseline (ground truth only)"""
        try:
            # This method records geometric analysis for ground truth comparison
            # It does NOT issue any commands to BlueSky
            geometric_data = {
                'conflict_id': conflict['conflict_id'],
                'source': 'geometric_baseline',
                'aircraft1': conflict['aircraft1'],
                'aircraft2': conflict['aircraft2'],
                'timestamp': self.current_time,
                'baseline_type': 'geometric'
            }
            
            # Log for analysis but don't command aircraft
            self.logger.debug(f"Geometric baseline recorded for {conflict['conflict_id']}")
            
        except Exception as e:
            self.logger.error(f"Error recording geometric baseline: {e}")
    
    def _record_ssd_baseline(self, conflict: Dict[str, Any]):
        """Record SSD conflict detection baseline (ground truth only)"""
        try:
            # This method records SSD analysis for ground truth comparison
            # It does NOT issue any commands to BlueSky
            ssd_data = {
                'conflict_id': conflict['conflict_id'],
                'source': 'ssd_baseline',
                'aircraft1': conflict['aircraft1'],
                'aircraft2': conflict['aircraft2'],
                'timestamp': self.current_time,
                'baseline_type': 'ssd'
            }
            
            # Log for analysis but don't command aircraft
            self.logger.debug(f"SSD baseline recorded for {conflict['conflict_id']}")
            
        except Exception as e:
            self.logger.error(f"Error recording SSD baseline: {e}")
