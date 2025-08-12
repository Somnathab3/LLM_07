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
    cycle_interval_seconds: float = 300.0  # Changed from 60.0 to 300.0 (5 minutes)
    lookahead_minutes: float = 5.0  # Match BlueSky native asas_dtlookahead=300s
    max_simulation_time_minutes: float = 120.0
    separation_min_nm: float = 5.0  # Match BlueSky native asas_pzr=5.0
    separation_min_ft: float = 1000.0  # Match BlueSky native asas_pzh=1000.0
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
        self.aircraft_states: Dict[str, Dict[str, Any]] = {}  # Track aircraft states for completion check
        self.conflict_history: List[Dict[str, Any]] = []
        self.resolution_history: List[Dict[str, Any]] = []
        self.pending_intruders: List[Dict[str, Any]] = []
        
        # Route management for reroute_via resolutions
        self.resume_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Initialize conflict detector
        self.conflict_detector = ConflictDetector(
            separation_min_nm=config.separation_min_nm,
            separation_min_ft=config.separation_min_ft,
            lookahead_minutes=config.lookahead_minutes
        )
        
        # Check for direct bridge availability
        if hasattr(bluesky_client, 'use_direct_bridge') and bluesky_client.use_direct_bridge:
            self.logger.info("✅ Using enhanced BlueSky direct bridge for improved communication")
        else:
            self.logger.warning("⚠️ Using fallback BlueSky communication mode")
    
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
                    # Arm the aircraft with proper autopilot settings for movement
                    callsign = ownship_data['callsign']
                    self.bluesky_client.heading_command(callsign, ownship_data['heading_deg'])
                    self.bluesky_client.altitude_command(callsign, ownship_data['altitude_ft'])
                    self.bluesky_client.set_speed(callsign, ownship_data['speed_kt'])
                    
                    self.active_aircraft[ownship_data['callsign']] = {
                        'callsign': ownship_data['callsign'],
                        'type': 'ownship',
                        'created_at': self.current_time,
                        'scat_data': ownship_data
                    }
                    self.logger.info(f"Created ownship: {ownship_data['callsign']}")
                    
                    # Verify aircraft state immediately after creation
                    verification_state = self.bluesky_client.get_aircraft_state(ownship_data['callsign'])
                    if verification_state:
                        self.logger.info(f"✅ Ownship verified: {verification_state.callsign} at {verification_state.latitude:.4f},{verification_state.longitude:.4f}")
                    else:
                        self.logger.warning(f"⚠️ Could not verify ownship state after creation")
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
                    # Arm the traffic aircraft with proper autopilot settings for movement
                    callsign = traffic['callsign']
                    self.bluesky_client.heading_command(callsign, traffic['heading_deg'])
                    self.bluesky_client.altitude_command(callsign, traffic['altitude_ft'])
                    self.bluesky_client.set_speed(callsign, traffic['speed_kt'])
                    
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
            
            # Start the simulation (critical for aircraft movement)
            if not self.bluesky_client.op():
                self.logger.warning("Failed to start simulation with OP command")
            else:
                self.logger.info("✅ Simulation started - aircraft should now be moving")

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
            # Advance simulation time both locally and in BlueSky
            self.current_time += self.config.cycle_interval_seconds
            
            # Use direct simulation stepping for reliable kinematics
            if cycle_count > 0:  # Skip first cycle to avoid double advancement
                step_success = self.bluesky_client.step_minutes(self.config.cycle_interval_seconds / 60.0)
                if not step_success:
                    self.logger.warning(f"Failed to advance BlueSky simulation time by {self.config.cycle_interval_seconds}s")
            
            # Inject pending intruders if scheduled
            self._inject_pending_intruders()
            
            # Get current aircraft states from BlueSky
            current_states = self.bluesky_client.get_aircraft_states()
            
            # Debug logging to verify aircraft motion and persistence
            if current_states:
                for callsign, state in current_states.items():
                    self.logger.debug(f"POS {callsign}: lat={state.latitude:.4f}, lon={state.longitude:.4f}, GS={state.speed_kt:.1f} kt")
            else:
                self.logger.warning(f"⚠️ No aircraft states returned at cycle {cycle_count}, time {self.current_time/60:.1f} min")
                
                # Try to diagnose missing aircraft issue
                if cycle_count > 1:  # Give some time for aircraft to be created
                    self.logger.warning("Attempting to diagnose missing aircraft...")
                    
                    # Check if aircraft are still tracked in active_aircraft
                    self.logger.warning(f"Active aircraft tracked: {list(self.active_aircraft.keys())}")
                    
                    # Try to get states for specific known aircraft
                    for known_callsign in self.active_aircraft.keys():
                        single_state = self.bluesky_client.get_aircraft_state(known_callsign)
                        if single_state:
                            self.logger.warning(f"Found {known_callsign} via single query: {single_state.latitude:.4f},{single_state.longitude:.4f}")
                            current_states[known_callsign] = single_state
                        else:
                            self.logger.warning(f"Could not retrieve state for known aircraft: {known_callsign}")
            
            # Update aircraft states for destination checking
            self.aircraft_states.update(current_states)
            
            # Process conflicts with real detection and resolution
            conflicts_resolved = self._process_conflicts(current_states, output_dir)
            
            # Save trajectory data if enabled
            if self.config.save_trajectories:
                self._save_trajectory_snapshot(current_states, output_dir)
            
            # Check if test should be completed (aircraft reached destination)
            if self._complete_test_if_destination_reached():
                self.logger.info("✅ Test completed: Aircraft reached destination")
                break
            
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
            # CRITICAL FIX: Pause simulation BEFORE conflict detection to ensure stable aircraft states
            # This prevents stale state issues when querying aircraft positions for conflict detection and LLM input
            print("⏸️  HOLDING simulation for conflict detection and LLM processing...")
            pause_success = self.bluesky_client.pause_simulation()
            if not pause_success:
                print("⚠️  Failed to pause simulation, proceeding anyway")
            
            # Step 1: Get stable aircraft states while simulation is paused
            stable_states = self.bluesky_client.get_aircraft_states()
            
            # Step 2: Multi-layer conflict detection using stable states
            all_conflicts = self._detect_conflicts_multilayer(stable_states)
            
            if not all_conflicts:
                # Resume simulation if no conflicts found
                print("▶️  Resuming simulation - no conflicts detected...")
                resume_success = self.bluesky_client.resume_simulation()
                if not resume_success:
                    print("⚠️  Failed to resume simulation")
                return 0
            
            self.logger.info(f"Detected {len(all_conflicts)} conflicts at time {self.current_time/60:.1f} min")
            
            # Step 3: Prioritize conflicts by urgency and severity
            prioritized_conflicts = self._prioritize_conflicts(all_conflicts)
            
            # Step 4: Process each conflict with appropriate resolution strategy
            for conflict in prioritized_conflicts:
                try:
                    # Record baseline methods for ground truth (no commands issued)
                    if self.config.resolution_policy.use_geometric_baseline:
                        self._record_geometric_baseline(conflict, stable_states)
                    
                    if self.config.resolution_policy.apply_ssd_resolution:
                        self._record_ssd_baseline(conflict)
                    
                    # Only LLM generates actual resolutions if enabled
                    resolution = None
                    if self.config.resolution_policy.use_llm:
                        # Use stable aircraft states for accurate LLM input
                        resolution = self._generate_conflict_resolution(conflict, stable_states)
                    
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
                        
                        # Mark conflict as resolved
                        conflict['resolved'] = True
                        conflict['resolution_applied_at'] = self.current_time
                        
                        # Record successful resolution
                        self._record_resolution_success(conflict, resolution)
                        
                        # Save resolution data to file
                        resolution_with_metadata = resolution.copy()
                        resolution_with_metadata.update({
                            'conflict_id': conflict['conflict_id'],
                            'success': True,
                            'applied_at': self.current_time
                        })
                        self._save_resolution_data(resolution_with_metadata, output_dir)
                        
                        # Monitor resolution effectiveness
                        self._schedule_resolution_monitoring(conflict, resolution)
                        
                        self.logger.info(f"Successfully resolved conflict {conflict.get('conflict_id')}")
                    else:
                        # Save failed resolution data
                        resolution_with_metadata = resolution.copy()
                        resolution_with_metadata.update({
                            'conflict_id': conflict['conflict_id'],
                            'success': False,
                            'applied_at': self.current_time,
                            'failure_reason': 'BlueSky application failed'
                        })
                        self._save_resolution_data(resolution_with_metadata, output_dir)
                        
                        self.logger.error(f"Failed to apply resolution for conflict {conflict.get('conflict_id')}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing conflict {conflict.get('conflict_id', 'unknown')}: {e}")
                    continue
            
            # CRITICAL FIX: Resume simulation after all resolutions are applied
            # This allows BlueSky to process the commands and update aircraft states
            print("▶️  Resuming simulation after LLM resolutions...")
            
            # Check for aircraft that should resume to destination
            for callsign in list(self.resume_tasks.keys()):
                self._maybe_resume_to_destination(callsign, stable_states)
            
            resume_success = self.bluesky_client.resume_simulation()
            if not resume_success:
                print("⚠️  Failed to resume simulation")
            
            # Allow multiple simulation cycles for BlueSky to process the heading commands
            # BlueSky needs sufficient time to update aircraft states after OP
            import time
            time.sleep(1.0)  # 1000ms = several simulation cycles at 8x speed to ensure state propagation
            
            # Step 5: Update conflict history and memory
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
                # Support both spawn_time_minutes and injection_time_minutes
                spawn_time = intruder.get('spawn_time_minutes', intruder.get('injection_time_minutes', 0))
                
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
        elif hasattr(scenario, 'to_simple_scenario'):
            # It's a SCAT adapter - convert to simple scenario format
            return scenario.to_simple_scenario()
        elif isinstance(scenario, dict):
            # Check if this is raw SCAT data with scenario_config
            if 'scenario_config' in scenario:
                config = scenario['scenario_config']
                return {
                    'ownship': config.get('ownship', {}),
                    'initial_traffic': [],
                    'pending_intruders': config.get('intruders', [])
                }
            elif 'plots' in scenario:
                # This is raw SCAT data - try to extract basic info
                # Find callsign from flight plan data
                callsign = 'SCAT_AC'
                aircraft_type = 'B738'
                
                if 'fpl' in scenario and 'fpl_base' in scenario['fpl']:
                    for base in scenario['fpl']['fpl_base']:
                        if 'callsign' in base:
                            callsign = base['callsign']
                        if 'aircraft_type' in base:
                            aircraft_type = base['aircraft_type']
                        break
                
                # Find first plot with position data
                first_plot = None
                if 'plots' in scenario and scenario['plots']:
                    for plot in scenario['plots']:
                        if 'I062/105' in plot:
                            pos_data = plot['I062/105']
                            if 'latitude' in pos_data and 'longitude' in pos_data:
                                first_plot = plot
                                break
                
                if not first_plot:
                    # Create a default scenario if no valid plot found
                    return {
                        'ownship': {
                            'callsign': callsign,
                            'aircraft_type': aircraft_type,
                            'latitude': 40.0,
                            'longitude': -80.0,
                            'altitude_ft': 37000,
                            'heading_deg': 90,
                            'speed_kt': 450
                        },
                        'initial_traffic': [],
                        'pending_intruders': []
                    }
                
                # Extract position and flight data from first plot
                pos_data = first_plot['I062/105']
                lat = pos_data.get('latitude', pos_data.get('lat', 40.0))
                lon = pos_data.get('longitude', pos_data.get('lon', -80.0))
                
                # Extract altitude from flight level
                altitude_ft = 37000  # Default
                if 'I062/136' in first_plot:
                    fl = first_plot['I062/136'].get('measured_flight_level', 370)
                    altitude_ft = fl * 100
                
                # Extract heading and speed from I062/380
                heading_deg = 90  # Default
                speed_kt = 450    # Default
                if 'I062/380' in first_plot:
                    i380 = first_plot['I062/380']
                    if 'magnetic_heading' in i380:
                        heading_deg = i380['magnetic_heading']
                    if 'indicated_airspeed' in i380:
                        speed_kt = i380['indicated_airspeed']
                
                return {
                    'ownship': {
                        'callsign': callsign,
                        'aircraft_type': aircraft_type,
                        'latitude': lat,
                        'longitude': lon,
                        'altitude_ft': altitude_ft,
                        'heading_deg': heading_deg,
                        'speed_kt': speed_kt
                    },
                    'initial_traffic': [],
                    'pending_intruders': [
                        # Add test intruder for conflict testing - place closer to ownship for guaranteed conflict
                        {
                            'callsign': 'TEST_INTRUDER',
                            'aircraft_type': 'A320',
                            'latitude': lat + 0.01,  # Much closer - only 0.01 degrees (~0.6 NM)
                            'longitude': lon + 0.01,
                            'altitude_ft': altitude_ft,
                            'heading_deg': (heading_deg + 180) % 360,  # Head-on approach
                            'speed_kt': 420,
                            'injection_time_minutes': 1.0  # Inject after 1 minute instead of 5
                        }
                    ]
                }
            else:
                return scenario
        elif isinstance(scenario, (str, Path)):
            # Load from file
            with open(scenario, 'r') as f:
                data = json.load(f)
                return self._load_scenario_data(data)  # Recursive call to handle loaded data
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
            
            # Set more sensitive conflict detection parameters
            # Configure BlueSky separation minima to match our pipeline config
            sep_nm = self.config.separation_min_nm
            sep_ft = self.config.separation_min_ft
            
            # Set horizontal separation (in nautical miles)
            self.bluesky_client._send_command(f"DTMULT 1.0", expect_response=True)  # Real-time
            self.bluesky_client._send_command(f"DTNOLOOK 1.0", expect_response=True)  # 1 second intervals
            
            # Configure ASAS parameters for more sensitive detection
            self.bluesky_client._send_command(f"RFACH 1.0", expect_response=True)  # No extra horizontal margin
            self.bluesky_client._send_command(f"RFACV 1.0", expect_response=True)  # No extra vertical margin
            
            # Disable automatic resolution (LLM will handle it)
            self.bluesky_client._send_command("RESOOFF", expect_response=True)
            
            self.logger.info(f"Conflict detection systems initialized with sep={sep_nm}NM/{sep_ft}ft")
            
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
                nearby_traffic=nearby_traffic,
                destination=self._get_aircraft_destination(a)  # Add destination information
            )
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error preparing conflict context: {e}")
            raise
    
    def _get_aircraft_destination(self, callsign: str) -> Optional[Dict[str, Any]]:
        """Get destination information for aircraft"""
        try:
            # In a real implementation, this would query the aircraft's flight plan
            # For now, we'll create a placeholder destination
            # You could extend this to:
            # 1. Parse flight plan from BlueSky
            # 2. Look up destination from scenario file
            # 3. Use predefined waypoints
            
            # Placeholder implementation - in production, would extract from flight plan
            # Using Chicago O'Hare as a more realistic destination for testing
            return {
                "name": "DST",
                "lat": 41.9742,  # Chicago O'Hare coordinates
                "lon": -87.9073
            }
            
        except Exception as e:
            self.logger.warning(f"Could not get destination for {callsign}: {e}")
            return None

    def _check_destination_reached(self, callsign: str, threshold_nm: float = 5.0) -> bool:
        """Check if aircraft has reached its destination"""
        try:
            # Get current aircraft position
            aircraft_state = self.bluesky_client.get_aircraft_state(callsign)
            if not aircraft_state:
                return False
            
            # Get destination
            destination = self._get_aircraft_destination(callsign)
            if not destination:
                return False
            
            # Calculate distance to destination
            import math
            current_lat = aircraft_state.latitude if hasattr(aircraft_state, 'latitude') else 0
            current_lon = aircraft_state.longitude if hasattr(aircraft_state, 'longitude') else 0
            dest_lat = destination['lat']
            dest_lon = destination['lon']
            
            # Simple distance calculation in nautical miles
            dlat = dest_lat - current_lat
            dlon = dest_lon - current_lon
            distance_nm = math.sqrt(dlat**2 + dlon**2) * 60  # Rough conversion
            
            if distance_nm <= threshold_nm:
                self.logger.info(f"🎯 {callsign} has reached destination {destination['name']} "
                               f"(distance: {distance_nm:.2f} NM)")
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Could not check destination for {callsign}: {e}")
            return False

    def _complete_test_if_destination_reached(self) -> bool:
        """Check if test should be completed because aircraft reached destination"""
        try:
            # Check each tracked aircraft
            for callsign in self.aircraft_states:
                if self._check_destination_reached(callsign):
                    self.logger.info(f"✅ Test completed: {callsign} reached destination")
                    return True
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking test completion: {e}")
            return False
    
    def _generate_llm_resolution(self, context: ConflictContext, conflict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate resolution using LLM"""
        try:
            if not self.llm_client:
                return None

            # Check if combined mode is enabled
            if hasattr(self.llm_client.config, 'enable_combined_mode') and self.llm_client.config.enable_combined_mode:
                return self._generate_combined_llm_resolution(context, conflict)

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

    def _generate_combined_llm_resolution(self, context: ConflictContext, conflict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate resolution using combined LLM method (single call for detection + resolution)"""
        try:
            if not self.llm_client:
                return None
            
            # Use the combined detect_and_resolve method
            conflict_id = conflict['conflict_id']
            combined_response = self.llm_client.detect_and_resolve(context, conflict_id)
            
            if not combined_response:
                self.logger.warning(f"Combined LLM response failed for conflict {conflict_id}")
                return None
            
            # Check if conflicts were detected
            if not combined_response.get('conflicts_detected', False):
                self.logger.debug(f"No conflicts detected by combined LLM for {conflict_id}")
                return {
                    'aircraft_callsign': context.ownship_callsign,
                    'resolution_type': 'no_action',
                    'parameters': {},
                    'reasoning': 'No conflicts detected by LLM',
                    'confidence': combined_response.get('resolution', {}).get('confidence', 0.5),
                    'source': 'llm_combined',
                    'timestamp': self.current_time
                }
            
            # Extract resolution from combined response
            resolution = combined_response.get('resolution', {})
            if not resolution:
                self.logger.warning(f"No resolution in combined response for {conflict_id}")
                return None
            
            # Convert LLM parameter format to BlueSky format
            parameters = resolution.get('parameters', {})
            converted_params = {}
            
            # Map LLM parameter names to BlueSky expected names
            if 'new_heading_deg' in parameters:
                converted_params['new_heading'] = parameters['new_heading_deg']
            if 'target_altitude_ft' in parameters:
                converted_params['new_altitude'] = parameters['target_altitude_ft']
            if 'target_speed_kt' in parameters:
                converted_params['new_speed'] = parameters['target_speed_kt']
            
            # Handle route-aware parameters
            if 'waypoint_name' in parameters:
                converted_params['waypoint_name'] = parameters['waypoint_name']
                if 'lat' in parameters:
                    converted_params['lat'] = parameters['lat']
                if 'lon' in parameters:
                    converted_params['lon'] = parameters['lon']
            
            if 'via_waypoint' in parameters:
                converted_params['via_waypoint'] = parameters['via_waypoint']
                converted_params['resume_to_destination'] = parameters.get('resume_to_destination', True)
            
            # Convert to standard format
            result = {
                'aircraft_callsign': context.ownship_callsign,
                'resolution_type': resolution.get('resolution_type', 'no_action'),
                'reasoning': resolution.get('reasoning', 'Combined LLM resolution'),
                'confidence': resolution.get('confidence', 0.5),
                'source': 'llm_combined',
                'timestamp': self.current_time,
                'combined_response': combined_response,  # Keep full response for debugging
                'parameters': parameters  # Add the original parameters for route-aware processing
            }
            
            # Add converted parameters to result
            result.update(converted_params)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in combined LLM resolution generation: {e}")
            return None
    
    def _generate_geometric_resolution(self, conflict: Dict[str, Any], current_states: Dict[str, Any]) -> Dict[str, Any]:
        """Generate resolution using geometric algorithms"""
        aircraft_callsign = conflict['aircraft1']  # Usually ownship
        aircraft_state = current_states[aircraft_callsign]
        
        # Convert AircraftState to dict if needed
        if hasattr(aircraft_state, 'heading_deg'):
            current_heading = aircraft_state.heading_deg
        else:
            current_heading = getattr(aircraft_state, 'heading_deg', getattr(aircraft_state, 'heading', 0))
        
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
            resolution_type = resolution.get('resolution_type', '')
            parameters = resolution.get('parameters', {})
            
            # Handle traditional resolution types
            if 'new_heading' in resolution:
                success = self.bluesky_client.heading_command(
                    aircraft_callsign, resolution['new_heading']
                )
                if not success:
                    return False
                self.logger.info(f"Applied heading {resolution['new_heading']}° to {aircraft_callsign}")
            
            if 'new_altitude' in resolution:
                success = self.bluesky_client.altitude_command(
                    aircraft_callsign, resolution['new_altitude']
                )
                if not success:
                    return False
                self.logger.info(f"Applied altitude {resolution['new_altitude']}ft to {aircraft_callsign}")
            
            if 'new_speed' in resolution:
                success = self.bluesky_client.speed_command(
                    aircraft_callsign, resolution['new_speed']
                )
                if not success:
                    return False
                self.logger.info(f"Applied speed {resolution['new_speed']}kt to {aircraft_callsign}")
            
            # Handle new route-aware resolution types
            if resolution_type == "direct_to":
                waypoint_name = parameters.get('waypoint_name', 'DST')
                lat = parameters.get('lat')
                lon = parameters.get('lon')
                
                # Add waypoint if coordinates are provided
                if lat is not None and lon is not None:
                    waypoint_added = self.bluesky_client.add_waypoint(aircraft_callsign, waypoint_name, lat, lon)
                    if not waypoint_added:
                        self.logger.error(f"Failed to add waypoint {waypoint_name}")
                        return False
                
                # Wait for BlueSky to process waypoint addition
                import time
                time.sleep(0.5)
                
                # Direct to waypoint
                success = self.bluesky_client.direct_to(aircraft_callsign, waypoint_name)
                if not success:
                    self.logger.error(f"Failed to direct {aircraft_callsign} to {waypoint_name}")
                    return False
                self.logger.info(f"Applied direct_to {waypoint_name} for {aircraft_callsign}")
            
            elif resolution_type == "reroute_via":
                via = parameters.get('via_waypoint', {})
                waypoint_name = via.get('name', 'AVOID1')
                lat = via.get('lat', 0)
                lon = via.get('lon', 0)
                
                # Debug logging to track coordinate transfer
                self.logger.debug(f"Reroute via waypoint: {waypoint_name} at {lat:.4f},{lon:.4f}")
                self.logger.debug(f"Via waypoint structure: {via}")
                
                # CRITICAL FIX: Add via waypoint to route BEFORE directing to it
                # This fixes the "AVOID1 not found in the route" error
                waypoint_added = self.bluesky_client.add_waypoint(aircraft_callsign, waypoint_name, lat, lon)
                if not waypoint_added:
                    self.logger.error(f"Failed to add waypoint {waypoint_name} to route")
                    return False
                
                # Wait a moment for BlueSky to process the waypoint addition
                import time
                time.sleep(0.5)
                
                # Now direct to via waypoint
                success = self.bluesky_client.direct_to(aircraft_callsign, waypoint_name)
                if not success:
                    self.logger.error(f"Failed to direct {aircraft_callsign} to {waypoint_name}")
                    return False
                
                # Schedule auto-resume to destination
                if parameters.get('resume_to_destination', True):
                    self._schedule_resume_to_destination(aircraft_callsign, resolution)
                
                self.logger.info(f"Applied reroute_via {waypoint_name} for {aircraft_callsign}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying resolution to BlueSky: {e}")
            return False
    
    def _schedule_resume_to_destination(self, callsign: str, resolution: Dict[str, Any]):
        """Schedule auto-resume to final destination for reroute_via resolutions"""
        try:
            # Extract destination from resolution context (would be better to pass explicitly)
            # For now, we'll create a simple destination placeholder
            destination_name = "DST"  # Default destination
            
            # Try to extract destination from aircraft's original flight plan if available
            aircraft_states = self.bluesky_client.get_aircraft_states()
            if callsign in aircraft_states:
                # In a real implementation, you'd extract the final destination from the flight plan
                # For now, we'll use a placeholder
                pass
                
            self.resume_tasks[callsign] = {
                "dest_name": destination_name,
                "dest_lat": 0.0,  # Would be extracted from flight plan
                "dest_lon": 0.0,  # Would be extracted from flight plan
                "armed": True,
                "scheduled_at": self.current_time,
                "timeout_minutes": 8.0  # Auto-resume after 8 minutes if conditions not met
            }
            
            self.logger.info(f"Scheduled resume-to-destination for {callsign}")
            
        except Exception as e:
            self.logger.error(f"Error scheduling resume task for {callsign}: {e}")
    
    def _maybe_resume_to_destination(self, callsign: str, aircraft_states: Dict[str, Any]):
        """Check if aircraft should resume to destination"""
        task = self.resume_tasks.get(callsign)
        if not task or not task["armed"]:
            return
        
        try:
            # Condition A: Near the via waypoint (within 2 NM)
            near_via = self._near_current_direct_target(callsign, aircraft_states, radius_nm=2.0)
            
            # Condition B: Clear of conflicts for multiple cycles
            clear_for_cycles = self._ownship_clear_for_n_cycles(callsign, n=2, sep_nm=7.0)
            
            # Condition C: Timeout reached
            elapsed_minutes = (self.current_time - task["scheduled_at"]) / 60.0
            timeout_reached = elapsed_minutes >= task["timeout_minutes"]
            
            if near_via or clear_for_cycles or timeout_reached:
                # Resume to final destination
                dest_name = task["dest_name"]
                dest_lat = task["dest_lat"]
                dest_lon = task["dest_lon"]
                
                # Add final destination waypoint if coordinates available
                if dest_lat != 0.0 or dest_lon != 0.0:
                    self.bluesky_client.add_waypoint(callsign, dest_name, dest_lat, dest_lon)
                
                # Direct to destination
                success = self.bluesky_client.direct_to(callsign, dest_name)
                
                if success:
                    task["armed"] = False
                    reason = "near waypoint" if near_via else "clear of conflicts" if clear_for_cycles else "timeout"
                    self.logger.info(f"Resumed {callsign} to destination {dest_name} ({reason})")
                else:
                    self.logger.warning(f"Failed to resume {callsign} to destination")
                    
        except Exception as e:
            self.logger.error(f"Error in resume-to-destination for {callsign}: {e}")
    
    def _near_current_direct_target(self, callsign: str, aircraft_states: Dict[str, Any], radius_nm: float = 2.0) -> bool:
        """Check if aircraft is near its current direct-to target"""
        # Simplified implementation - in real scenario, would query BlueSky for current flight plan
        # For now, assume aircraft is near waypoint if it has been flying for some time
        return False  # Placeholder implementation
    
    def _ownship_clear_for_n_cycles(self, callsign: str, n: int = 2, sep_nm: float = 7.0) -> bool:
        """Check if ownship has been clear of conflicts for N cycles"""
        # Check recent conflict history for this aircraft
        recent_conflicts = [
            c for c in self.conflict_history[-10:]  # Last 10 conflicts
            if (c.get('aircraft1') == callsign or c.get('aircraft2') == callsign) and
               (self.current_time - c.get('timestamp', 0)) < (n * self.config.cycle_interval_seconds)
        ]
        
        # Consider clear if no recent conflicts
        return len(recent_conflicts) == 0
    
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
    
    def _save_resolution_data(self, resolution: Dict[str, Any], output_dir: Path):
        """Save resolution data for post-analysis"""
        try:
            resolution_file = output_dir / "resolutions.jsonl"
            
            # Create resolution snapshot
            resolution_snapshot = {
                'timestamp': self.current_time,
                'time_minutes': self.current_time / 60.0,
                'conflict_id': resolution.get('conflict_id', 'unknown'),
                'aircraft_callsign': resolution.get('aircraft_callsign', 'unknown'),
                'resolution_type': resolution.get('resolution_type', 'unknown'),
                'parameters': resolution.get('parameters', {}),
                'reasoning': resolution.get('reasoning', ''),
                'confidence': resolution.get('confidence', 0.0),
                'method': resolution.get('method', 'LLM'),
                'success': resolution.get('success', False),
                'applied_at': resolution.get('applied_at', self.current_time)
            }
            
            with open(resolution_file, 'a') as f:
                f.write(json.dumps(resolution_snapshot) + '\n')
                
        except Exception as e:
            self.logger.warning(f"Failed to save resolution data: {e}")

    
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
                # Arm the intruder aircraft with proper autopilot settings for movement
                callsign = intruder['callsign']
                self.bluesky_client.heading_command(callsign, spawn_position['heading_deg'])
                self.bluesky_client.altitude_command(callsign, spawn_position['altitude_ft'])
                self.bluesky_client.set_speed(callsign, spawn_position['speed_kt'])
                
                self.active_aircraft[intruder['callsign']] = {
                    'callsign': intruder['callsign'],
                    'type': 'intruder',
                    'created_at': self.current_time,
                    'spawn_data': intruder
                }
                
                # Verify both ownship and intruder are present
                all_states = self.bluesky_client.get_aircraft_states()
                self.logger.info(f"✅ Intruder {callsign} injected. Active aircraft: {list(all_states.keys())}")
                
                # Check distances between aircraft for conflict potential
                if len(all_states) >= 2:
                    aircraft_list = list(all_states.items())
                    for i, (cs1, state1) in enumerate(aircraft_list):
                        for cs2, state2 in aircraft_list[i+1:]:
                            # Calculate rough distance
                            import math
                            dlat = state2.latitude - state1.latitude
                            dlon = state2.longitude - state1.longitude
                            dist_nm = math.sqrt(dlat**2 + dlon**2) * 60  # Rough conversion
                            
                            alt_diff = abs(state2.altitude_ft - state1.altitude_ft)
                            
                            self.logger.info(f"Aircraft separation: {cs1} vs {cs2}: "
                                           f"H={dist_nm:.1f}NM, V={alt_diff:.0f}ft")
            
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
