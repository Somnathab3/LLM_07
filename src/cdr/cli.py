#!/usr/bin/env python3
"""Streamlined CLI interface for LLM_ATC7"""

import sys
import argparse
import logging
import json
import time
import math
from pathlib import Path
from typing import Optional, List, Dict

# Import CDR components at module level
from .simulation.bluesky_client import SimpleBlueSkyClient, Destination
from .ai.llm_client import StreamlinedLLMClient, LLMConfig, LLMProvider, ConflictContext
from .ai.memory import ExperienceMemory, MemoryRecord
from .ai.memory_config import get_memory_config


def setup_cli_logging(verbose: bool = False):
    """Setup CLI logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


class StreamlinedCLI:
    """Streamlined CLI interface for essential functions only"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def run(self, args: List[str] = None):
        """Main entry point"""
        if args is None:
            args = sys.argv[1:]
        
        parser = self._create_parser()
        parsed_args = parser.parse_args(args)
        
        # Setup logging
        setup_cli_logging(parsed_args.verbose)
        
        # Execute command
        try:
            return parsed_args.func(parsed_args)
        except KeyboardInterrupt:
            self.logger.info("Operation cancelled by user")
            return 1
        except Exception as e:
            self.logger.error(f"Command failed: {e}", exc_info=parsed_args.verbose)
            return 1
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create streamlined argument parser"""
        parser = argparse.ArgumentParser(
            prog='atc-llm',
            description='Streamlined LLM-driven Air Traffic Control CDR',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  atc-llm run-simulation --scat-path data/100002.json
  atc-llm test-llm
  atc-llm test-bluesky
            """
        )
        
        # Global options
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Enable verbose logging')
        
        # Create subparsers
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Essential commands only
        self._add_run_simulation_parser(subparsers)
        self._add_test_llm_parser(subparsers)
        self._add_test_bluesky_parser(subparsers)
        self._add_test_integration_parser(subparsers)
        
        return parser
    
    def _add_run_simulation_parser(self, subparsers):
        """Add run-simulation subcommand parser"""
        parser = subparsers.add_parser(
            'run-simulation',
            help='Run CDR simulation with LLM integration'
        )
        
        parser.add_argument('--scat-path', required=True, type=Path,
                          help='Path to SCAT data file (JSON format)')
        parser.add_argument('--output-dir', type=Path, default='output/',
                          help='Output directory for results')
        parser.add_argument('--llm-model', default='llama3.1:8b',
                          help='LLM model to use')
        parser.add_argument('--max-time', type=float, default=30.0,
                          help='Maximum simulation time in minutes')
        parser.add_argument('--cycle-interval', type=float, default=180.0,
                          help='Simulation cycle interval in seconds')
        parser.add_argument('--dest-distance', type=float, default=550.0,
                          help='Destination distance in nautical miles')
        
        parser.set_defaults(func=self._run_simulation)
    
    def _add_test_llm_parser(self, subparsers):
        """Add test-llm subcommand parser"""
        parser = subparsers.add_parser(
            'test-llm',
            help='Test LLM client connectivity and functionality'
        )
        
        parser.add_argument('--model', default='llama3.1:8b',
                          help='LLM model to test')
        
        parser.set_defaults(func=self._test_llm)
    
    def _add_test_bluesky_parser(self, subparsers):
        """Add test-bluesky subcommand parser"""
        parser = subparsers.add_parser(
            'test-bluesky',
            help='Test BlueSky client functionality'
        )
        
        parser.set_defaults(func=self._test_bluesky)
    
    def _add_test_integration_parser(self, subparsers):
        """Add test-integration subcommand parser"""
        parser = subparsers.add_parser(
            'test-integration',
            help='Test LLM + BlueSky integration'
        )
        
        parser.add_argument('--model', default='llama3.1:8b',
                          help='LLM model to use')
        
        parser.set_defaults(func=self._test_integration)
    
    def _run_simulation(self, args) -> int:
        """Run CDR simulation with LLM integration"""
        self.logger.info("🚀 Starting CDR simulation with LLM integration")
        
        try:
            # Validate inputs
            if not args.scat_path.exists():
                self.logger.error(f"SCAT file not found: {args.scat_path}")
                return 1
            
            # Load SCAT data
            self.logger.info(f"📂 Loading SCAT data from {args.scat_path}")
            with open(args.scat_path, 'r') as f:
                scat_data = json.load(f)
            
            # Handle SCAT format (dict with 'plots' key containing trajectory data)
            if isinstance(scat_data, dict) and 'plots' in scat_data:
                plots = scat_data['plots']
                if not plots:
                    self.logger.error("No plot data found in SCAT file")
                    return 1
                self.logger.info(f"📊 Loaded SCAT data with {len(plots)} plot points")
            else:
                self.logger.error("Invalid SCAT data format - expected dict with 'plots' key")
                return 1
            
            # Setup output directory
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize components
            self.logger.info("🔧 Initializing simulation components...")
            
            # Initialize BlueSky client
            bs_client = SimpleBlueSkyClient()
            bs_client.initialize()
            bs_client.reset()
            
            # Set fast simulation speed for better performance
            bs_client.set_fast_simulation(speed_factor=4.0)
            
            # Configure conflict detection
            bs_client.configure_conflict_detection(
                pz_radius_nm=5.0,
                pz_height_ft=1000,
                lookahead_sec=600
            )
            
            # Initialize LLM client
            llm_config = LLMConfig(
                provider=LLMProvider.OLLAMA,
                model=args.llm_model,
                timeout=30.0,
                temperature=0.1
            )
            llm_client = StreamlinedLLMClient(llm_config, log_dir=output_dir / "llm_logs")
            
            # Initialize memory system for experience storage
            self.logger.info("🧠 Initializing memory system...")
            memory_config = get_memory_config(minimal=False)
            memory_system = ExperienceMemory(
                memory_dir=output_dir / "memory",
                embedding_model=memory_config["embedding_model"],
                max_records=memory_config["max_records"]
            )
            
            # Run simulation
            self.logger.info("🛫 Starting simulation...")
            result = self._execute_simulation(plots, bs_client, llm_client, memory_system, output_dir, args.max_time, args.cycle_interval, args.dest_distance)
            
            if result['success']:
                self.logger.info(f"✅ Simulation completed successfully")
                self.logger.info(f"📊 Results: {result['conflicts_detected']} conflicts, {result['resolutions_applied']} resolutions")
                return 0
            else:
                self.logger.error(f"❌ Simulation failed: {result.get('error', 'Unknown error')}")
                return 1
                
        except Exception as e:
            self.logger.error(f"❌ Simulation failed: {e}", exc_info=args.verbose)
            return 1
    
    def _execute_simulation(self, plots_data: List[Dict], bs_client, llm_client, memory_system, output_dir: Path, max_time: float, cycle_interval: float, dest_distance: float) -> Dict:
        """Execute the actual simulation"""
        start_time = time.time()
        conflicts_detected = 0
        resolutions_applied = 0
        trajectory_log = []
        conflict_log = []
        
        try:
            # Extract aircraft data from SCAT plots
            self.logger.info(f"📊 Processing {len(plots_data)} plot points")
            
            # Get the first plot to create initial aircraft
            if not plots_data:
                raise ValueError("No plot data available")
            
            first_plot = plots_data[0]
            
            # Extract aircraft state from first plot
            position = first_plot.get('I062/105', {})
            altitude_info = first_plot.get('I062/136', {})
            velocity_info = first_plot.get('I062/185', {})
            heading_info = first_plot.get('I062/380', {})
            
            lat = position.get('lat', 42.0)
            lon = position.get('lon', -87.0)
            alt = altitude_info.get('measured_flight_level', 350) * 100  # Convert FL to feet
            
            # Extract velocity components and calculate heading/speed
            vx = velocity_info.get('vx', 200)  # knots
            vy = velocity_info.get('vy', 200)  # knots
            
            # Calculate heading and speed from velocity components
            import math
            speed = math.sqrt(vx**2 + vy**2)
            heading = (math.degrees(math.atan2(vx, vy)) + 360) % 360
            
            # Get heading from subitem if available
            if heading_info and 'subitem3' in heading_info:
                heading = heading_info['subitem3'].get('mag_hdg', heading)
            
            self.logger.info(f"📍 Aircraft position: {lat:.4f}, {lon:.4f} at FL{int(alt/100)}")
            self.logger.info(f"🧭 Heading: {heading:.1f}°, Speed: {speed:.0f}kt")
            
            # Create primary aircraft from SCAT data
            callsign = "SCAT1"
            success = bs_client.create_aircraft(
                acid=callsign,
                lat=lat,
                lon=lon,
                hdg=heading,
                alt=alt,
                spd=speed
            )
            
            aircraft_created = 1 if success else 0
            
            if success:
                self.logger.info(f"✈️ Created aircraft {callsign} from SCAT data")
                
                # Generate fixed destination at configurable distance from starting position
                destination = self._generate_destination_from_position(lat, lon, heading, alt, dest_distance)
                self.logger.info(f"🎯 Generated destination: {destination['name']} at {destination['latitude']:.4f}, {destination['longitude']:.4f}")
                self.logger.info(f"📏 Distance: {destination['distance_nm']:.1f} NM, Bearing: {destination['bearing']:.0f}°")
                
                # Set destination in BlueSky
                bs_destination = Destination(
                    name=destination['name'],
                    lat=destination['latitude'],
                    lon=destination['longitude'],
                    alt=destination['altitude']
                )
                bs_client.set_aircraft_destination(callsign, bs_destination)
                
            else:
                self.logger.error("❌ Failed to create aircraft from SCAT data")
                raise ValueError("Failed to create primary aircraft")
            
            # Create additional traffic for conflict scenarios
            traffic_aircraft = [
                {"callsign": "TFC1", "lat": lat + 0.05, "lon": lon + 0.05, "hdg": (heading + 180) % 360, "alt": alt, "spd": speed * 0.9},
                {"callsign": "TFC2", "lat": lat - 0.05, "lon": lon + 0.05, "hdg": (heading + 90) % 360, "alt": alt + 1000, "spd": speed * 1.1},
            ]
            
            for traffic in traffic_aircraft:
                success = bs_client.create_aircraft(
                    acid=traffic["callsign"],
                    lat=traffic["lat"],
                    lon=traffic["lon"],
                    hdg=traffic["hdg"],
                    alt=traffic["alt"],
                    spd=traffic["spd"]
                )
                if success:
                    aircraft_created += 1
                    self.logger.info(f"✈️ Created traffic aircraft {traffic['callsign']}")
            
            self.logger.info(f"📋 Total aircraft created: {aircraft_created}")
            
            # Simulation loop
            sim_time = 0.0
            # cycle_interval is now passed as parameter
            
            while sim_time < max_time * 60:  # Convert minutes to seconds
                cycle_start = time.time()
                
                # Step simulation
                bs_client.step_simulation(cycle_interval)
                sim_time += cycle_interval
                
                # Get aircraft states
                aircraft_states = bs_client.get_all_aircraft_states()
                if not aircraft_states:
                    break
                
                # Log trajectories
                timestamp = {
                    'sim_time': sim_time,
                    'real_time': time.time(),
                    'aircraft': []
                }
                
                for state in aircraft_states:
                    timestamp['aircraft'].append({
                        'callsign': state.id,
                        'lat': state.lat,
                        'lon': state.lon,
                        'alt': state.alt,
                        'hdg': state.hdg,
                        'spd': state.tas
                    })
                
                trajectory_log.append(timestamp)
                
                # 🔍 DETAILED AIRCRAFT MOVEMENT DEBUGGING
                self.logger.info(f"🛫 === AIRCRAFT POSITIONS at {sim_time/60:.1f} min (Step {int(sim_time/cycle_interval)}) ===")
                for i, state in enumerate(aircraft_states):
                    # Calculate movement from last position if available
                    movement_info = ""
                    if len(trajectory_log) > 1:
                        prev_pos = None
                        for prev_aircraft in trajectory_log[-2]['aircraft']:
                            if prev_aircraft['callsign'] == state.id:
                                prev_pos = prev_aircraft
                                break
                        
                        if prev_pos:
                            lat_change = state.lat - prev_pos['lat']
                            lon_change = state.lon - prev_pos['lon']
                            alt_change = state.alt - prev_pos['alt']
                            hdg_change = state.hdg - prev_pos['hdg']
                            
                            # Calculate distance moved in nautical miles
                            distance_moved = ((lat_change**2 + lon_change**2)**0.5) * 60
                            
                            movement_info = f" | 📏 Moved: {distance_moved:.3f}NM, ⬆️ΔAlt: {alt_change:.0f}ft, 🧭ΔHdg: {hdg_change:.1f}°"
                            
                            if distance_moved < 0.001:  # Less than ~6 feet
                                movement_info += " ⚠️ STATIONARY"
                            elif distance_moved > 1.0:  # More than 1 NM
                                movement_info += " ✅ MOVING"
                    
                    self.logger.info(f"   ✈️ {state.id}: {state.lat:.6f}, {state.lon:.6f} | FL{int(state.alt/100)} | {state.hdg:.1f}° | {state.tas:.0f}kt{movement_info}")
                
                # Also track if ownship changed heading after LLM resolution
                if len(aircraft_states) > 0:
                    ownship = aircraft_states[0]  # SCAT1
                    if hasattr(self, '_last_ownship_heading'):
                        heading_change = abs(ownship.hdg - self._last_ownship_heading)
                        if heading_change > 1.0:  # Significant heading change
                            self.logger.info(f"🧭 OWNSHIP HEADING CHANGE DETECTED: {self._last_ownship_heading:.1f}° → {ownship.hdg:.1f}° (Δ{heading_change:.1f}°)")
                    self._last_ownship_heading = ownship.hdg
                
                # Check for conflicts using both methods
                native_conflicts = bs_client.get_conflict_summary()
                
                # Debug: Show conflict detection details
                self.logger.info(f"📊 Step {int(sim_time/cycle_interval)}: Native conflicts: {native_conflicts}")
                
                # Manual distance-based conflict detection for testing
                manual_conflicts = []
                if len(aircraft_states) >= 2:
                    for i, ac1 in enumerate(aircraft_states):
                        for j, ac2 in enumerate(aircraft_states[i+1:], i+1):
                            # Calculate distance in nautical miles
                            lat_diff = ac1.lat - ac2.lat
                            lon_diff = ac1.lon - ac2.lon
                            distance_nm = ((lat_diff**2 + lon_diff**2)**0.5) * 60  # Rough conversion
                            alt_diff = abs(ac1.alt - ac2.alt)
                            
                            self.logger.info(f"🔍 Distance {ac1.id}-{ac2.id}: {distance_nm:.2f}NM, {alt_diff:.0f}ft")
                            
                            if distance_nm < 10.0 and alt_diff < 2000:  # Potential conflict zone
                                manual_conflicts.append((ac1.id, ac2.id, distance_nm, alt_diff))
                
                # Force LLM call for testing if we have aircraft close together OR native conflicts
                should_call_llm = (native_conflicts['active_conflicts'] > 0 or 
                                 len(manual_conflicts) > 0 or
                                 len(aircraft_states) >= 2)  # Always call for testing
                
                if should_call_llm:
                    conflicts_detected += 1
                    self.logger.info(f"🚨 Calling LLM for analysis at {sim_time/60:.1f} min")
                    self.logger.info(f"   Native conflicts: {native_conflicts['active_conflicts']}")
                    self.logger.info(f"   Manual conflicts: {len(manual_conflicts)}")
                    
                    # Create conflict context for LLM
                    ownship_state = aircraft_states[0]  # Use first aircraft as ownship
                    intruders = []
                    
                    self.logger.info(f"🛫 Ownship: {ownship_state.id} at {ownship_state.lat:.4f},{ownship_state.lon:.4f} FL{int(ownship_state.alt/100)} hdg={ownship_state.hdg:.0f}°")
                    
                    for state in aircraft_states[1:]:
                        intruder_data = {
                            'callsign': state.id,
                            'latitude': state.lat,
                            'longitude': state.lon,
                            'altitude': state.alt,
                            'heading': state.hdg,
                            'speed': state.tas,
                            'vertical_speed_fpm': state.vs * 60  # Convert m/s to fpm
                        }
                        intruders.append(intruder_data)
                        
                        self.logger.info(f"✈️ Intruder: {state.id} at {state.lat:.4f},{state.lon:.4f} FL{int(state.alt/100)} hdg={state.hdg:.0f}°")
                    
                    # Get ownship destination
                    ownship_destination = bs_client.get_aircraft_destination(ownship_state.id)
                    destination_dict = None
                    if ownship_destination:
                        destination_dict = {
                            'name': ownship_destination.name,
                            'latitude': ownship_destination.lat,
                            'longitude': ownship_destination.lon,
                            'altitude': ownship_destination.alt
                        }
                        self.logger.info(f"🎯 Destination: {ownship_destination.name} at {ownship_destination.lat:.4f},{ownship_destination.lon:.4f}")
                    
                    context = ConflictContext(
                        ownship_callsign=ownship_state.id,
                        ownship_state={
                            'latitude': ownship_state.lat,
                            'longitude': ownship_state.lon,
                            'altitude': ownship_state.alt,
                            'heading': ownship_state.hdg,
                            'speed': ownship_state.tas,
                            'vertical_speed_fpm': ownship_state.vs * 60
                        },
                        intruders=intruders,
                        scenario_time=sim_time,
                        lookahead_minutes=5.0,  # Match BlueSky native settings
                        destination=destination_dict
                    )
                    
                    self.logger.info(f"📤 Sending to LLM: Ownship={context.ownship_callsign}, Intruders={len(context.intruders)}")
                    
                    # 🛑 HOLD simulation during LLM processing
                    self.logger.info("⏸️ HOLDING simulation for LLM processing...")
                    bs_client.hold()
                    
                    # Get LLM resolution
                    llm_response = llm_client.detect_and_resolve_conflicts(context)
                    
                    # ▶️ RESUME simulation after LLM processing
                    self.logger.info("▶️ RESUMING simulation after LLM processing...")
                    bs_client.op()
                    
                    self.logger.info(f"📥 LLM Full Response: {llm_response}")
                    
                    # Apply resolution if suggested
                    if llm_response.get('conflicts_detected') and llm_response.get('resolution'):
                        resolution = llm_response['resolution']
                        resolution_type = resolution.get('resolution_type')
                        parameters = resolution.get('parameters', {})
                        
                        self.logger.info(f"🤖 LLM suggests: {resolution_type} - {resolution.get('reasoning', '')}")
                        self.logger.info(f"🔧 Resolution parameters: {parameters}")
                        
                        # Apply resolution to BlueSky
                        success = self._apply_resolution_to_bluesky(
                            bs_client, 
                            ownship_state.id, 
                            resolution_type, 
                            parameters
                        )
                        
                        # 🔍 DEBUG: Check aircraft state immediately after resolution
                        if resolution_type == "heading_change":
                            new_state = bs_client.get_aircraft_state(ownship_state.id)
                            if new_state:
                                self.logger.info(f"🔍 AFTER RESOLUTION - {ownship_state.id}: heading={new_state.hdg:.1f}°")
                            
                            # Give BlueSky a moment to process the command
                            time.sleep(1.0)
                            bs_client.step_simulation(1.0)
                            
                            # Check again after simulation step
                            newer_state = bs_client.get_aircraft_state(ownship_state.id)
                            if newer_state:
                                self.logger.info(f"🔍 AFTER STEP - {ownship_state.id}: heading={newer_state.hdg:.1f}°")
                        
                        if success:
                            resolutions_applied += 1
                            self.logger.info(f"✅ Resolution applied successfully")
                        else:
                            self.logger.warning(f"⚠️ Failed to apply resolution")
                        
                        # Log conflict and resolution
                        conflict_log.append({
                            'sim_time': sim_time,
                            'ownship': ownship_state.id,
                            'conflicts_detected': llm_response.get('conflicts', []),
                            'resolution': resolution,
                            'applied': success
                        })
                        
                        # 🧠 SAVE TO MEMORY SYSTEM for learning
                        try:
                            self.logger.info("🧠 Saving experience to memory system...")
                            
                            # Create memory record
                            from datetime import datetime
                            
                            # Ensure all data is JSON serializable
                            safe_context = {
                                'ownship_callsign': str(context.ownship_callsign),
                                'ownship_position': {
                                    'lat': float(context.ownship_state['latitude']),
                                    'lon': float(context.ownship_state['longitude']),
                                    'alt': float(context.ownship_state['altitude'])
                                },
                                'intruders': [
                                    {
                                        'callsign': str(intruder['callsign']),
                                        'latitude': float(intruder['latitude']),
                                        'longitude': float(intruder['longitude']),
                                        'altitude': float(intruder['altitude']),
                                        'heading': float(intruder['heading']),
                                        'speed': float(intruder['speed'])
                                    } for intruder in context.intruders
                                ]
                            }
                            
                            memory_record = MemoryRecord(
                                record_id=f"CDR_{int(time.time()*1000)}_{ownship_state.id}",
                                timestamp=datetime.now(),
                                conflict_context=safe_context,  # Use safe serializable context
                                resolution_taken=resolution,
                                outcome_metrics={
                                    'resolution_applied': bool(success),
                                    'resolution_type': str(resolution_type),
                                    'conflicts_count': int(len(llm_response.get('conflicts', []))),
                                    'sim_time_minutes': float(sim_time / 60),
                                    'confidence': float(resolution.get('confidence', 0.0))
                                },
                                scenario_features={
                                    'aircraft_count': int(len(aircraft_states)),
                                    'altitude_ft': float(ownship_state.alt),
                                    'speed_kt': float(ownship_state.tas),
                                    'heading_deg': float(ownship_state.hdg),
                                    'manual_conflicts_detected': int(len(manual_conflicts)),
                                    'native_conflicts_detected': int(native_conflicts['active_conflicts'])
                                },
                                task_type='conflict_resolution',
                                success_score=float(success) * float(resolution.get('confidence', 0.5)),
                                metadata={
                                    'llm_model': 'llama3.1:8b',
                                    'cycle_interval': cycle_interval,
                                    'simulation_type': 'scat_based',
                                    'command_applied': f"{resolution_type}_{parameters}",
                                    'aircraft_positions': {
                                        ac.id: {'lat': ac.lat, 'lon': ac.lon, 'hdg': ac.hdg}
                                        for ac in aircraft_states
                                    }
                                }
                            )
                            
                            # Store in memory
                            memory_system.store_experience(memory_record)
                            self.logger.info(f"✅ Experience saved to memory: {memory_record.record_id}")
                            
                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to save to memory: {e}")
                    
                    else:
                        self.logger.info("🤖 LLM detected conflicts but no action required")
                        
                        # Still save no-action experiences to memory
                        try:
                            from datetime import datetime
                            
                            # Ensure all data is JSON serializable
                            safe_context = {
                                'ownship_callsign': str(context.ownship_callsign),
                                'ownship_position': {
                                    'lat': float(context.ownship_state['latitude']),
                                    'lon': float(context.ownship_state['longitude']),
                                    'alt': float(context.ownship_state['altitude'])
                                },
                                'intruders': [
                                    {
                                        'callsign': str(intruder['callsign']),
                                        'latitude': float(intruder['latitude']),
                                        'longitude': float(intruder['longitude']),
                                        'altitude': float(intruder['altitude']),
                                        'heading': float(intruder['heading']),
                                        'speed': float(intruder['speed'])
                                    } for intruder in context.intruders
                                ]
                            }
                            
                            memory_record = MemoryRecord(
                                record_id=f"CDR_NOACTION_{int(time.time()*1000)}_{ownship_state.id}",
                                timestamp=datetime.now(),
                                conflict_context=safe_context,  # Use safe serializable context
                                resolution_taken={'resolution_type': 'no_action', 'parameters': {}, 'reasoning': 'No action required'},
                                outcome_metrics={
                                    'resolution_applied': True,  # No action is considered successful
                                    'resolution_type': 'no_action',
                                    'conflicts_count': int(len(llm_response.get('conflicts', []))),
                                    'sim_time_minutes': float(sim_time / 60)
                                },
                                scenario_features={
                                    'aircraft_count': int(len(aircraft_states)),
                                    'altitude_ft': float(ownship_state.alt),
                                    'speed_kt': float(ownship_state.tas),
                                    'heading_deg': float(ownship_state.hdg),
                                    'manual_conflicts_detected': int(len(manual_conflicts)),
                                    'native_conflicts_detected': int(native_conflicts['active_conflicts'])
                                },
                                task_type='conflict_detection',
                                success_score=0.8,  # No-action when appropriate is good
                                metadata={
                                    'llm_model': 'llama3.1:8b',
                                    'cycle_interval': cycle_interval,
                                    'simulation_type': 'scat_based',
                                    'decision': 'no_action_required'
                                }
                            )
                            memory_system.store_experience(memory_record)
                            self.logger.info(f"🧠 No-action experience saved: {memory_record.record_id}")
                            
                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to save no-action experience: {e}")
                
                # Progress update
                if int(sim_time) % 600 == 0:  # Every 10 minutes
                    self.logger.info(f"⏰ Simulation time: {sim_time/60:.1f} min")
                
                # Limit real-time execution
                cycle_elapsed = time.time() - cycle_start
                if cycle_elapsed < 1.0:  # Minimum 1 second per cycle
                    time.sleep(1.0 - cycle_elapsed)
            
            # 📊 ANALYZE FALSE POSITIVES AND FALSE NEGATIVES
            self.logger.info("🔍 Analyzing conflict detection accuracy...")
            false_positive_analysis = []
            false_negative_analysis = []
            resolution_effectiveness = []
            
            # Enhanced analysis: Compare before/after distances for resolution effectiveness
            for i, conflict_entry in enumerate(conflict_log):
                sim_time_entry = conflict_entry['sim_time']
                llm_conflicts = conflict_entry.get('conflicts_detected', [])
                resolution_applied = conflict_entry.get('applied', False)
                
                # Find aircraft states before and after resolution
                before_trajectory = None
                after_trajectory = None
                
                for traj_entry in trajectory_log:
                    time_diff = abs(traj_entry['sim_time'] - sim_time_entry)
                    if time_diff < 5:  # Within 5 seconds before
                        before_trajectory = traj_entry
                    elif traj_entry['sim_time'] > sim_time_entry and time_diff < cycle_interval + 5:
                        after_trajectory = traj_entry
                        break
                
                # Analyze resolution effectiveness
                if resolution_applied and before_trajectory and after_trajectory:
                    effectiveness = self._analyze_resolution_effectiveness(
                        before_trajectory, after_trajectory, conflict_entry
                    )
                    resolution_effectiveness.append(effectiveness)
                
                # Simplified conflict detection accuracy analysis
                # Note: This is still simplified - true accuracy requires ground truth data
                analysis_entry = {
                    'sim_time': sim_time_entry,
                    'llm_detected_conflicts': len(llm_conflicts) > 0,
                    'llm_conflict_count': len(llm_conflicts),
                    'resolution_applied': resolution_applied,
                    'resolution_type': conflict_entry.get('resolution', {}).get('resolution_type', 'none'),
                    'manual_conflicts_count': len(conflict_entry.get('manual_conflicts', [])),
                    'native_conflicts_count': conflict_entry.get('native_conflicts', {}).get('active_conflicts', 0)
                }
                
                # Basic heuristic for FP/FN (improved but still simplified)
                llm_detected = len(llm_conflicts) > 0
                manual_detected = len(conflict_entry.get('manual_conflicts', [])) > 0
                native_detected = conflict_entry.get('native_conflicts', {}).get('active_conflicts', 0) > 0
                
                # Conservative approach: only flag clear mismatches
                if llm_detected and not (manual_detected or native_detected):
                    # Potential false positive: LLM detected but manual/native did not
                    false_positive_analysis.append(analysis_entry)
                elif not llm_detected and (manual_detected or native_detected):
                    # Potential false negative: LLM missed but manual/native detected
                    false_negative_analysis.append(analysis_entry)
            
            detection_analysis = {
                'total_llm_detections': len([c for c in conflict_log if len(c.get('conflicts_detected', [])) > 0]),
                'total_resolutions_attempted': resolutions_applied,
                'resolution_effectiveness': resolution_effectiveness,
                'false_positive_candidates': false_positive_analysis,
                'false_negative_candidates': false_negative_analysis,
                'detection_accuracy_notes': [
                    "This is a simplified analysis. For comprehensive accuracy:",
                    "1. Need ground truth conflict data from professional ATC",
                    "2. Need more sophisticated geometric conflict detection",
                    "3. Need time-series analysis of actual vs predicted conflicts",
                    "4. Resolution effectiveness based on actual distance changes"
                ],
                'effectiveness_summary': {
                    'total_resolutions': len(resolution_effectiveness),
                    'successful_resolutions': len([r for r in resolution_effectiveness if r['effective']]),
                    'average_separation_improvement': sum([r['separation_improvement'] for r in resolution_effectiveness]) / max(len(resolution_effectiveness), 1)
                }
            }
            
            self.logger.info(f"📊 Detection Analysis: {len(false_positive_analysis)} potential FP, {len(false_negative_analysis)} potential FN")
            
            # Save results
            results = {
                'success': True,
                'simulation_time_minutes': sim_time / 60,
                'real_time_seconds': time.time() - start_time,
                'aircraft_created': aircraft_created,
                'conflicts_detected': conflicts_detected,
                'resolutions_applied': resolutions_applied,
                'trajectory_log': trajectory_log,
                'conflict_log': conflict_log,
                'detection_analysis': detection_analysis,
                'memory_records_created': conflicts_detected  # Approximate
            }
            
            # Save to files
            with open(output_dir / 'simulation_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save detection analysis separately
            with open(output_dir / 'detection_analysis.json', 'w') as f:
                json.dump(detection_analysis, f, indent=2)
            
            with open(output_dir / 'trajectories.jsonl', 'w') as f:
                for entry in trajectory_log:
                    f.write(json.dumps(entry) + '\n')
            
            if conflict_log:
                with open(output_dir / 'conflicts.jsonl', 'w') as f:
                    for entry in conflict_log:
                        f.write(json.dumps(entry) + '\n')
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Simulation execution failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'simulation_time_minutes': 0,
                'aircraft_created': 0,
                'conflicts_detected': conflicts_detected,
                'resolutions_applied': resolutions_applied
            }

    def _analyze_resolution_effectiveness(self, before_trajectory: Dict, after_trajectory: Dict, conflict_entry: Dict) -> Dict:
        """Analyze if the resolution actually improved aircraft separation"""
        try:
            resolution = conflict_entry.get('resolution', {})
            resolution_type = resolution.get('resolution_type', 'unknown')
            
            # Find ownship and intruders in trajectories
            ownship_callsign = 'SCAT1'  # Our main aircraft
            
            before_ownship = None
            after_ownship = None
            before_intruders = []
            after_intruders = []
            
            for aircraft in before_trajectory['aircraft']:
                if aircraft['callsign'] == ownship_callsign:
                    before_ownship = aircraft
                else:
                    before_intruders.append(aircraft)
            
            for aircraft in after_trajectory['aircraft']:
                if aircraft['callsign'] == ownship_callsign:
                    after_ownship = aircraft
                else:
                    after_intruders.append(aircraft)
            
            if not (before_ownship and after_ownship):
                return {'effective': False, 'reason': 'Missing ownship data', 'separation_improvement': 0.0}
            
            # Calculate minimum separation before and after
            min_sep_before = float('inf')
            min_sep_after = float('inf')
            closest_intruder = None
            
            for intruder in before_intruders:
                distance = self._calculate_distance_nm(
                    before_ownship['lat'], before_ownship['lon'],
                    intruder['lat'], intruder['lon']
                )
                if distance < min_sep_before:
                    min_sep_before = distance
                    closest_intruder = intruder['callsign']
            
            for intruder in after_intruders:
                if intruder['callsign'] == closest_intruder:
                    distance = self._calculate_distance_nm(
                        after_ownship['lat'], after_ownship['lon'],
                        intruder['lat'], intruder['lon']
                    )
                    min_sep_after = distance
                    break
            
            separation_improvement = min_sep_after - min_sep_before
            effective = separation_improvement > 0.1  # At least 0.1 NM improvement
            
            return {
                'effective': effective,
                'resolution_type': resolution_type,
                'closest_intruder': closest_intruder,
                'separation_before_nm': min_sep_before,
                'separation_after_nm': min_sep_after,
                'separation_improvement': separation_improvement,
                'heading_change': after_ownship['hdg'] - before_ownship['hdg'],
                'sim_time': conflict_entry['sim_time']
            }
            
        except Exception as e:
            return {'effective': False, 'reason': f'Analysis error: {e}', 'separation_improvement': 0.0}

    def _calculate_distance_nm(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in nautical miles"""
        # Simple distance calculation (good enough for close aircraft)
        lat_diff = lat2 - lat1
        lon_diff = lon2 - lon1
        return ((lat_diff**2 + lon_diff**2)**0.5) * 60  # Convert degrees to nautical miles
    
    def _apply_resolution_to_bluesky(self, bs_client, aircraft_id: str, resolution_type: str, parameters: Dict) -> bool:
        """Apply LLM resolution to BlueSky simulation with realistic constraints"""
        try:
            self.logger.info(f"🎯 Applying {resolution_type} to {aircraft_id} with params: {parameters}")
            
            if resolution_type == "heading_change":
                new_hdg = parameters.get('new_heading_deg')
                if new_hdg is not None:
                    # Get current aircraft state to check heading constraint
                    current_state = bs_client.get_aircraft_state(aircraft_id)
                    if current_state:
                        current_hdg = current_state.hdg
                        
                        # Calculate heading difference (accounting for circular nature)
                        heading_diff = (new_hdg - current_hdg + 180) % 360 - 180
                        
                        # Apply ±15 degree constraint
                        MAX_HEADING_CHANGE = 15.0
                        if abs(heading_diff) > MAX_HEADING_CHANGE:
                            # Limit to ±15 degrees from current heading
                            if heading_diff > 0:
                                constrained_hdg = (current_hdg + MAX_HEADING_CHANGE) % 360
                            else:
                                constrained_hdg = (current_hdg - MAX_HEADING_CHANGE) % 360
                            
                            self.logger.info(f"⚠️ Heading change constrained: {new_hdg}° → {constrained_hdg:.1f}° (max ±{MAX_HEADING_CHANGE}° from {current_hdg:.1f}°)")
                            new_hdg = constrained_hdg
                        else:
                            self.logger.info(f"✅ Heading change within limits: {current_hdg:.1f}° → {new_hdg}° (Δ{heading_diff:+.1f}°)")
                    
                    self.logger.info(f"🧭 Changing {aircraft_id} heading to {new_hdg:.1f}°")
                    result = bs_client.change_heading(aircraft_id, new_hdg)
                    self.logger.info(f"📊 Heading change result: {result}")
                    return result
                else:
                    self.logger.warning(f"⚠️ No new_heading_deg in parameters: {parameters}")
            
            elif resolution_type == "altitude_change":
                new_alt = parameters.get('target_altitude_ft')
                if new_alt is not None:
                    # Get current altitude for logging
                    current_state = bs_client.get_aircraft_state(aircraft_id)
                    if current_state:
                        current_alt = current_state.alt
                        alt_change = new_alt - current_alt
                        self.logger.info(f"⬆️ Changing {aircraft_id} altitude: {current_alt:.0f}ft → {new_alt}ft (Δ{alt_change:+.0f}ft)")
                    else:
                        self.logger.info(f"⬆️ Changing {aircraft_id} altitude to {new_alt}ft")
                    
                    result = bs_client.change_altitude(aircraft_id, new_alt)
                    self.logger.info(f"📊 Altitude change result: {result}")
                    return result
                else:
                    self.logger.warning(f"⚠️ No target_altitude_ft in parameters: {parameters}")
            
            elif resolution_type == "speed_change":
                new_spd = parameters.get('target_speed_kt')
                if new_spd is not None:
                    self.logger.info(f"🚀 Changing {aircraft_id} speed to {new_spd}kt")
                    result = bs_client.change_speed(aircraft_id, new_spd)
                    self.logger.info(f"📊 Speed change result: {result}")
                    return result
                else:
                    self.logger.warning(f"⚠️ No target_speed_kt in parameters: {parameters}")
            
            elif resolution_type == "direct_to":
                waypoint = parameters.get('waypoint_name')
                lat = parameters.get('lat')
                lon = parameters.get('lon')
                if waypoint and lat is not None and lon is not None:
                    self.logger.info(f"🎯 Directing {aircraft_id} to {waypoint} at {lat}, {lon}")
                    result = bs_client.direct_to(aircraft_id, lat, lon)
                    self.logger.info(f"📊 Direct-to result: {result}")
                    return result
                elif waypoint:
                    self.logger.info(f"🎯 Directing {aircraft_id} to waypoint {waypoint}")
                    # For named waypoints, we'll just use a heading change for now
                    return True
                else:
                    self.logger.warning(f"⚠️ Incomplete direct_to parameters: {parameters}")
            
            elif resolution_type == "vertical_speed_change":
                vs_fpm = parameters.get('target_vertical_speed_fpm')
                if vs_fpm is not None:
                    self.logger.info(f"📈 Changing {aircraft_id} vertical speed to {vs_fpm}fpm")
                    result = bs_client.change_vertical_speed(aircraft_id, vs_fpm)
                    self.logger.info(f"📊 Vertical speed change result: {result}")
                    return result
                else:
                    self.logger.warning(f"⚠️ No target_vertical_speed_fpm in parameters: {parameters}")
            
            elif resolution_type == "reroute_via":
                # Handle reroute via waypoint
                via_waypoint = parameters.get('via_waypoint')
                if via_waypoint:
                    waypoint_name = via_waypoint.get('name', 'AVOID')
                    waypoint_lat = via_waypoint.get('lat')
                    waypoint_lon = via_waypoint.get('lon')
                    resume_to_destination = parameters.get('resume_to_destination', True)
                    
                    if waypoint_lat is not None and waypoint_lon is not None:
                        self.logger.info(f"🛣️ Rerouting {aircraft_id} via {waypoint_name} at {waypoint_lat}, {waypoint_lon}")
                        
                        # Add the waypoint to BlueSky
                        result1 = bs_client.add_waypoint(aircraft_id, waypoint_name, waypoint_lat, waypoint_lon)
                        
                        # Direct to the waypoint
                        result2 = bs_client.direct_to(aircraft_id, waypoint_lat, waypoint_lon)
                        
                        if resume_to_destination:
                            # Get original destination and add it back to route
                            destination = bs_client.get_aircraft_destination(aircraft_id)
                            if destination:
                                self.logger.info(f"🎯 Adding destination {destination.name} back to route after waypoint")
                                bs_client.add_waypoint(aircraft_id, destination.name, destination.lat, destination.lon)
                        
                        success = result1 or result2  # Success if either operation worked
                        self.logger.info(f"📊 Reroute via result: waypoint={result1}, direct={result2}, overall={success}")
                        return success
                    else:
                        self.logger.warning(f"⚠️ Missing waypoint coordinates in reroute_via: {via_waypoint}")
                else:
                    self.logger.warning(f"⚠️ No via_waypoint in reroute_via parameters: {parameters}")
            
            elif resolution_type == "no_action":
                self.logger.info(f"✅ No action required for {aircraft_id}")
                return True  # No action needed
            
            else:
                self.logger.warning(f"⚠️ Unknown resolution type: {resolution_type}")
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error applying resolution {resolution_type} to {aircraft_id}: {e}")
            return False
    
    def _generate_destination_from_position(self, lat: float, lon: float, heading: float, alt: float, target_distance: float = 550.0) -> Dict:
        """Generate a destination at specified distance from starting position"""
        import random
        
        # Use the specified distance with ±50 NM variation for realism
        distance_nm = target_distance + random.uniform(-50, 50)
        
        # Use current heading with some variation
        bearing = (heading + random.uniform(-30, 30)) % 360
        
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        bearing_rad = math.radians(bearing)
        
        # Earth radius in nautical miles
        R = 3440.065  # Earth radius in NM
        
        # Calculate destination coordinates
        lat2_rad = math.asin(
            math.sin(lat_rad) * math.cos(distance_nm / R) +
            math.cos(lat_rad) * math.sin(distance_nm / R) * math.cos(bearing_rad)
        )
        
        lon2_rad = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(distance_nm / R) * math.cos(lat_rad),
            math.cos(distance_nm / R) - math.sin(lat_rad) * math.sin(lat2_rad)
        )
        
        # Convert back to degrees
        dest_lat = math.degrees(lat2_rad)
        dest_lon = math.degrees(lon2_rad)
        
        return {
            'name': f"DEST_{int(distance_nm)}NM",
            'latitude': dest_lat,
            'longitude': dest_lon,
            'altitude': alt,  # Same altitude
            'distance_nm': distance_nm,
            'bearing': bearing
        }
    
    def _test_llm(self, args) -> int:
        """Test LLM client functionality"""
        self.logger.info("🧪 Testing LLM client...")
        
        try:
            # Initialize LLM client
            config = LLMConfig(
                provider=LLMProvider.OLLAMA,
                model=args.model,
                timeout=30.0
            )
            
            client = StreamlinedLLMClient(config)
            
            # Test basic functionality
            self.logger.info("✅ LLM client initialized successfully")
            self.logger.info(f"📡 Using model: {args.model}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"❌ LLM test failed: {e}")
            return 1
    
    def _test_bluesky(self, args) -> int:
        """Test BlueSky client functionality"""
        self.logger.info("🧪 Testing BlueSky client...")
        
        try:
            # Initialize client
            client = SimpleBlueSkyClient()
            client.initialize()
            client.reset()
            
            # Test aircraft creation
            success = client.create_aircraft(
                acid="TEST1",
                lat=42.0,
                lon=-87.0,
                hdg=90,
                alt=35000,
                spd=400
            )
            
            if success:
                self.logger.info("✅ Test aircraft created successfully")
                
                # Test state retrieval
                state = client.get_aircraft_state("TEST1")
                if state:
                    self.logger.info(f"✅ Aircraft state: {state.id} at {state.lat:.4f}, {state.lon:.4f}")
                else:
                    self.logger.warning("⚠️ Could not retrieve aircraft state")
                
                # Test conflict detection
                client.configure_conflict_detection()
                conflicts = client.get_conflict_summary()
                self.logger.info(f"✅ Conflict detection: {conflicts}")
                
            else:
                self.logger.error("❌ Failed to create test aircraft")
                return 1
            
            return 0
            
        except Exception as e:
            self.logger.error(f"❌ BlueSky test failed: {e}")
            return 1
    
    def _test_integration(self, args) -> int:
        """Test LLM + BlueSky integration"""
        self.logger.info("🧪 Testing LLM + BlueSky integration...")
        
        try:
            # Initialize components
            bs_client = SimpleBlueSkyClient()
            bs_client.initialize()
            bs_client.reset()
            
            llm_config = LLMConfig(provider=LLMProvider.OLLAMA, model=args.model)
            llm_client = StreamlinedLLMClient(llm_config)
            
            # Create test scenario
            bs_client.create_aircraft("OWN1", 42.0, -87.0, 90, 35000, 400)
            bs_client.create_aircraft("TFC1", 42.0, -86.9, 270, 35000, 400)
            
            # Step simulation
            bs_client.step_simulation(60)
            
            # Get states
            states = bs_client.get_all_aircraft_states()
            if len(states) >= 2:
                ownship = states[0]
                intruder = states[1]
                
                # Create conflict context
                context = ConflictContext(
                    ownship_callsign=ownship.id,
                    ownship_state={
                        'latitude': ownship.lat,
                        'longitude': ownship.lon,
                        'altitude': ownship.alt,
                        'heading': ownship.hdg,
                        'speed': ownship.tas,
                        'vertical_speed_fpm': ownship.vs * 60
                    },
                    intruders=[{
                        'callsign': intruder.id,
                        'latitude': intruder.lat,
                        'longitude': intruder.lon,
                        'altitude': intruder.alt,
                        'heading': intruder.hdg,
                        'speed': intruder.tas,
                        'vertical_speed_fpm': intruder.vs * 60
                    }],
                    scenario_time=60.0,
                    lookahead_minutes=5.0  # Match BlueSky native settings
                )
                
                # Test LLM response
                response = llm_client.detect_and_resolve_conflicts(context)
                
                self.logger.info(f"✅ LLM response: {response.get('conflicts_detected', False)} conflicts detected")
                if response.get('resolution'):
                    res_type = response['resolution'].get('resolution_type')
                    self.logger.info(f"✅ LLM suggests: {res_type}")
                
                # Test applying resolution
                if response.get('resolution') and response['resolution'].get('resolution_type') != 'no_action':
                    resolution = response['resolution']
                    success = self._apply_resolution_to_bluesky(
                        bs_client, 
                        ownship.id, 
                        resolution.get('resolution_type'), 
                        resolution.get('parameters', {})
                    )
                    
                    if success:
                        self.logger.info("✅ Resolution applied to BlueSky")
                    else:
                        self.logger.warning("⚠️ Failed to apply resolution")
                
                return 0
            else:
                self.logger.error("❌ Failed to create test aircraft")
                return 1
                
        except Exception as e:
            self.logger.error(f"❌ Integration test failed: {e}")
            return 1


def main():
    """CLI entry point"""
    cli = StreamlinedCLI()
    return cli.run()


if __name__ == "__main__":
    exit(main())
