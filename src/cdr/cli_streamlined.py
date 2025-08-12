#!/usr/bin/env python3
"""Streamlined CLI interface for LLM_ATC7"""

import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Optional, List


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
        parser.add_argument('--max-time', type=float, default=120.0,
                          help='Maximum simulation time in minutes')
        
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
            # Import required components
            from .simulation.bluesky_client import SimpleBlueSkyClient
            from .ai.llm_client import StreamlinedLLMClient, LLMConfig, LLMProvider, ConflictContext
            
            # Validate inputs
            if not args.scat_path.exists():
                self.logger.error(f"SCAT file not found: {args.scat_path}")
                return 1
            
            # Load SCAT data
            self.logger.info(f"📂 Loading SCAT data from {args.scat_path}")
            with open(args.scat_path, 'r') as f:
                scat_data = json.load(f)
            
            if not isinstance(scat_data, list) or len(scat_data) == 0:
                self.logger.error("Invalid SCAT data format")
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
            
            # Run simulation
            self.logger.info("🛫 Starting simulation...")
            result = self._execute_simulation(scat_data, bs_client, llm_client, output_dir, args.max_time)
            
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
    
    def _execute_simulation(self, scat_data: List[Dict], bs_client, llm_client, output_dir: Path, max_time: float) -> Dict:
        """Execute the actual simulation"""
        start_time = time.time()
        conflicts_detected = 0
        resolutions_applied = 0
        trajectory_log = []
        conflict_log = []
        
        try:
            # Create aircraft from SCAT data
            aircraft_created = 0
            for entry in scat_data[:5]:  # Limit to 5 aircraft for testing
                callsign = entry.get('ACID', f'TFC{aircraft_created+1}')
                lat = entry.get('Latitude', 42.0)
                lon = entry.get('Longitude', -87.0)
                alt = entry.get('Altitude', 35000)
                hdg = entry.get('Course', 90)
                spd = entry.get('Speed', 400)
                
                success = bs_client.create_aircraft(
                    acid=callsign,
                    lat=lat,
                    lon=lon,
                    hdg=hdg,
                    alt=alt,
                    spd=spd
                )
                
                if success:
                    aircraft_created += 1
                    self.logger.info(f"✈️ Created aircraft {callsign}")
            
            self.logger.info(f"📋 Created {aircraft_created} aircraft")
            
            # Simulation loop
            sim_time = 0.0
            cycle_interval = 300.0  # 5 minutes
            
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
                
                # Check for conflicts using both methods
                native_conflicts = bs_client.get_conflict_summary()
                
                # If conflicts detected, use LLM for resolution
                if native_conflicts['active_conflicts'] > 0:
                    conflicts_detected += 1
                    self.logger.info(f"🚨 Conflict detected at {sim_time/60:.1f} min")
                    
                    # Create conflict context for LLM
                    ownship_state = aircraft_states[0]  # Use first aircraft as ownship
                    intruders = []
                    
                    for state in aircraft_states[1:]:
                        intruders.append({
                            'callsign': state.id,
                            'latitude': state.lat,
                            'longitude': state.lon,
                            'altitude': state.alt,
                            'heading': state.hdg,
                            'speed': state.tas,
                            'vertical_speed_fpm': state.vs * 60  # Convert m/s to fpm
                        })
                    
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
                        lookahead_minutes=10.0
                    )
                    
                    # Get LLM resolution
                    llm_response = llm_client.detect_and_resolve_conflicts(context)
                    
                    # Apply resolution if suggested
                    if llm_response.get('conflicts_detected') and llm_response.get('resolution'):
                        resolution = llm_response['resolution']
                        resolution_type = resolution.get('resolution_type')
                        parameters = resolution.get('parameters', {})
                        
                        self.logger.info(f"🤖 LLM suggests: {resolution_type} - {resolution.get('reasoning', '')}")
                        
                        # Apply resolution to BlueSky
                        success = self._apply_resolution_to_bluesky(
                            bs_client, 
                            ownship_state.id, 
                            resolution_type, 
                            parameters
                        )
                        
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
                
                # Progress update
                if int(sim_time) % 600 == 0:  # Every 10 minutes
                    self.logger.info(f"⏰ Simulation time: {sim_time/60:.1f} min")
                
                # Limit real-time execution
                cycle_elapsed = time.time() - cycle_start
                if cycle_elapsed < 1.0:  # Minimum 1 second per cycle
                    time.sleep(1.0 - cycle_elapsed)
            
            # Save results
            results = {
                'success': True,
                'simulation_time_minutes': sim_time / 60,
                'real_time_seconds': time.time() - start_time,
                'aircraft_created': aircraft_created,
                'conflicts_detected': conflicts_detected,
                'resolutions_applied': resolutions_applied,
                'trajectory_log': trajectory_log,
                'conflict_log': conflict_log
            }
            
            # Save to files
            with open(output_dir / 'simulation_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            with open(output_dir / 'trajectories.jsonl', 'w') as f:
                for entry in trajectory_log:
                    f.write(json.dumps(entry) + '\n')
            
            if conflict_log:
                with open(output_dir / 'conflicts.jsonl', 'w') as f:
                    for entry in conflict_log:
                        f.write(json.dumps(entry) + '\n')
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'conflicts_detected': conflicts_detected,
                'resolutions_applied': resolutions_applied
            }
    
    def _apply_resolution_to_bluesky(self, bs_client, aircraft_id: str, resolution_type: str, parameters: Dict) -> bool:
        """Apply LLM resolution to BlueSky simulation"""
        try:
            if resolution_type == "heading_change":
                new_hdg = parameters.get('new_heading_deg')
                if new_hdg is not None:
                    cmd = f"{aircraft_id} HDG {int(new_hdg)}"
                    return bs_client.stack.stack(cmd)
            
            elif resolution_type == "altitude_change":
                new_alt = parameters.get('target_altitude_ft')
                if new_alt is not None:
                    cmd = f"{aircraft_id} ALT {int(new_alt)}"
                    return bs_client.stack.stack(cmd)
            
            elif resolution_type == "speed_change":
                new_spd = parameters.get('target_speed_kt')
                if new_spd is not None:
                    cmd = f"{aircraft_id} SPD {int(new_spd)}"
                    return bs_client.stack.stack(cmd)
            
            elif resolution_type == "direct_to":
                waypoint = parameters.get('waypoint_name')
                lat = parameters.get('lat')
                lon = parameters.get('lon')
                if waypoint and lat is not None and lon is not None:
                    return bs_client.direct_to(aircraft_id, lat, lon)
                elif waypoint:
                    cmd = f"{aircraft_id} DIRECT {waypoint}"
                    return bs_client.stack.stack(cmd)
            
            elif resolution_type == "vertical_speed_change":
                vs_fpm = parameters.get('target_vertical_speed_fpm')
                if vs_fpm is not None:
                    cmd = f"{aircraft_id} VS {int(vs_fpm)}"
                    return bs_client.stack.stack(cmd)
            
            elif resolution_type == "no_action":
                return True  # No action needed
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error applying resolution: {e}")
            return False
    
    def _test_llm(self, args) -> int:
        """Test LLM client functionality"""
        self.logger.info("🧪 Testing LLM client...")
        
        try:
            from .ai.llm_client import StreamlinedLLMClient, LLMConfig, LLMProvider
            
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
            from .simulation.bluesky_client import SimpleBlueSkyClient
            
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
            from .simulation.bluesky_client import SimpleBlueSkyClient
            from .ai.llm_client import StreamlinedLLMClient, LLMConfig, LLMProvider, ConflictContext
            
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
                    lookahead_minutes=10.0
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
