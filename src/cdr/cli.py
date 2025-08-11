#!/usr/bin/env python3
"""CLI interface for LLM_ATC7"""

import sys
import argparse
import logging
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


class LLMATCCLIConsole:
    """Main CLI interface for LLM_ATC7"""
    
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
        """Create main argument parser"""
        parser = argparse.ArgumentParser(
            prog='atc-llm',
            description='LLM-driven Air Traffic Control Conflict Detection and Resolution',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  atc-llm run-e2e --scat-path data/sample_scat.json --visualize
  atc-llm batch --scat-dir data/SCAT_extracted --output-dir results/
  atc-llm health-check
  atc-llm metrics --results-dir results/latest/
            """
        )
        
        # Global options
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Enable verbose logging')
        parser.add_argument('--config-file', '-c', type=Path,
                          help='Configuration file path')
        
        # Create subparsers
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Add subcommands
        self._add_run_e2e_parser(subparsers)
        self._add_batch_parser(subparsers)
        self._add_health_check_parser(subparsers)
        self._add_metrics_parser(subparsers)
        self._add_visualize_parser(subparsers)
        
        return parser
    
    def _add_run_e2e_parser(self, subparsers):
        """Add run-e2e subcommand parser"""
        parser = subparsers.add_parser(
            'run-e2e',
            help='Run end-to-end conflict detection and resolution simulation'
        )
        
        parser.add_argument('--scat-path', required=True, type=Path,
                          help='Path to SCAT data file (JSON format)')
        parser.add_argument('--output-dir', type=Path, default='output/',
                          help='Output directory for results')
        parser.add_argument('--visualize', action='store_true',
                          help='Enable real-time visualization')
        parser.add_argument('--llm-model', default='llama3.1:8b',
                          help='LLM model to use')
        parser.add_argument('--seed', type=int, default=42,
                          help='Random seed for reproducibility')
        
        parser.set_defaults(func=self._run_e2e)
    
    def _add_batch_parser(self, subparsers):
        """Add batch subcommand parser"""
        parser = subparsers.add_parser(
            'batch',
            help='Run batch processing on multiple SCAT scenarios'
        )
        
        parser.add_argument('--scat-dir', required=True, type=Path,
                          help='Directory containing SCAT JSON files')
        parser.add_argument('--output-dir', type=Path, default='output/batch/',
                          help='Output directory for batch results')
        parser.add_argument('--max-scenarios', type=int,
                          help='Maximum number of scenarios to process')
        
        parser.set_defaults(func=self._run_batch)
    
    def _add_health_check_parser(self, subparsers):
        """Add health-check subcommand parser"""
        parser = subparsers.add_parser(
            'health-check',
            help='Verify system components are working correctly'
        )
        
        parser.add_argument('--quick', action='store_true',
                          help='Run only quick checks')
        
        parser.set_defaults(func=self._run_health_check)
    
    def _add_metrics_parser(self, subparsers):
        """Add metrics subcommand parser"""
        parser = subparsers.add_parser(
            'metrics',
            help='Analyze and display simulation metrics'
        )
        
        parser.add_argument('--results-dir', type=Path, default='output/',
                          help='Results directory to analyze')
        parser.add_argument('--format', choices=['json', 'csv', 'table'],
                          default='table', help='Output format')
        
        parser.set_defaults(func=self._run_metrics)
    
    def _add_visualize_parser(self, subparsers):
        """Add visualize subcommand parser"""
        parser = subparsers.add_parser(
            'visualize',
            help='Visualize simulation results'
        )
        
        parser.add_argument('--trajectory-file', required=True, type=Path,
                          help='Trajectory file to visualize')
        parser.add_argument('--interactive', action='store_true',
                          help='Show interactive plot')
        
        parser.set_defaults(func=self._run_visualize)
    
    def _run_e2e(self, args) -> int:
        """Run end-to-end simulation command"""
        self.logger.info("Starting end-to-end CDR simulation")
        
        try:
            from .adapters.scat_adapter import SCATAdapter
            from .simulation.bluesky_client import BlueSkyClient, BlueSkyConfig
            from .pipeline.cdr_pipeline import CDRPipeline, PipelineConfig
            from .ai.llm_client import LLMClient
            from .ai.memory import ExperienceMemory
            import json
            import time
            
            # Validate inputs
            if not args.scat_path.exists():
                self.logger.error(f"SCAT file not found: {args.scat_path}")
                return 1
            
            # Load SCAT data
            self.logger.info(f"Loading SCAT data from {args.scat_path}")
            scat_adapter = SCATAdapter(args.scat_path)
            scat_adapter.load_file()
            
            # Setup output directory
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Running simulation with output to: {output_dir}")
            
            # Initialize components
            self.logger.info("Initializing simulation components...")
            
            # Initialize LLM client with enhanced settings
            from .ai.llm_client import LLMConfig, LLMProvider
            llm_config = LLMConfig(
                provider=LLMProvider.OLLAMA,
                model_name=args.llm_model,
                base_url="http://localhost:11434",
                enable_verifier=True,
                enable_agree_on_two=True,
                enable_reprompt_on_failure=True,
                temperature=0.2,  # Lower temperature for more consistent outputs
                seed=args.seed
            )
            llm_client = LLMClient(llm_config)
            
            # Initialize memory store with CPU device to avoid GPU contention
            memory_store = ExperienceMemory(device="cpu")
            
            # Initialize BlueSky client (will use mock if connection fails)
            bluesky_config = BlueSkyConfig()
            bluesky_client = BlueSkyClient(bluesky_config)
            
            # Initialize CDR pipeline
            pipeline_config = PipelineConfig(
                cycle_interval_seconds=60.0,
                lookahead_minutes=10.0,
                max_simulation_time_minutes=60.0,  # Shorter for testing
                llm_enabled=True,
                memory_enabled=True,
                save_trajectories=True
            )
            
            cdr_pipeline = CDRPipeline(
                config=pipeline_config,
                bluesky_client=bluesky_client,
                llm_client=llm_client,
                memory_store=memory_store
            )
            
            # Run simulation
            self.logger.info("Starting CDR simulation execution...")
            start_time = time.time()
            
            # Execute the pipeline
            simulation_result = cdr_pipeline.run_scenario(scat_adapter._data, output_dir)
            
            execution_time = time.time() - start_time
            
            # Save results
            result_file = output_dir / "simulation_result.json"
            
            # Convert args to JSON-serializable format
            cli_args = {}
            for key, value in vars(args).items():
                if hasattr(value, '__fspath__'):  # It's a Path object
                    cli_args[key] = str(value)
                else:
                    cli_args[key] = value
            
            with open(result_file, 'w') as f:
                json.dump({
                    **simulation_result.to_dict(),
                    'execution_time_seconds': execution_time,
                    'cli_args': cli_args
                }, f, indent=2)
            
            self.logger.info(f"✓ Simulation completed in {execution_time:.2f} seconds")
            self.logger.info(f"✓ Results saved to: {result_file}")
            
            # Print summary
            total_conflicts = simulation_result.total_conflicts
            resolved_conflicts = simulation_result.successful_resolutions
            self.logger.info(f"Summary: {resolved_conflicts}/{total_conflicts} conflicts resolved")
            
            # Run visualization if requested
            if args.visualize:
                self.logger.info("Running visualization...")
                try:
                    from .visualization.visualization import visualize_trajectory_file
                    
                    # Look for trajectory file in output directory
                    trajectory_file = output_dir / "trajectories.jsonl"
                    if trajectory_file.exists():
                        visualize_trajectory_file(trajectory_file, interactive=False)
                        self.logger.info("✓ Visualization completed")
                    else:
                        self.logger.warning("No trajectory file found for visualization")
                        
                except ImportError:
                    self.logger.warning("Visualization not available (install matplotlib)")
                except Exception as e:
                    self.logger.warning(f"Visualization failed: {e}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"E2E simulation failed: {e}")
            return 1
    
    def _run_batch(self, args) -> int:
        """Run batch processing command"""
        self.logger.info("Starting batch processing")
        
        if not args.scat_dir.exists():
            self.logger.error(f"SCAT directory not found: {args.scat_dir}")
            return 1
        
        # Find SCAT files
        scat_files = list(args.scat_dir.glob('*.json'))
        if args.max_scenarios:
            scat_files = scat_files[:args.max_scenarios]
        
        self.logger.info(f"Found {len(scat_files)} SCAT files to process")
        
        # Setup output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Processing batch with output to: {output_dir}")
        
        # Initialize components once for batch processing
        try:
            from .adapters.scat_adapter import SCATAdapter
            from .simulation.bluesky_client import BlueSkyClient, BlueSkyConfig
            from .pipeline.cdr_pipeline import CDRPipeline, PipelineConfig
            from .ai.llm_client import LLMClient
            from .ai.memory import ExperienceMemory
            import json
            import time
            
            # Initialize shared components with enhanced settings
            from .ai.llm_client import LLMConfig, LLMProvider
            llm_config = LLMConfig(
                provider=LLMProvider.OLLAMA,
                model_name="llama3.1:8b",
                base_url="http://localhost:11434",
                enable_verifier=True,
                enable_agree_on_two=True,
                enable_reprompt_on_failure=True,
                temperature=0.2,
                seed=42
            )
            llm_client = LLMClient(llm_config)
            memory_store = ExperienceMemory(device="cpu")
            bluesky_config = BlueSkyConfig()
            bluesky_client = BlueSkyClient(bluesky_config)
            
            batch_results = []
            successful_runs = 0
            
            # Process each SCAT file
            for i, scat_file in enumerate(scat_files, 1):
                self.logger.info(f"Processing scenario {i}/{len(scat_files)}: {scat_file.name}")
                
                try:
                    # Load SCAT data
                    scat_adapter = SCATAdapter(scat_file)
                    scat_adapter.load_file()
                    
                    # Setup scenario output directory
                    scenario_output = output_dir / f"scenario_{i:03d}_{scat_file.stem}"
                    scenario_output.mkdir(exist_ok=True)
                    
                    # Create pipeline for this scenario
                    pipeline_config = PipelineConfig(
                        cycle_interval_seconds=60.0,
                        lookahead_minutes=10.0,
                        max_simulation_time_minutes=30.0,  # Shorter for batch
                        llm_enabled=True,
                        memory_enabled=True,
                        save_trajectories=True
                    )
                    
                    cdr_pipeline = CDRPipeline(
                        config=pipeline_config,
                        bluesky_client=bluesky_client,
                        llm_client=llm_client,
                        memory_store=memory_store
                    )
                    
                    # Run simulation
                    start_time = time.time()
                    simulation_result = cdr_pipeline.run_scenario(scat_adapter._data, scenario_output)
                    execution_time = time.time() - start_time
                    
                    # Save individual result
                    result_file = scenario_output / "simulation_result.json"
                    
                    # Convert paths to strings for JSON serialization
                    with open(result_file, 'w') as f:
                        json.dump({
                            **simulation_result.to_dict(),
                            'execution_time_seconds': execution_time,
                            'scenario_file': str(scat_file),
                            'scenario_index': i
                        }, f, indent=2)
                    
                    batch_results.append({
                        'scenario_file': scat_file.name,
                        'success': True,
                        'execution_time': execution_time,
                        'total_conflicts': simulation_result.total_conflicts,
                        'resolved_conflicts': simulation_result.successful_resolutions
                    })
                    
                    successful_runs += 1
                    self.logger.info(f"✓ Scenario {i} completed in {execution_time:.2f}s")
                    
                except Exception as e:
                    self.logger.error(f"✗ Scenario {i} failed: {e}")
                    batch_results.append({
                        'scenario_file': scat_file.name,
                        'success': False,
                        'error': str(e)
                    })
            
            # Save batch summary
            batch_summary = {
                'total_scenarios': len(scat_files),
                'successful_runs': successful_runs,
                'success_rate': successful_runs / len(scat_files) if scat_files else 0,
                'results': batch_results
            }
            
            summary_file = output_dir / "batch_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(batch_summary, f, indent=2)
            
            self.logger.info(f"✓ Batch processing complete: {successful_runs}/{len(scat_files)} scenarios successful")
            self.logger.info(f"✓ Summary saved to: {summary_file}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return 1
    
    def _run_health_check(self, args) -> int:
        """Run health check command"""
        self.logger.info("Running system health check")
        
        checks_passed = 0
        total_checks = 0
        
        # Check Python environment
        total_checks += 1
        try:
            python_version = sys.version_info
            if python_version >= (3, 8):
                self.logger.info("✓ Python version check passed")
                checks_passed += 1
            else:
                self.logger.error("✗ Python version too old (requires 3.8+)")
        except Exception as e:
            self.logger.error(f"✗ Python environment check failed: {e}")
        
        # Check required packages
        total_checks += 1
        try:
            import numpy, scipy, pandas, matplotlib, requests
            self.logger.info("✓ Required packages available")
            checks_passed += 1
        except ImportError as e:
            self.logger.error(f"✗ Missing required package: {e}")
        
        # Check project structure
        total_checks += 1
        try:
            from .adapters import scat_adapter
            from .simulation import bluesky_client
            from .ai import llm_client
            self.logger.info("✓ Project structure check passed")
            checks_passed += 1
        except ImportError as e:
            self.logger.error(f"✗ Project structure check failed: {e}")
        
        # Summary
        self.logger.info(f"Health Check Summary: {checks_passed}/{total_checks} passed")
        self.logger.info(f"Status: {'HEALTHY' if checks_passed == total_checks else 'ISSUES DETECTED'}")
        
        return 0 if checks_passed == total_checks else 1
    
    def _run_metrics(self, args) -> int:
        """Run metrics analysis command"""
        self.logger.info("Analyzing simulation metrics")
        
        if not args.results_dir.exists():
            self.logger.error(f"Results directory not found: {args.results_dir}")
            return 1
        
        try:
            from .metrics.calculator import MetricsCalculator
            import json
            from pathlib import Path
            
            # Find result files
            result_files = list(args.results_dir.rglob("simulation_result.json"))
            
            if not result_files:
                self.logger.error(f"No simulation result files found in {args.results_dir}")
                return 1
            
            self.logger.info(f"Found {len(result_files)} result files to analyze")
            
            # Initialize metrics calculator
            metrics_calc = MetricsCalculator()
            
            # Calculate aggregated metrics
            aggregated_metrics = metrics_calc.aggregate_metrics(result_files)
            
            # Output results based on format
            if args.format == 'json':
                print(json.dumps(aggregated_metrics, indent=2))
            elif args.format == 'csv':
                self._output_metrics_csv(aggregated_metrics)
            else:  # table format
                self._output_metrics_table(aggregated_metrics)
            
            # Save metrics to file
            metrics_file = args.results_dir / f"metrics_analysis.{args.format}"
            if args.format == 'json':
                with open(metrics_file, 'w') as f:
                    json.dump(aggregated_metrics, f, indent=2)
            
            self.logger.info(f"✓ Metrics analysis complete")
            self.logger.info(f"✓ Results saved to: {metrics_file}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Metrics analysis failed: {e}")
            return 1
    
    def _output_metrics_table(self, metrics: dict):
        """Output metrics in table format"""
        print("\n" + "="*60)
        print("SIMULATION METRICS SUMMARY")
        print("="*60)
        
        print(f"Total Scenarios: {metrics.get('total_scenarios', 0)}")
        print(f"Successful Runs: {metrics.get('successful_runs', 0)}")
        print(f"Success Rate: {metrics.get('success_rate', 0):.1%}")
        
        if 'safety_metrics' in metrics:
            safety = metrics['safety_metrics']
            print(f"\nSAFETY METRICS:")
            print(f"  Total Conflicts: {safety.get('total_conflicts', 0)}")
            print(f"  Resolved Conflicts: {safety.get('total_resolved', 0)}")
            print(f"  Safety Violations: {safety.get('total_violations', 0)}")
            print(f"  TBAS Score: {safety.get('avg_tbas_score', 0):.3f}")
            print(f"  LAT Score: {safety.get('avg_lat_score', 0):.3f}")
            print(f"  Resolution Success Rate: {safety.get('resolution_success_rate', 0):.1%}")
        
        if 'performance_metrics' in metrics:
            perf = metrics['performance_metrics']
            print(f"\nPERFORMANCE METRICS:")
            print(f"  Avg Resolution Time: {perf.get('avg_resolution_time_s', 0):.1f}s")
            print(f"  Avg Additional Distance: {perf.get('avg_additional_distance_nm', 0):.1f} nm")
            print(f"  Avg Additional Time: {perf.get('avg_additional_time_min', 0):.1f} min")
            print(f"  LLM Success Rate: {perf.get('overall_llm_success_rate', 0):.1%}")
            print(f"  Fallback Rate: {perf.get('overall_fallback_rate', 0):.1%}")
        
        print("="*60)
    
    def _output_metrics_csv(self, metrics: dict):
        """Output metrics in CSV format"""
        # Basic CSV output for metrics
        print("metric,value")
        print(f"total_scenarios,{metrics.get('total_scenarios', 0)}")
        print(f"successful_runs,{metrics.get('successful_runs', 0)}")
        print(f"success_rate,{metrics.get('success_rate', 0)}")
        
        if 'safety_metrics' in metrics:
            safety = metrics['safety_metrics']
            for key, value in safety.items():
                print(f"safety_{key},{value}")
        
        if 'performance_metrics' in metrics:
            perf = metrics['performance_metrics']
            for key, value in perf.items():
                print(f"performance_{key},{value}")
    
    def _run_visualize(self, args) -> int:
        """Run visualization command"""
        self.logger.info("Creating visualization")
        
        if not args.trajectory_file.exists():
            self.logger.error(f"Trajectory file not found: {args.trajectory_file}")
            return 1
        
        try:
            # Import visualization module
            from .visualization.visualization import visualize_trajectory_file
            
            # Create visualization
            success = visualize_trajectory_file(
                trajectory_file=args.trajectory_file,
                interactive=args.interactive
            )
            
            if success:
                self.logger.info(f"✓ Visualization completed for: {args.trajectory_file}")
                return 0
            else:
                self.logger.error("✗ Visualization failed")
                return 1
                
        except ImportError as e:
            self.logger.error(f"Visualization module not available: {e}")
            self.logger.info("Install matplotlib for visualization: pip install matplotlib")
            return 1
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            return 1


def main():
    """Console script entry point"""
    cli = LLMATCCLIConsole()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
