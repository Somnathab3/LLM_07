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
            
            self.logger.info(f"Simulation would run with output to: {output_dir}")
            self.logger.info("✓ End-to-end simulation setup complete")
            
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
        self.logger.info(f"Would process batch with output to: {args.output_dir}")
        
        return 0
    
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
        
        self.logger.info(f"Would analyze metrics in: {args.results_dir}")
        self.logger.info(f"Output format: {args.format}")
        
        return 0
    
    def _run_visualize(self, args) -> int:
        """Run visualization command"""
        self.logger.info("Creating visualization")
        
        if not args.trajectory_file.exists():
            self.logger.error(f"Trajectory file not found: {args.trajectory_file}")
            return 1
        
        self.logger.info(f"Would visualize: {args.trajectory_file}")
        self.logger.info(f"Interactive: {args.interactive}")
        
        return 0


def main():
    """Console script entry point"""
    cli = LLMATCCLIConsole()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
