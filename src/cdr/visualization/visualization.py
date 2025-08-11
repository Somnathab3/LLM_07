"""Visualization components for ATC-LLM pipeline."""

import logging
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from .config import Config, VisualizationConfig, DEFAULT_CONFIG
from .models import Scenario, TrackPoint, Aircraft, Position, create_track_point_from_dict


class Visualizer:
    """Visualization component for ATC scenarios and results."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize visualizer with configuration."""
        self.config = config or DEFAULT_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Available backends
        self.matplotlib_available = MATPLOTLIB_AVAILABLE
        self.pygame_available = PYGAME_AVAILABLE
        
        if not (MATPLOTLIB_AVAILABLE or PYGAME_AVAILABLE):
            self.logger.warning("No visualization backends available (matplotlib or pygame)")
    
    def visualize_trajectory_file(self, trajectory_file: Path, interactive: bool = False) -> bool:
        """Visualize trajectory from JSONL file."""
        if not trajectory_file.exists():
            self.logger.error(f"Trajectory file not found: {trajectory_file}")
            return False
        
        try:
            # Load trajectory data
            trajectory_data = self._load_trajectory_from_jsonl(trajectory_file)
            
            if not trajectory_data:
                self.logger.error("No trajectory data found")
                return False
            
            # Use matplotlib for visualization if available
            if MATPLOTLIB_AVAILABLE:
                return self._visualize_with_matplotlib(trajectory_data, interactive)
            else:
                self.logger.error("Matplotlib not available for visualization")
                return False
                
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            return False
    
    def _load_trajectory_from_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load trajectory data from JSONL file."""
        trajectory_data = []
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            trajectory_data.append(data)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Invalid JSON on line {line_num}: {e}")
                            continue
            
            self.logger.info(f"Loaded {len(trajectory_data)} trajectory points")
            return trajectory_data
            
        except Exception as e:
            self.logger.error(f"Failed to load trajectory file: {e}")
            return []
    
    def _visualize_with_matplotlib(self, trajectory_data: List[Dict[str, Any]], interactive: bool) -> bool:
        """Visualize using matplotlib."""
        try:
            # Extract aircraft positions over time
            aircraft_data = self._group_by_aircraft(trajectory_data)
            
            if not aircraft_data:
                self.logger.error("No aircraft data to visualize")
                return False
            
            # Create figure and axis
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.set_title("Aircraft Trajectory Visualization")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True, alpha=0.3)
            
            # Plot trajectories for each aircraft
            for callsign, positions in aircraft_data.items():
                if len(positions) < 2:
                    continue
                
                lats = [pos['latitude'] for pos in positions]
                lons = [pos['longitude'] for pos in positions]
                
                # Plot trajectory line
                ax.plot(lons, lats, '-', label=f"{callsign} trajectory", alpha=0.7)
                
                # Mark start and end positions
                ax.plot(lons[0], lats[0], 'go', markersize=8, label=f"{callsign} start")
                ax.plot(lons[-1], lats[-1], 'ro', markersize=8, label=f"{callsign} end")
                
                # Add callsign labels
                ax.annotate(callsign, (lons[0], lats[0]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            
            # Set equal aspect ratio and adjust layout
            ax.set_aspect('equal', adjustable='box')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Show plot
            if interactive:
                plt.show()
            else:
                # Save to file
                output_file = trajectory_data[0].get('scenario_id', 'trajectory') + '_visualization.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                self.logger.info(f"Visualization saved to: {output_file}")
                plt.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Matplotlib visualization failed: {e}")
            return False
    
    def _group_by_aircraft(self, trajectory_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group trajectory points by aircraft callsign."""
        aircraft_data = {}
        
        for point in trajectory_data:
            if 'aircraft_states' in point:
                # Handle simulation trajectory format
                for callsign, state in point['aircraft_states'].items():
                    if callsign not in aircraft_data:
                        aircraft_data[callsign] = []
                    
                    aircraft_data[callsign].append({
                        'latitude': state.get('latitude', 0),
                        'longitude': state.get('longitude', 0),
                        'altitude_ft': state.get('altitude_ft', 0),
                        'heading_deg': state.get('heading_deg', 0),
                        'speed_kt': state.get('speed_kt', 0),
                        'timestamp': point.get('timestamp', 0)
                    })
            elif 'callsign' in point:
                # Handle individual aircraft point format
                callsign = point['callsign']
                if callsign not in aircraft_data:
                    aircraft_data[callsign] = []
                
                aircraft_data[callsign].append(point)
        
        return aircraft_data
    
    def visualize_scenario_summary(self, scenario_data: Dict[str, Any], output_file: Optional[Path] = None) -> bool:
        """Create a summary visualization of scenario results."""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("Matplotlib not available for scenario summary")
            return False
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"Scenario Summary: {scenario_data.get('scenario_id', 'Unknown')}", fontsize=16)
            
            # Conflicts over time
            conflicts = scenario_data.get('conflicts', [])
            if conflicts:
                conflict_times = [c.get('detection_time', 0) for c in conflicts]
                ax1.hist(conflict_times, bins=20, alpha=0.7, color='red')
                ax1.set_title("Conflict Detection Times")
                ax1.set_xlabel("Time (minutes)")
                ax1.set_ylabel("Number of Conflicts")
            
            # Resolution types
            resolutions = scenario_data.get('resolutions', [])
            if resolutions:
                resolution_types = [r.get('resolution_type', 'unknown') for r in resolutions]
                resolution_counts = {}
                for rt in resolution_types:
                    resolution_counts[rt] = resolution_counts.get(rt, 0) + 1
                
                ax2.pie(resolution_counts.values(), labels=resolution_counts.keys(), autopct='%1.1f%%')
                ax2.set_title("Resolution Types Distribution")
            
            # Safety metrics
            total_conflicts = scenario_data.get('total_conflicts', 0)
            resolved_conflicts = scenario_data.get('successful_resolutions', 0)
            
            ax3.bar(['Total Conflicts', 'Resolved', 'Unresolved'], 
                   [total_conflicts, resolved_conflicts, total_conflicts - resolved_conflicts],
                   color=['blue', 'green', 'red'])
            ax3.set_title("Conflict Resolution Summary")
            ax3.set_ylabel("Count")
            
            # Execution timeline
            execution_time = scenario_data.get('execution_time_seconds', 0)
            simulation_time = scenario_data.get('final_time_minutes', 0)
            
            ax4.bar(['Execution Time (s)', 'Simulation Time (min)'], 
                   [execution_time, simulation_time],
                   color=['orange', 'purple'])
            ax4.set_title("Timing Summary")
            ax4.set_ylabel("Time")
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                self.logger.info(f"Scenario summary saved to: {output_file}")
            else:
                plt.show()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Scenario summary visualization failed: {e}")
            return False


def create_visualizer(config: Optional[Config] = None) -> Visualizer:
    """Factory function to create visualizer instance."""
    return Visualizer(config)


# Convenience function for CLI usage
def visualize_trajectory_file(trajectory_file: Path, interactive: bool = False) -> bool:
    """Standalone function to visualize trajectory file."""
    visualizer = create_visualizer()
    return visualizer.visualize_trajectory_file(trajectory_file, interactive)
    
    def visualize_scenario(self, scenario: Scenario, tracks: Dict[str, List[TrackPoint]]) -> None:
        """Visualize a complete scenario with tracks."""
        if not self.config.visualization.enable:
            self.logger.info("Visualization disabled")
            return
        
        if not PYGAME_AVAILABLE:
            self.logger.warning("Pygame not available for visualization")
            return
        
        self.logger.info(f"Visualizing scenario: {scenario.scenario_id}")
        
        # Calculate bounds for the scenario
        bounds = self._calculate_bounds(scenario, tracks)
        
        # Animation parameters
        max_time_steps = max(len(track) for track in tracks.values()) if tracks else 100
        time_step = 0
        paused = False
        
        running = True
        while running and time_step < max_time_steps:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_RIGHT and paused:
                        time_step = min(time_step + 1, max_time_steps - 1)
                    elif event.key == pygame.K_LEFT and paused:
                        time_step = max(time_step - 1, 0)
            
            if not paused:
                time_step += 1
            
            # Clear screen
            self.screen.fill((20, 30, 40))  # Dark blue background
            
            # Draw scenario
            self._draw_scenario_frame(scenario, tracks, time_step, bounds)
            
            # Draw UI
            self._draw_ui(scenario, time_step, max_time_steps, paused)
            
            # Update display
            pygame.display.flip()
            self.clock.tick(self.config.visualization.fps)
        
        # Keep window open briefly
        pygame.time.wait(2000)
    
    def visualize_from_file(self, scenario_file: Path) -> None:
        """Visualize scenario from saved file."""
        if not scenario_file.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_file}")
        
        # Load scenario
        with open(scenario_file, 'r') as f:
            scenario_data = json.load(f)
        
        scenario = Scenario.from_dict(scenario_data)
        
        # Load tracks from the same directory
        scenario_dir = scenario_file.parent
        tracks = {}
        
        # Load ownship track
        ownship_track_file = scenario_dir / "ownship_track.jsonl"
        if ownship_track_file.exists():
            tracks[scenario.ownship.callsign] = self._load_track_from_jsonl(ownship_track_file)
        
        # Load intruder tracks
        for intruder in scenario.intruders:
            track_file = scenario_dir / f"track_{intruder.callsign}.jsonl"
            if track_file.exists():
                tracks[intruder.callsign] = self._load_track_from_jsonl(track_file)
        
        self.visualize_scenario(scenario, tracks)
    
    def _load_track_from_jsonl(self, track_file: Path) -> List[TrackPoint]:
        """Load track data from JSONL file."""
        track = []
        with open(track_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    track_point = TrackPoint(
                        timestamp=data['timestamp'],
                        position=Position(**data['position']),
                        velocity=data['velocity']  # We don't need to reconstruct Velocity object for visualization
                    )
                    track.append(track_point)
        return track
    
    def _calculate_bounds(self, scenario: Scenario, tracks: Dict[str, List[TrackPoint]]) -> Dict[str, float]:
        """Calculate geographical bounds for the visualization."""
        lats, lons = [], []
        
        # Add initial positions
        for aircraft in [scenario.ownship] + scenario.neighbors + scenario.intruders:
            lats.append(aircraft.position.latitude)
            lons.append(aircraft.position.longitude)
        
        # Add track positions
        for track in tracks.values():
            for point in track:
                lats.append(point.position.latitude)
                lons.append(point.position.longitude)
        
        if not lats:
            # Default bounds
            return {'min_lat': 55.0, 'max_lat': 65.0, 'min_lon': 10.0, 'max_lon': 20.0}
        
        # Add margin
        lat_margin = (max(lats) - min(lats)) * 0.1
        lon_margin = (max(lons) - min(lons)) * 0.1
        
        return {
            'min_lat': min(lats) - lat_margin,
            'max_lat': max(lats) + lat_margin,
            'min_lon': min(lons) - lon_margin,
            'max_lon': max(lons) + lon_margin
        }
    
    def _draw_scenario_frame(self, scenario: Scenario, tracks: Dict[str, List[TrackPoint]], 
                           time_step: int, bounds: Dict[str, float]) -> None:
        """Draw a single frame of the scenario."""
        # Draw tracks if enabled
        if self.config.visualization.show_tracks:
            self._draw_tracks(tracks, time_step, bounds)
        
        # Draw aircraft at current time step
        self._draw_aircraft_at_time(scenario, tracks, time_step, bounds)
        
        # Draw conflicts if detected
        if self.config.visualization.show_conflicts:
            self._draw_conflicts(scenario, time_step, bounds)
    
    def _draw_tracks(self, tracks: Dict[str, List[TrackPoint]], 
                    time_step: int, bounds: Dict[str, float]) -> None:
        """Draw aircraft tracks up to current time step."""
        for callsign, track in tracks.items():
            if len(track) < 2:
                continue
            
            # Determine color based on aircraft type
            if "INTRUDER" in callsign:
                color = (255, 100, 100)  # Red for intruders
            else:
                color = (100, 255, 100)  # Green for others
            
            # Draw track up to current time
            points = []
            for i in range(min(time_step + 1, len(track))):
                x, y = self._geo_to_screen(track[i].position.latitude, track[i].position.longitude, bounds)
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)
    
    def _draw_aircraft_at_time(self, scenario: Scenario, tracks: Dict[str, List[TrackPoint]], 
                              time_step: int, bounds: Dict[str, float]) -> None:
        """Draw aircraft positions at current time step."""
        all_aircraft = [scenario.ownship] + scenario.neighbors + scenario.intruders
        
        for aircraft in all_aircraft:
            # Get position from track if available, otherwise use initial position
            if aircraft.callsign in tracks and time_step < len(tracks[aircraft.callsign]):
                position = tracks[aircraft.callsign][time_step].position
            else:
                position = aircraft.position
            
            x, y = self._geo_to_screen(position.latitude, position.longitude, bounds)
            
            # Choose color and size based on aircraft type
            if aircraft.callsign == scenario.ownship.callsign:
                color = (0, 255, 255)  # Cyan for ownship
                size = 8
            elif "INTRUDER" in aircraft.callsign:
                color = (255, 0, 0)  # Red for intruders
                size = 6
            else:
                color = (0, 255, 0)  # Green for neighbors
                size = 5
            
            # Draw aircraft symbol
            pygame.draw.circle(self.screen, color, (int(x), int(y)), size)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y)), size, 2)
            
            # Draw callsign label
            label = self.font.render(aircraft.callsign, True, (255, 255, 255))
            self.screen.blit(label, (x + size + 5, y - 10))
            
            # Draw heading indicator
            if aircraft.callsign in tracks and time_step < len(tracks[aircraft.callsign]):
                heading = tracks[aircraft.callsign][time_step].velocity.heading
            else:
                heading = aircraft.velocity.heading
            
            heading_rad = math.radians(heading - 90)  # Convert to screen coordinates
            end_x = x + 20 * math.cos(heading_rad)
            end_y = y + 20 * math.sin(heading_rad)
            pygame.draw.line(self.screen, color, (x, y), (end_x, end_y), 3)
    
    def _draw_conflicts(self, scenario: Scenario, time_step: int, bounds: Dict[str, float]) -> None:
        """Draw conflict indicators."""
        # This is a simplified representation
        # In a real implementation, you would track conflicts over time
        for conflict in scenario.conflicts:
            # Find aircraft positions
            ownship_pos = scenario.ownship.position
            intruder_pos = None
            
            for aircraft in scenario.intruders + scenario.neighbors:
                if aircraft.callsign == conflict.intruder_id:
                    intruder_pos = aircraft.position
                    break
            
            if intruder_pos:
                x1, y1 = self._geo_to_screen(ownship_pos.latitude, ownship_pos.longitude, bounds)
                x2, y2 = self._geo_to_screen(intruder_pos.latitude, intruder_pos.longitude, bounds)
                
                # Draw conflict line
                pygame.draw.line(self.screen, (255, 255, 0), (x1, y1), (x2, y2), 3)
                
                # Draw conflict symbol at midpoint
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                pygame.draw.circle(self.screen, (255, 255, 0), (int(mid_x), int(mid_y)), 10, 3)
    
    def _draw_ui(self, scenario: Scenario, time_step: int, max_time_steps: int, paused: bool) -> None:
        """Draw user interface elements."""
        # Time indicator
        time_text = f"Time: {time_step}/{max_time_steps}"
        time_surface = self.font.render(time_text, True, (255, 255, 255))
        self.screen.blit(time_surface, (10, 10))
        
        # Pause indicator
        if paused:
            pause_surface = self.font.render("PAUSED - Press SPACE to continue", True, (255, 255, 0))
            self.screen.blit(pause_surface, (10, 40))
        
        # Controls
        controls = [
            "SPACE: Pause/Resume",
            "LEFT/RIGHT: Step when paused",
            "ESC: Exit"
        ]
        
        for i, control in enumerate(controls):
            control_surface = self.font.render(control, True, (200, 200, 200))
            self.screen.blit(control_surface, (10, self.config.visualization.window_height - 80 + i * 25))
        
        # Scenario info
        info_lines = [
            f"Scenario: {scenario.scenario_id}",
            f"Ownship: {scenario.ownship.callsign}",
            f"Neighbors: {len(scenario.neighbors)}",
            f"Intruders: {len(scenario.intruders)}",
            f"Conflicts: {len(scenario.conflicts)}"
        ]
        
        for i, line in enumerate(info_lines):
            info_surface = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(info_surface, (self.config.visualization.window_width - 250, 10 + i * 25))
    
    def _geo_to_screen(self, lat: float, lon: float, bounds: Dict[str, float]) -> Tuple[float, float]:
        """Convert geographical coordinates to screen coordinates."""
        lat_range = bounds['max_lat'] - bounds['min_lat']
        lon_range = bounds['max_lon'] - bounds['min_lon']
        
        # Add margins
        margin_x = self.config.visualization.window_width * 0.1
        margin_y = self.config.visualization.window_height * 0.1
        
        screen_width = self.config.visualization.window_width - 2 * margin_x
        screen_height = self.config.visualization.window_height - 2 * margin_y
        
        # Convert to screen coordinates
        x = margin_x + ((lon - bounds['min_lon']) / lon_range) * screen_width
        y = margin_y + ((bounds['max_lat'] - lat) / lat_range) * screen_height  # Flip Y axis
        
        return x, y
    
    def close(self) -> None:
        """Close visualization window."""
        if PYGAME_AVAILABLE and pygame.get_init():
            pygame.quit()


class BlueSkyVisualizer:
    """Wrapper for BlueSky's native visualization."""
    
    def __init__(self, config: VisualizationConfig):
        """Initialize BlueSky visualizer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def visualize_scenario(self, scenario: Scenario, tracks: Dict[str, List[TrackPoint]]) -> None:
        """Visualize scenario using BlueSky GUI."""
        self.logger.info("BlueSky visualization not yet implemented")
        # In real implementation:
        # - Launch BlueSky in GUI mode
        # - Load scenario into BlueSky
        # - Play back the simulation with visualization
        
        # Fallback to Pygame
        pygame_viz = Visualizer(self.config)
        pygame_viz.visualize_scenario(scenario, tracks)