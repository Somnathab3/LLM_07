"""Configuration classes for visualization module."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    enable: bool = False
    window_width: int = 1200
    window_height: int = 800
    fps: int = 30
    scale_factor: float = 1.0
    show_trajectories: bool = True
    show_conflicts: bool = True
    show_resolutions: bool = True
    auto_zoom: bool = True
    background_color: tuple = (0, 0, 50)  # Dark blue
    aircraft_color: tuple = (255, 255, 255)  # White
    conflict_color: tuple = (255, 0, 0)  # Red
    resolution_color: tuple = (0, 255, 0)  # Green


@dataclass
class Config:
    """Main configuration class"""
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    debug: bool = False
    log_level: str = "INFO"
    
    # Additional config parameters can be added here
    output_dir: str = "output"
    data_dir: str = "data"
    temp_dir: str = "temp"
    
    def __post_init__(self):
        """Post-initialization validation"""
        if self.visualization.window_width <= 0 or self.visualization.window_height <= 0:
            raise ValueError("Window dimensions must be positive")
        if self.visualization.fps <= 0:
            raise ValueError("FPS must be positive")


# Default configuration instance
DEFAULT_CONFIG = Config()
