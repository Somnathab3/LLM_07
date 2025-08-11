"""Configuration management utilities"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration"""
    
    simulation: Dict[str, Any] = None
    llm: Dict[str, Any] = None
    bluesky: Dict[str, Any] = None
    memory: Dict[str, Any] = None
    
    def __post_init__(self):
        """Set defaults"""
        if self.simulation is None:
            self.simulation = {
                'cycle_interval_seconds': 60.0,
                'lookahead_minutes': 10.0,
                'max_simulation_time_minutes': 120.0
            }
        
        if self.llm is None:
            self.llm = {
                'provider': 'ollama',
                'model': 'llama3.1:8b',
                'temperature': 0.1
            }
        
        if self.bluesky is None:
            self.bluesky = {
                'host': '127.0.0.1',
                'port': 8888,
                'headless': True
            }
        
        if self.memory is None:
            self.memory = {
                'enabled': True,
                'embedding_dim': 768
            }
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None):
        """Load configuration from file"""
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            return cls(**config_data)
        return cls()
