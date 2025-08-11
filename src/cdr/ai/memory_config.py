"""
Memory system configuration and utilities.
"""

from pathlib import Path
from typing import Dict, Any

# Default memory configuration
DEFAULT_MEMORY_CONFIG = {
    "memory_dir": "data/memory",
    "embedding_model": "all-MiniLM-L6-v2",
    "index_type": "Flat",  # Use Flat index for better reliability
    "embedding_dim": 384,
    "max_records": 10000,
    "backup_enabled": True,
    "export_enabled": True,
    "auto_save_interval": 100,  # Save every 100 records
}

# Fallback configuration for environments with limited resources
MINIMAL_MEMORY_CONFIG = {
    "memory_dir": "data/memory",
    "embedding_model": None,  # Use manual features only
    "index_type": "Flat",
    "embedding_dim": 50,  # Smaller dimension for manual features
    "max_records": 1000,
    "backup_enabled": False,
    "export_enabled": True,
    "auto_save_interval": 50,
}

def get_memory_config(minimal: bool = False) -> Dict[str, Any]:
    """
    Get memory configuration.
    
    Args:
        minimal: Use minimal configuration for limited resources
        
    Returns:
        Memory configuration dictionary
    """
    return MINIMAL_MEMORY_CONFIG if minimal else DEFAULT_MEMORY_CONFIG

def ensure_memory_directories(memory_dir: str = "data/memory"):
    """
    Ensure all memory directories exist.
    
    Args:
        memory_dir: Base memory directory path
    """
    base_dir = Path(memory_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (base_dir / "backups").mkdir(exist_ok=True)
    (base_dir / "exports").mkdir(exist_ok=True)
    (base_dir / "temp").mkdir(exist_ok=True)
    
    return base_dir

def create_memory_system(config: Dict[str, Any] = None):
    """
    Create and configure memory system.
    
    Args:
        config: Memory configuration (uses default if None)
        
    Returns:
        Configured ExperienceMemory instance
    """
    if config is None:
        config = get_memory_config()
    
    # Ensure directories exist
    ensure_memory_directories(config["memory_dir"])
    
    try:
        from .memory import ExperienceMemory
        
        return ExperienceMemory(
            memory_dir=Path(config["memory_dir"]),
            embedding_model=config["embedding_model"],
            index_type=config["index_type"],
            embedding_dim=config["embedding_dim"],
            max_records=config["max_records"]
        )
    except Exception as e:
        print(f"Failed to create memory system: {e}")
        print("Falling back to minimal configuration...")
        
        # Try with minimal configuration
        minimal_config = get_memory_config(minimal=True)
        return ExperienceMemory(
            memory_dir=Path(minimal_config["memory_dir"]),
            embedding_model=minimal_config["embedding_model"],
            index_type=minimal_config["index_type"],
            embedding_dim=minimal_config["embedding_dim"],
            max_records=minimal_config["max_records"]
        )
