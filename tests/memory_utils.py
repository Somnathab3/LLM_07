"""
Memory system utilities for LLM_ATC7.
"""

import shutil
from pathlib import Path
import logging

def cleanup_memory(memory_dir: str = "data/memory", backup: bool = True):
    """
    Clean up memory system files.
    
    Args:
        memory_dir: Memory directory path
        backup: Whether to create backup before cleaning
    """
    memory_path = Path(memory_dir)
    
    if not memory_path.exists():
        print(f"Memory directory {memory_dir} does not exist.")
        return
    
    if backup and any(memory_path.glob("*.json")):
        backup_dir = memory_path / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        # Create timestamped backup
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = backup_dir / f"backup_{timestamp}"
        backup_subdir.mkdir(exist_ok=True)
        
        # Backup existing files
        for file_pattern in ["*.json", "*.pkl", "*.idx"]:
            for file_path in memory_path.glob(file_pattern):
                if file_path.is_file():
                    shutil.copy2(file_path, backup_subdir / file_path.name)
        
        print(f"Backed up memory files to {backup_subdir}")
    
    # Remove memory files
    for file_pattern in ["*.json", "*.pkl", "*.idx"]:
        for file_path in memory_path.glob(file_pattern):
            if file_path.is_file():
                file_path.unlink()
                print(f"Removed {file_path}")

def reset_memory_system(memory_dir: str = "data/memory"):
    """
    Completely reset the memory system.
    
    Args:
        memory_dir: Memory directory path
    """
    print("Resetting memory system...")
    cleanup_memory(memory_dir, backup=True)
    
    # Ensure directories exist
    memory_path = Path(memory_dir)
    memory_path.mkdir(parents=True, exist_ok=True)
    (memory_path / "backups").mkdir(exist_ok=True)
    (memory_path / "exports").mkdir(exist_ok=True)
    
    print("Memory system reset complete.")

def get_memory_info(memory_dir: str = "data/memory"):
    """
    Get information about the memory system.
    
    Args:
        memory_dir: Memory directory path
    """
    memory_path = Path(memory_dir)
    
    if not memory_path.exists():
        print(f"Memory directory {memory_dir} does not exist.")
        return
    
    print(f"Memory Directory: {memory_path.absolute()}")
    print("=" * 50)
    
    # Check for memory files
    records_file = memory_path / "records.json"
    index_file = memory_path / "faiss_index.idx"
    ids_file = memory_path / "record_ids.pkl"
    
    if records_file.exists():
        try:
            import json
            with open(records_file, 'r') as f:
                records = json.load(f)
            print(f"Records file: {len(records)} records")
        except Exception as e:
            print(f"Records file: Error reading ({e})")
    else:
        print("Records file: Not found")
    
    if index_file.exists():
        file_size = index_file.stat().st_size
        print(f"FAISS index: {file_size:,} bytes")
    else:
        print("FAISS index: Not found")
    
    if ids_file.exists():
        try:
            import pickle
            with open(ids_file, 'rb') as f:
                ids = pickle.load(f)
            print(f"Record IDs: {len(ids)} IDs")
        except Exception as e:
            print(f"Record IDs: Error reading ({e})")
    else:
        print("Record IDs: Not found")
    
    # Check subdirectories
    backups_dir = memory_path / "backups"
    if backups_dir.exists():
        backup_count = len(list(backups_dir.glob("backup_*")))
        print(f"Backups: {backup_count} backup directories")
    
    exports_dir = memory_path / "exports"
    if exports_dir.exists():
        export_count = len(list(exports_dir.glob("*.json")))
        print(f"Exports: {export_count} export files")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python memory_utils.py [info|cleanup|reset]")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    memory_dir = sys.argv[2] if len(sys.argv) > 2 else "data/memory"
    
    if action == "info":
        get_memory_info(memory_dir)
    elif action == "cleanup":
        cleanup_memory(memory_dir, backup=True)
    elif action == "reset":
        reset_memory_system(memory_dir)
    else:
        print(f"Unknown action: {action}")
        print("Available actions: info, cleanup, reset")
