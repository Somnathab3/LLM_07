#!/usr/bin/env python3
"""
Test script to verify memory system functionality and fix any issues.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_memory_system():
    """Test the memory system functionality"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Import memory system
        from src.cdr.ai.memory import ExperienceMemory, create_memory_record
        from src.cdr.ai.llm_client import ConflictContext
        
        logger.info("Successfully imported memory system")
        
        # Test memory initialization
        memory_dir = Path("data/memory")
        memory = ExperienceMemory(memory_dir=memory_dir)
        logger.info(f"Memory system initialized with directory: {memory_dir}")
        
        # Test creating a sample memory record
        sample_context = ConflictContext(
            ownship_callsign="TEST001",
            ownship_state={
                'altitude': 35000,
                'heading': 90,
                'speed': 450,
                'latitude': 40.0,
                'longitude': -75.0
            },
            intruders=[{
                'callsign': 'TEST002',
                'altitude': 35000,
                'heading': 270,
                'speed': 420,
                'latitude': 40.1,
                'longitude': -74.9
            }],
            scenario_time=3600,
            lookahead_minutes=10,
            constraints={},
            nearby_traffic=[]
        )
        
        sample_resolution = {
            'resolution_type': 'altitude_change',
            'parameters': {'new_altitude': 37000},
            'reasoning': 'Test resolution for memory system',
            'confidence': 0.8
        }
        
        sample_metrics = {
            'success_rate': 0.9,
            'safety_margin': 5.2,
            'response_time': 2.1
        }
        
        # Create and store memory record
        record = create_memory_record(
            conflict_context=sample_context,
            resolution=sample_resolution,
            outcome_metrics=sample_metrics,
            task_type='resolution'
        )
        
        logger.info(f"Created memory record: {record.record_id}")
        
        # Store the experience
        memory.store_experience(record)
        logger.info("Successfully stored experience in memory")
        
        # Test memory statistics
        stats = memory.get_memory_stats()
        logger.info(f"Memory statistics: {stats}")
        
        # Test saving memory state
        logger.info("Testing memory save functionality...")
        memory._save_memory_state()
        logger.info("Memory save completed successfully")
        
        # Test querying similar experiences
        similar_experiences = memory.query_similar(sample_context, top_k=3)
        logger.info(f"Found {len(similar_experiences)} similar experiences")
        
        # Test export functionality
        export_path = memory_dir / "exports" / "test_export.json"
        memory.export_experiences(export_path)
        logger.info(f"Exported experiences to {export_path}")
        
        logger.info("✅ All memory system tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory system test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test function"""
    print("Testing LLM_ATC7 Memory System...")
    print("=" * 50)
    
    success = test_memory_system()
    
    print("=" * 50)
    if success:
        print("✅ Memory system is working correctly!")
    else:
        print("❌ Memory system has issues that need to be fixed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
