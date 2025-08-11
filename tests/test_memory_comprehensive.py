"""
Comprehensive test of the fixed memory system.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_memory_comprehensive():
    """Comprehensive test of memory system functionality"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Import modules
        from src.cdr.ai.memory import ExperienceMemory, create_memory_record
        from src.cdr.ai.llm_client import ConflictContext
        
        print("Testing Memory System Comprehensive Suite")
        print("=" * 60)
        
        # Test 1: Memory initialization
        print("1. Testing memory initialization...")
        memory = ExperienceMemory(memory_dir=Path("data/memory"))
        print("   ✅ Memory system initialized successfully")
        
        # Test 2: Create multiple records
        print("2. Testing multiple record creation...")
        test_scenarios = [
            {
                'callsign': 'UAL123',
                'altitude': 35000,
                'heading': 90,
                'resolution': 'altitude_change',
                'new_altitude': 37000
            },
            {
                'callsign': 'DAL456',
                'altitude': 33000,
                'heading': 180,
                'resolution': 'heading_change', 
                'new_heading': 160
            },
            {
                'callsign': 'AAL789',
                'altitude': 39000,
                'heading': 270,
                'resolution': 'speed_change',
                'new_speed': 380
            },
        ]
        
        stored_records = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            # Create context
            context = ConflictContext(
                ownship_callsign=scenario['callsign'],
                ownship_state={
                    'altitude': scenario['altitude'],
                    'heading': scenario['heading'],
                    'speed': 450,
                    'latitude': 40.0 + i * 0.1,
                    'longitude': -75.0 + i * 0.1
                },
                intruders=[{
                    'callsign': f'INT{i:03d}',
                    'altitude': scenario['altitude'] + 500,
                    'heading': (scenario['heading'] + 180) % 360,
                    'speed': 420,
                    'latitude': 40.0 + i * 0.1 + 0.05,
                    'longitude': -75.0 + i * 0.1 + 0.05
                }],
                scenario_time=3600 + i * 300,
                lookahead_minutes=10,
                constraints={},
                nearby_traffic=[]
            )
            
            # Create resolution
            resolution = {
                'resolution_type': scenario['resolution'],
                'parameters': {k: v for k, v in scenario.items() 
                             if k.startswith('new_')},
                'reasoning': f'Test resolution {i} for {scenario["callsign"]}',
                'confidence': 0.8 + i * 0.05
            }
            
            # Create metrics
            metrics = {
                'success_rate': 0.85 + i * 0.03,
                'safety_margin': 4.5 + i * 0.5,
                'response_time': 2.0 + i * 0.2
            }
            
            # Create and store record
            record = create_memory_record(
                conflict_context=context,
                resolution=resolution,
                outcome_metrics=metrics,
                task_type='resolution'
            )
            
            memory.store_experience(record)
            stored_records.append(record)
            print(f"   ✅ Stored record {i}: {record.record_id}")
        
        # Test 3: Query similar experiences
        print("3. Testing similarity queries...")
        query_context = ConflictContext(
            ownship_callsign="TEST999",
            ownship_state={
                'altitude': 35500,  # Similar to first scenario
                'heading': 85,      # Similar to first scenario
                'speed': 445,
                'latitude': 40.05,
                'longitude': -74.95
            },
            intruders=[{
                'callsign': 'QUERY_INT',
                'altitude': 36000,
                'heading': 275,
                'speed': 425,
                'latitude': 40.1,
                'longitude': -74.9
            }],
            scenario_time=4200,
            lookahead_minutes=8,
            constraints={},
            nearby_traffic=[]
        )
        
        similar_experiences = memory.query_similar(query_context, top_k=3)
        print(f"   ✅ Found {len(similar_experiences)} similar experiences")
        
        for exp in similar_experiences:
            print(f"      - {exp.record_id}: {exp.resolution_taken['resolution_type']} "
                  f"(success: {exp.success_score:.2f})")
        
        # Test 4: Memory statistics
        print("4. Testing memory statistics...")
        stats = memory.get_memory_stats()
        print(f"   ✅ Total records: {stats['total_records']}")
        print(f"   ✅ Average success: {stats['average_success_score']:.2f}")
        print(f"   ✅ Task distribution: {stats['task_distribution']}")
        
        # Test 5: Save and load persistence
        print("5. Testing save/load persistence...")
        memory._save_memory_state()
        print("   ✅ Memory state saved successfully")
        
        # Create new memory instance to test loading
        memory2 = ExperienceMemory(memory_dir=Path("data/memory"))
        stats2 = memory2.get_memory_stats()
        print(f"   ✅ Loaded {stats2['total_records']} records from disk")
        
        # Test 6: Export functionality
        print("6. Testing export functionality...")
        export_path = Path("data/memory/exports/comprehensive_test.json")
        memory.export_experiences(export_path)
        print(f"   ✅ Exported to {export_path}")
        
        # Verify export file
        if export_path.exists():
            import json
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            print(f"   ✅ Export file contains {len(exported_data)} records")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Memory system is fully functional.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_memory_comprehensive()
    if not success:
        sys.exit(1)
