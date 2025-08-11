#!/usr/bin/env python3
"""
Test the Experience Memory System

This test script validates the complete memory system functionality including:
- Embedding generation and similarity search
- Experience storage and retrieval
- FAISS index operations
- Geometric feature extraction
- Memory persistence and loading
"""

import sys
import os
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cdr.ai.memory import (
    ExperienceMemory, MemoryRecord, ConflictGeometry, create_memory_record
)
from src.cdr.ai.llm_client import ConflictContext


def create_test_context(scenario_id: int = 1) -> ConflictContext:
    """Create a test conflict context"""
    return ConflictContext(
        ownship_callsign=f"OWNSHIP_{scenario_id}",
        ownship_state={
            'latitude': 41.978 + scenario_id * 0.01,
            'longitude': -87.904 + scenario_id * 0.01,
            'altitude': 35000 + scenario_id * 1000,
            'heading': 270 + scenario_id * 10,
            'speed': 450 + scenario_id * 10,
        },
        intruders=[
            {
                'callsign': f'INTRUDER_{scenario_id}',
                'latitude': 41.998 + scenario_id * 0.01,
                'longitude': -87.884 + scenario_id * 0.01,
                'altitude': 35000 + scenario_id * 500,
                'heading': 90 + scenario_id * 15,
                'speed': 430 + scenario_id * 5,
            }
        ],
        scenario_time=1200.0 + scenario_id * 60,
        lookahead_minutes=10.0,
        constraints={'max_heading_change': 45, 'max_altitude_change': 2000},
        nearby_traffic=[]
    )


def test_memory_initialization():
    """Test memory system initialization"""
    print("Testing memory system initialization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_dir = Path(temp_dir) / "test_memory"
        
        # Test basic initialization
        memory = ExperienceMemory(
            memory_dir=memory_dir,
            embedding_model="all-MiniLM-L6-v2",
            index_type="Flat",
            max_records=1000
        )
        
        print(f"✓ Memory system initialized")
        print(f"  - Memory directory: {memory_dir}")
        print(f"  - Index type: {memory.index_type}")
        print(f"  - Embedding dimension: {memory.embedding_dim}")
        print(f"  - Max records: {memory.max_records}")
        
        # Test statistics
        stats = memory.get_memory_stats()
        print(f"  - Initial stats: {stats}")
        
        return True


def test_embedding_generation():
    """Test context embedding generation"""
    print("\nTesting embedding generation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_dir = Path(temp_dir) / "test_memory"
        memory = ExperienceMemory(memory_dir=memory_dir, index_type="Flat")
        
        # Test with different contexts
        for i in range(3):
            context = create_test_context(i + 1)
            context_dict = {
                'ownship_callsign': context.ownship_callsign,
                'ownship_state': context.ownship_state,
                'intruders': context.intruders,
                'scenario_time': context.scenario_time,
                'lookahead_minutes': context.lookahead_minutes,
                'constraints': context.constraints,
            }
            
            embedding = memory._generate_context_embedding(context_dict, 'resolution')
            print(f"✓ Embedding {i+1}: shape={embedding.shape}, type={embedding.dtype}")
            
            # Verify embedding properties
            assert embedding.shape[0] == memory.embedding_dim
            assert embedding.dtype == np.float32
            assert not np.isnan(embedding).any()
        
        return True


def test_geometric_features():
    """Test geometric feature extraction"""
    print("\nTesting geometric feature extraction...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_dir = Path(temp_dir) / "test_memory"
        memory = ExperienceMemory(memory_dir=memory_dir, index_type="Flat")
        
        # Test different conflict scenarios
        scenarios = [
            (1, "head-on scenario"),
            (10, "crossing scenario"),
            (20, "overtaking scenario"),
        ]
        
        for scenario_id, description in scenarios:
            context = create_test_context(scenario_id)
            geometry = memory._extract_geometric_features(context)
            
            print(f"✓ {description}:")
            print(f"  - Approach angle: {geometry.approach_angle:.1f}°")
            print(f"  - Speed ratio: {geometry.speed_ratio:.2f}")
            print(f"  - Altitude separation: {geometry.altitude_separation:.0f} ft")
            print(f"  - Horizontal separation: {geometry.horizontal_separation:.1f} nm")
            print(f"  - Time to CPA: {geometry.time_to_cpa:.1f} min")
            print(f"  - Conflict type: {geometry.conflict_type}")
            print(f"  - Urgency level: {geometry.urgency_level}")
            
            # Verify geometry properties
            assert 0 <= geometry.approach_angle <= 180
            assert geometry.speed_ratio > 0
            assert geometry.conflict_type in ['head_on', 'crossing', 'overtaking', 'vertical', 'none']
            assert geometry.urgency_level in ['low', 'medium', 'high', 'critical']
        
        return True


def test_experience_storage():
    """Test experience storage and retrieval"""
    print("\nTesting experience storage...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_dir = Path(temp_dir) / "test_memory"
        memory = ExperienceMemory(memory_dir=memory_dir, index_type="Flat")
        
        # Create and store multiple experiences
        num_experiences = 10
        stored_records = []
        
        for i in range(num_experiences):
            context = create_test_context(i + 1)
            
            # Create resolution and outcome
            resolution = {
                'resolution_type': 'heading_change',
                'parameters': {'new_heading_deg': 270 + i * 10},
                'reasoning': f'Test resolution {i+1}',
                'confidence': 0.8 + i * 0.02
            }
            
            outcome_metrics = {
                'success_rate': 0.7 + i * 0.03,
                'safety_margin': 5.0 + i * 0.5,
                'resolution_time': 120 + i * 10,
                'additional_distance': i * 2.0
            }
            
            # Create and store record
            record = create_memory_record(
                conflict_context=context,
                resolution=resolution,
                outcome_metrics=outcome_metrics,
                task_type='resolution'
            )
            
            memory.store_experience(record)
            stored_records.append(record)
            
            print(f"✓ Stored experience {i+1}: {record.record_id}")
        
        # Verify storage
        stats = memory.get_memory_stats()
        print(f"✓ Storage complete: {stats['total_records']} records stored")
        
        assert stats['total_records'] == num_experiences
        
        return True


def test_similarity_search():
    """Test similarity search functionality"""
    print("\nTesting similarity search...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_dir = Path(temp_dir) / "test_memory"
        memory = ExperienceMemory(memory_dir=memory_dir, index_type="Flat")
        
        # Store some experiences first
        for i in range(15):
            context = create_test_context(i + 1)
            resolution = {
                'resolution_type': 'heading_change' if i % 2 == 0 else 'altitude_change',
                'parameters': {'new_heading_deg': 270 + i * 5} if i % 2 == 0 else {'target_altitude_ft': 35000 + i * 500},
                'confidence': 0.7 + i * 0.02
            }
            outcome_metrics = {
                'success_rate': 0.6 + i * 0.025,
                'safety_margin': 4.0 + i * 0.3
            }
            
            record = create_memory_record(context, resolution, outcome_metrics)
            memory.store_experience(record)
        
        # Test similarity queries
        query_context = create_test_context(3)  # Should be similar to stored experience 3
        
        similar_records = memory.query_similar(query_context, top_k=5)
        
        print(f"✓ Found {len(similar_records)} similar experiences")
        
        for i, record in enumerate(similar_records):
            print(f"  {i+1}. {record.record_id}")
            print(f"     Success score: {record.success_score:.3f}")
            print(f"     Task type: {record.task_type}")
            print(f"     Resolution: {record.resolution_taken['resolution_type']}")
        
        # Verify results
        assert len(similar_records) <= 5
        assert all(isinstance(record, MemoryRecord) for record in similar_records)
        
        # Test memory statistics after queries
        stats = memory.get_memory_stats()
        print(f"✓ Query stats: {stats['query_count']} queries, {stats['hit_count']} hits")
        
        return True


def test_memory_persistence():
    """Test memory persistence and loading"""
    print("\nTesting memory persistence...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_dir = Path(temp_dir) / "test_memory"
        
        # Create and populate memory
        memory1 = ExperienceMemory(memory_dir=memory_dir, index_type="Flat")
        
        # Store some experiences
        original_records = {}
        for i in range(5):
            context = create_test_context(i + 1)
            resolution = {'resolution_type': 'heading_change', 'parameters': {'new_heading_deg': 270 + i * 10}}
            outcome_metrics = {'success_rate': 0.8, 'safety_margin': 5.0}
            
            record = create_memory_record(context, resolution, outcome_metrics)
            memory1.store_experience(record)
            original_records[record.record_id] = record
        
        # Force save
        memory1._save_memory_state()
        stats1 = memory1.get_memory_stats()
        print(f"✓ Saved {stats1['total_records']} records to disk")
        
        # Create new memory instance and load
        memory2 = ExperienceMemory(memory_dir=memory_dir, index_type="Flat")
        stats2 = memory2.get_memory_stats()
        print(f"✓ Loaded {stats2['total_records']} records from disk")
        
        # Verify loaded data
        assert stats1['total_records'] == stats2['total_records']
        
        # Test that queries work on loaded memory
        query_context = create_test_context(2)
        similar_records = memory2.query_similar(query_context, top_k=3)
        print(f"✓ Query on loaded memory returned {len(similar_records)} results")
        
        return True


def test_advanced_features():
    """Test advanced memory features"""
    print("\nTesting advanced features...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_dir = Path(temp_dir) / "test_memory"
        memory = ExperienceMemory(memory_dir=memory_dir, index_type="Flat", max_records=10)
        
        # Test duplicate detection and merging
        context = create_test_context(1)
        resolution = {'resolution_type': 'heading_change', 'parameters': {'new_heading_deg': 280}}
        outcome_metrics = {'success_rate': 0.7, 'safety_margin': 4.0}
        
        record1 = create_memory_record(context, resolution, outcome_metrics)
        record2 = create_memory_record(context, resolution, outcome_metrics)
        
        memory.store_experience(record1)
        initial_count = len(memory.records)
        
        memory.store_experience(record2)  # Should merge with first
        final_count = len(memory.records)
        
        print(f"✓ Duplicate handling: {initial_count} -> {final_count} records")
        
        # Test capacity management
        for i in range(15):  # Exceed max_records
            context = create_test_context(i + 10)
            resolution = {'resolution_type': 'altitude_change', 'parameters': {'target_altitude_ft': 35000 + i * 1000}}
            outcome_metrics = {'success_rate': 0.8, 'safety_margin': 5.0}
            
            record = create_memory_record(context, resolution, outcome_metrics)
            memory.store_experience(record)
        
        stats = memory.get_memory_stats()
        print(f"✓ Capacity management: {stats['total_records']} records (max: {memory.max_records})")
        assert stats['total_records'] <= memory.max_records
        
        # Test cleanup
        memory.cleanup_old_records(max_age_days=0)  # Remove all records
        stats_after_cleanup = memory.get_memory_stats()
        print(f"✓ Cleanup: {stats_after_cleanup['total_records']} records remaining")
        
        # Test export
        export_path = memory_dir / "export.json"
        memory.export_experiences(export_path)
        
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        print(f"✓ Export: {len(exported_data)} records exported")
        
        return True


def test_different_index_types():
    """Test different FAISS index types"""
    print("\nTesting different index types...")
    
    index_types = ["Flat", "IVFFlat", "HNSW"]
    
    for index_type in index_types:
        print(f"Testing {index_type} index...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_dir = Path(temp_dir) / f"test_memory_{index_type.lower()}"
            
            try:
                memory = ExperienceMemory(
                    memory_dir=memory_dir, 
                    index_type=index_type,
                    max_records=500
                )
                
                # Add some records
                for i in range(20):
                    context = create_test_context(i + 1)
                    resolution = {'resolution_type': 'heading_change', 'parameters': {'new_heading_deg': 270 + i * 5}}
                    outcome_metrics = {'success_rate': 0.8, 'safety_margin': 5.0}
                    
                    record = create_memory_record(context, resolution, outcome_metrics)
                    memory.store_experience(record)
                
                # Test search
                query_context = create_test_context(5)
                results = memory.query_similar(query_context, top_k=3)
                
                print(f"✓ {index_type}: {len(results)} results found")
                
            except Exception as e:
                print(f"✗ {index_type}: Error - {e}")
                return False
    
    return True


def main():
    """Run all tests"""
    print("=== Experience Memory System Tests ===\n")
    
    tests = [
        test_memory_initialization,
        test_embedding_generation,
        test_geometric_features,
        test_experience_storage,
        test_similarity_search,
        test_memory_persistence,
        test_advanced_features,
        test_different_index_types,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED\n")
            else:
                failed += 1
                print(f"❌ {test.__name__} FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} FAILED with error: {e}\n")
    
    print("=== Test Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Memory system is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
