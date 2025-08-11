#!/usr/bin/env python3
"""
Memory System Integration Example

This script demonstrates how to integrate the ExperienceMemory system
with the LLM client for enhanced conflict resolution.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cdr.ai.memory import ExperienceMemory, create_memory_record
from src.cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider, ConflictContext


def create_sample_conflict_context():
    """Create a sample conflict context for demonstration"""
    return ConflictContext(
        ownship_callsign="UAL123",
        ownship_state={
            'latitude': 41.978,
            'longitude': -87.904,
            'altitude': 35000,
            'heading': 270,
            'speed': 450,
        },
        intruders=[
            {
                'callsign': 'DAL456',
                'latitude': 41.998,
                'longitude': -87.884,
                'altitude': 35000,
                'heading': 90,
                'speed': 430,
            }
        ],
        scenario_time=1200.0,
        lookahead_minutes=10.0,
        constraints={'max_heading_change': 45, 'max_altitude_change': 2000},
        nearby_traffic=[]
    )


def demonstrate_memory_integration():
    """Demonstrate memory system integration with LLM client"""
    print("=== Memory System Integration Demo ===\n")
    
    # 1. Initialize memory system
    print("1. Initializing memory system...")
    memory_dir = Path("data/memory_demo")
    memory = ExperienceMemory(
        memory_dir=memory_dir,
        embedding_model="all-MiniLM-L6-v2",
        index_type="Flat",
        max_records=1000
    )
    print(f"   Memory system initialized with {len(memory.records)} existing records")
    
    # 2. Initialize LLM client with memory
    print("\n2. Initializing LLM client with memory integration...")
    llm_config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama2:7b",
        base_url="http://localhost:11434",
        temperature=0.1,
        max_tokens=1000
    )
    
    llm_client = LLMClient(config=llm_config, memory_store=memory)
    print("   LLM client initialized with memory store")
    
    # 3. Simulate some past experiences
    print("\n3. Adding sample past experiences to memory...")
    
    sample_experiences = [
        {
            'context': create_sample_conflict_context(),
            'resolution': {
                'resolution_type': 'heading_change',
                'parameters': {'new_heading_deg': 285},
                'reasoning': 'Turn right 15 degrees to maintain separation',
                'confidence': 0.85
            },
            'outcome': {
                'success_rate': 0.9,
                'safety_margin': 6.2,
                'resolution_time': 90,
                'additional_distance': 2.1
            }
        },
        {
            'context': create_sample_conflict_context(),
            'resolution': {
                'resolution_type': 'altitude_change',
                'parameters': {'target_altitude_ft': 37000},
                'reasoning': 'Climb to FL370 for vertical separation',
                'confidence': 0.92
            },
            'outcome': {
                'success_rate': 0.95,
                'safety_margin': 8.5,
                'resolution_time': 120,
                'additional_distance': 1.8
            }
        }
    ]
    
    for i, exp in enumerate(sample_experiences):
        record = create_memory_record(
            conflict_context=exp['context'],
            resolution=exp['resolution'],
            outcome_metrics=exp['outcome'],
            task_type='resolution'
        )
        memory.store_experience(record)
        print(f"   Stored experience {i+1}: {record.record_id}")
    
    # 4. Query memory for similar experiences
    print("\n4. Querying memory for similar experiences...")
    query_context = create_sample_conflict_context()
    similar_experiences = memory.query_similar(query_context, top_k=3)
    
    print(f"   Found {len(similar_experiences)} similar experiences:")
    for i, exp in enumerate(similar_experiences):
        print(f"   {i+1}. {exp.record_id}")
        print(f"      Success Score: {exp.success_score:.2f}")
        print(f"      Resolution: {exp.resolution_taken['resolution_type']}")
        print(f"      Reasoning: {exp.resolution_taken.get('reasoning', 'N/A')}")
    
    # 5. Demonstrate memory-enhanced LLM query
    print("\n5. Testing memory-enhanced conflict detection...")
    
    try:
        # This would normally call the LLM, but we'll simulate it
        print("   Querying memory for context...")
        memory_context = llm_client._query_memory_for_context(query_context)
        
        if memory_context:
            print("   Memory context retrieved:")
            print(memory_context[:200] + "..." if len(memory_context) > 200 else memory_context)
        else:
            print("   No relevant memory context found")
            
    except Exception as e:
        print(f"   Note: LLM query simulation failed (expected): {e}")
    
    # 6. Show memory statistics
    print("\n6. Memory system statistics:")
    stats = memory.get_memory_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 7. Demonstrate experience storage after resolution
    print("\n7. Simulating experience storage after conflict resolution...")
    
    # Simulate a resolution response
    class MockResolutionResponse:
        def __init__(self):
            self.resolution_type = 'heading_change'
            self.parameters = {'new_heading_deg': 275}
            self.reasoning = 'Turn left to avoid traffic'
            self.confidence = 0.88
    
    mock_response = MockResolutionResponse()
    mock_metrics = {
        'success_rate': 0.87,
        'safety_margin': 7.3,
        'resolution_time': 95
    }
    
    llm_client._store_experience(query_context, mock_response, 'resolution', mock_metrics)
    print("   Experience stored successfully")
    
    # 8. Final statistics
    print("\n8. Final memory statistics:")
    final_stats = memory.get_memory_stats()
    print(f"   Total records: {final_stats['total_records']}")
    print(f"   Queries made: {final_stats['query_count']}")
    print(f"   Hit rate: {final_stats['hit_rate']:.2f}")
    
    print("\n=== Demo Complete ===")
    print("The memory system is now integrated and ready for use!")
    
    return memory, llm_client


def demonstrate_advanced_features():
    """Demonstrate advanced memory features"""
    print("\n=== Advanced Memory Features Demo ===\n")
    
    memory_dir = Path("data/memory_advanced")
    
    # Test different index types
    index_types = ["Flat", "IVFFlat"]  # Skip HNSW for demo
    
    for index_type in index_types:
        print(f"Testing {index_type} index...")
        
        memory = ExperienceMemory(
            memory_dir=memory_dir / index_type.lower(),
            index_type=index_type,
            max_records=100
        )
        
        # Add some experiences
        for i in range(10):
            context = create_sample_conflict_context()
            # Modify context slightly for variety
            context.ownship_state['heading'] = 270 + i * 10
            context.ownship_state['altitude'] = 35000 + i * 1000
            
            resolution = {
                'resolution_type': 'heading_change' if i % 2 == 0 else 'altitude_change',
                'parameters': {'new_heading_deg': 270 + i * 5} if i % 2 == 0 else {'target_altitude_ft': 35000 + i * 500},
                'confidence': 0.8 + i * 0.01
            }
            
            outcome = {
                'success_rate': 0.85 + i * 0.01,
                'safety_margin': 5.0 + i * 0.2
            }
            
            record = create_memory_record(context, resolution, outcome)
            memory.store_experience(record)
        
        # Test query
        query_context = create_sample_conflict_context()
        results = memory.query_similar(query_context, top_k=3)
        print(f"   {index_type}: {len(results)} results found")
        
        # Test export
        export_path = memory_dir / f"{index_type.lower()}_export.json"
        memory.export_experiences(export_path)
        print(f"   Exported to {export_path}")
    
    print("\nAdvanced features demo complete!")


if __name__ == "__main__":
    try:
        memory, llm_client = demonstrate_memory_integration()
        demonstrate_advanced_features()
        
        print("\n🎉 Memory system integration successful!")
        print("\nNext steps:")
        print("- Install sentence-transformers: pip install sentence-transformers")
        print("- Run the memory test suite: python test_memory_system.py")
        print("- Integrate with your existing CDR pipeline")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("pip install sentence-transformers faiss-cpu")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Please check the implementation and try again.")
