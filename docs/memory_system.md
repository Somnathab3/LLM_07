# Experience Memory System Documentation

## Overview

The Experience Memory System is a sophisticated AI memory component that stores and retrieves past conflict resolution experiences to enhance decision-making quality in air traffic control scenarios. It uses advanced vector embeddings and similarity search to find relevant past experiences and improve LLM-based conflict resolution.

## Architecture

### Core Components

1. **ExperienceMemory**: Main memory management class
2. **MemoryRecord**: Data structure for storing individual experiences
3. **ConflictGeometry**: Geometric feature extraction for conflicts
4. **FAISS Integration**: High-performance similarity search
5. **Sentence Transformers**: Semantic embedding generation

### Key Features

- **Vector Embeddings**: Converts conflict scenarios into high-dimensional vectors
- **Similarity Search**: FAISS-powered efficient retrieval of similar experiences
- **Temporal Weighting**: More recent experiences have higher relevance
- **Success-based Ranking**: Prioritizes successful resolution strategies
- **Geometric Analysis**: Extracts spatial and temporal conflict features
- **Memory Persistence**: Automatic saving and loading of experience data

## Implementation Details

### ExperienceMemory Class

```python
class ExperienceMemory:
    def __init__(self, 
                 memory_dir: Path = Path("data/memory"),
                 embedding_model: str = "all-MiniLM-L6-v2",
                 index_type: str = "IVFFlat",
                 embedding_dim: int = 384,
                 max_records: int = 10000):
```

#### Key Methods

**`_generate_context_embedding(context, task_type)`**
- Converts conflict context to vector embedding
- Uses sentence transformers for semantic encoding
- Fallback to manual feature engineering if needed
- Handles different conflict geometries and temporal features

**`store_experience(record)`**
- Stores new experience with duplicate detection
- Maintains FAISS index for efficient search
- Implements capacity management (removes oldest when full)
- Automatic index training for IVF indices

**`query_similar(context, top_k=5)`**
- Multi-stage similarity search with sophisticated ranking
- Combines vector similarity, temporal relevance, success rate
- Returns most relevant past experiences
- Updates hit rate statistics

**`_rerank_candidates(candidates, query_context)`**
- Advanced reranking using multiple factors:
  - Vector similarity (40%)
  - Temporal relevance (20%)
  - Success rate (20%)
  - Context similarity (15%)
  - Task relevance (5%)

### MemoryRecord Structure

```python
@dataclass
class MemoryRecord:
    record_id: str                    # Unique identifier
    timestamp: datetime               # When experience occurred
    conflict_context: Dict[str, Any]  # Serialized ConflictContext
    resolution_taken: Dict[str, Any]  # Resolution action
    outcome_metrics: Dict[str, Any]   # Success metrics
    scenario_features: Dict[str, Any] # Geometric features
    task_type: str                    # Type of task performed
    success_score: float              # Overall success (0.0-1.0)
    metadata: Dict[str, Any]          # Additional context
```

### Geometric Feature Extraction

The system extracts rich geometric features from conflict scenarios:

- **Approach Angle**: Relative heading difference between aircraft
- **Speed Ratio**: Intruder speed relative to ownship
- **Altitude Separation**: Current vertical separation
- **Horizontal Separation**: Current horizontal distance
- **Time to CPA**: Estimated time to closest point of approach
- **Conflict Type**: Classification (head_on, crossing, overtaking, vertical)
- **Urgency Level**: Risk assessment (low, medium, high, critical)

## Usage Examples

### Basic Integration

```python
from src.cdr.ai.memory import ExperienceMemory
from src.cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider

# Initialize memory system
memory = ExperienceMemory(
    memory_dir=Path("data/memory"),
    embedding_model="all-MiniLM-L6-v2",
    index_type="IVFFlat",
    max_records=10000
)

# Initialize LLM client with memory
llm_config = LLMConfig(
    provider=LLMProvider.OLLAMA,
    model_name="llama2:7b"
)
llm_client = LLMClient(config=llm_config, memory_store=memory)
```

### Storing Experiences

```python
from src.cdr.ai.memory import create_memory_record

# After a conflict resolution
record = create_memory_record(
    conflict_context=context,
    resolution=resolution_response,
    outcome_metrics={
        'success_rate': 0.95,
        'safety_margin': 7.2,
        'resolution_time': 120,
        'additional_distance': 1.8
    },
    task_type='resolution'
)

memory.store_experience(record)
```

### Querying Similar Experiences

```python
# Find similar past experiences
similar_experiences = memory.query_similar(current_context, top_k=5)

for exp in similar_experiences:
    print(f"Similar conflict (Success: {exp.success_score:.2f})")
    print(f"Resolution: {exp.resolution_taken['resolution_type']}")
    print(f"Reasoning: {exp.resolution_taken.get('reasoning')}")
```

## FAISS Index Types

The system supports multiple FAISS index types for different performance characteristics:

### Flat Index (`IndexFlatL2`)
- **Use Case**: Small datasets (< 1000 records)
- **Performance**: Exact search, slower for large datasets
- **Memory**: Higher memory usage
- **Setup**: No training required

### IVF Index (`IndexIVFFlat`)
- **Use Case**: Medium to large datasets (1000-100k records)
- **Performance**: Approximate search, much faster
- **Memory**: Lower memory usage
- **Setup**: Requires training with ~100+ records

### HNSW Index (`IndexHNSWFlat`)
- **Use Case**: Large datasets requiring fast queries
- **Performance**: Very fast approximate search
- **Memory**: Moderate memory usage
- **Setup**: No training required, good for real-time applications

## Performance Optimization

### Embedding Generation
- Uses sentence-transformers for semantic understanding
- Fallback to manual feature engineering for reliability
- Caches embeddings to avoid recomputation
- Supports batch processing for multiple records

### Memory Management
- Automatic capacity management (LRU-style removal)
- Periodic index optimization and cleanup
- Configurable maximum record limits
- Efficient duplicate detection and merging

### Query Optimization
- Multi-stage candidate filtering and reranking
- Temporal relevance weighting
- Success-rate based prioritization
- Configurable top-k results

## Integration with LLM Client

The memory system seamlessly integrates with the LLM client:

### Enhanced Prompting
```python
def _query_memory_for_context(self, context: ConflictContext) -> str:
    """Query memory system for relevant past experiences"""
    similar_experiences = self.memory_store.query_similar(context, top_k=3)
    
    # Format memory context for LLM prompt
    memory_context = "\n\nRELEVANT PAST EXPERIENCES:\n"
    for exp in similar_experiences:
        memory_context += f"Resolution: {exp.resolution_taken['resolution_type']}\n"
        memory_context += f"Success Rate: {exp.success_score:.1f}\n"
        memory_context += f"Reasoning: {exp.resolution_taken.get('reasoning')}\n\n"
    
    return memory_context
```

### Automatic Experience Storage
```python
def _store_experience(self, context, response, task_type, success_metrics):
    """Store experience in memory system"""
    record = create_memory_record(
        conflict_context=context,
        resolution=response,
        outcome_metrics=success_metrics,
        task_type=task_type
    )
    self.memory_store.store_experience(record)
```

## Configuration Options

### Memory System Configuration
```python
memory = ExperienceMemory(
    memory_dir=Path("data/memory"),           # Storage directory
    embedding_model="all-MiniLM-L6-v2",      # Sentence transformer model
    index_type="IVFFlat",                     # FAISS index type
    embedding_dim=384,                        # Embedding dimension
    max_records=10000                         # Maximum stored records
)
```

### Performance Tuning
- **Small datasets (< 1K)**: Use `Flat` index
- **Medium datasets (1K-100K)**: Use `IVFFlat` index
- **Large datasets (> 100K)**: Use `HNSW` index
- **Memory constrained**: Reduce `max_records`
- **Speed critical**: Use smaller embedding models

## Monitoring and Statistics

### Memory Statistics
```python
stats = memory.get_memory_stats()
# Returns:
# {
#     'total_records': 1247,
#     'query_count': 89,
#     'hit_count': 67,
#     'hit_rate': 0.75,
#     'average_success_score': 0.84,
#     'task_distribution': {'resolution': 1200, 'detection': 47},
#     'embedding_model': 'all-MiniLM-L6-v2',
#     'index_type': 'IVFFlat',
#     'index_trained': True
# }
```

### Maintenance Operations
```python
# Clean up old records
memory.cleanup_old_records(max_age_days=90)

# Export experiences for analysis
memory.export_experiences(Path("analysis/experiences.json"))

# Force save current state
memory._save_memory_state()
```

## Dependencies

- **Core**: `numpy`, `faiss-cpu`, `pathlib`, `datetime`
- **ML**: `sentence-transformers` (optional, with fallback)
- **Data**: `json`, `pickle` for persistence
- **Utils**: `hashlib` for duplicate detection

## Testing

Run the comprehensive test suite:
```bash
python test_memory_system.py
```

Run the integration demo:
```bash
python demo_memory_integration.py
```

## Best Practices

1. **Regular Cleanup**: Run `cleanup_old_records()` periodically
2. **Monitor Hit Rate**: Track query success with `get_memory_stats()`
3. **Backup Data**: Export experiences regularly for backup
4. **Index Selection**: Choose appropriate FAISS index for your dataset size
5. **Success Metrics**: Provide accurate outcome metrics for better ranking
6. **Capacity Planning**: Set `max_records` based on available memory

## Troubleshooting

### Common Issues

**"Sentence transformers model not found"**
- Install: `pip install sentence-transformers`
- System will fallback to manual features if unavailable

**"FAISS index not trained"**
- IVFFlat indices require 100+ records for training
- Use Flat index for smaller datasets

**"Memory directory permission error"**
- Ensure write permissions to memory directory
- Check disk space availability

**"High memory usage"**
- Reduce `max_records` limit
- Use IVFFlat instead of Flat index
- Clean up old records more frequently

### Performance Optimization

- Use appropriate FAISS index for dataset size
- Implement batch processing for multiple queries
- Monitor and tune embedding model selection
- Regular maintenance and cleanup operations

## Future Enhancements

Planned improvements include:
- Distributed memory across multiple nodes
- Real-time learning from ongoing operations
- Advanced conflict pattern recognition
- Integration with external knowledge bases
- Automated hyperparameter tuning

---

The Experience Memory System provides a sophisticated foundation for continuous learning and improvement in AI-driven air traffic control applications.
