# Ollama LLM Client Integration

This implementation provides a complete integration with Ollama for conflict detection and resolution in the CDR (Collision Detection and Resolution) system.

## Features

- **Real Ollama Integration**: Direct API calls to Ollama server
- **Model Management**: Automatic model availability checking and pulling
- **Robust Error Handling**: Retry mechanisms and fallback responses
- **Advanced Prompting**: Aviation-specific prompts optimized for conflict detection
- **Safety Validation**: Response validation and confidence scoring
- **Connection Testing**: Built-in connectivity and functionality tests

## Setup

### 1. Install Ollama

Download and install Ollama from [https://ollama.ai](https://ollama.ai)

### 2. Pull a Model

```bash
# Pull the recommended model
ollama pull llama3.1:8b

# Or pull other supported models
ollama pull llama3:8b
ollama pull mistral:7b
ollama pull codellama:7b
```

### 3. Start Ollama Server

```bash
# Start the Ollama server (usually runs automatically)
ollama serve

# Test with a model
ollama run llama3.1:8b
```

## Usage

### Basic Usage

```python
from src.cdr.ai.llm_client import LLMClient, LLMConfig, LLMProvider, ConflictContext

# Configure the client
config = LLMConfig(
    provider=LLMProvider.OLLAMA,
    model_name="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.1,
    max_tokens=1000
)

# Initialize client
client = LLMClient(config)

# Test connection
test_result = client.test_connection()
if test_result['generation_test']:
    print("✅ LLM client ready!")
else:
    print("❌ Connection failed")
```

### Conflict Detection

```python
# Create conflict context
context = ConflictContext(
    ownship_callsign="AAL123",
    ownship_state={
        'latitude': 40.7128,
        'longitude': -74.0060,
        'altitude': 35000,
        'heading': 90,
        'speed': 450
    },
    intruders=[{
        'callsign': 'UAL456',
        'latitude': 40.7150,
        'longitude': -73.9950,
        'altitude': 35000,
        'heading': 270,
        'speed': 440
    }],
    scenario_time=0.0,
    lookahead_minutes=10.0,
    constraints={}
)

# Detect conflicts
result = client.detect_conflicts(context)
print(f"Conflicts detected: {result['conflicts_detected']}")
```

### Resolution Generation

```python
# Generate resolution for detected conflicts
if result['conflicts_detected']:
    resolution = client.generate_resolution(context, result)
    
    if resolution.success:
        print(f"Resolution: {resolution.resolution_type}")
        print(f"Parameters: {resolution.parameters}")
        print(f"Confidence: {resolution.confidence}")
```

## Configuration

### LLMConfig Parameters

- **provider**: Must be `LLMProvider.OLLAMA`
- **model_name**: Ollama model name (e.g., "llama3.1:8b", "mistral:7b")
- **base_url**: Ollama server URL (default: "http://localhost:11434")
- **temperature**: Response randomness (0.0-1.0, recommended: 0.1)
- **max_tokens**: Maximum response length (default: 1000)
- **timeout**: Request timeout in seconds (default: 30.0)

### Recommended Models

1. **llama3.1:8b** - Best balance of performance and accuracy
2. **mistral:7b** - Fast and efficient for basic tasks
3. **codellama:7b** - Good for structured responses
4. **llama3:8b** - Alternative to llama3.1

## Testing

### Run Complete Test Suite

```bash
python test_ollama_integration.py
```

### Run Example Demo

```bash
python ollama_example.py
```

### Manual Testing

```python
# Test individual components
client = LLMClient(config)

# 1. Test connection
test_result = client.test_connection()

# 2. Get model info
model_info = client.get_model_info()

# 3. Test conflict detection
conflicts = client.detect_conflicts(context)

# 4. Test resolution generation
resolution = client.generate_resolution(context, conflicts)
```

## Error Handling

The client includes robust error handling:

- **Connection Errors**: Automatic retry with exponential backoff
- **Model Not Found**: Automatic model pulling attempt
- **Parse Errors**: Multiple JSON parsing strategies
- **Timeout Handling**: Configurable timeout with clear error messages
- **Fallback Responses**: Safe defaults when LLM fails

## Troubleshooting

### Common Issues

1. **"Cannot connect to Ollama"**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama if needed
   ollama serve
   ```

2. **"Model not available"**
   ```bash
   # Pull the model
   ollama pull llama3.1:8b
   
   # List available models
   ollama list
   ```

3. **"Generation test failed"**
   ```bash
   # Test model directly
   ollama run llama3.1:8b "Say hello"
   ```

4. **Slow responses**
   - Increase timeout in config
   - Use smaller models (e.g., mistral:7b)
   - Check system resources

### Performance Optimization

1. **Model Selection**
   - Use 7B models for faster responses
   - Use 8B+ models for better accuracy

2. **Configuration Tuning**
   - Lower temperature (0.1) for consistent outputs
   - Adjust max_tokens based on needs
   - Set appropriate timeout values

3. **Hardware Considerations**
   - More RAM allows larger models
   - GPU acceleration improves speed
   - SSD storage speeds up model loading

## Integration with CDR Pipeline

The LLM client integrates seamlessly with the CDR pipeline:

```python
from src.cdr.pipeline.cdr_pipeline import CDRPipeline

# Pipeline will use LLM client automatically
pipeline = CDRPipeline(config)
pipeline.run_simulation()
```

The LLM client is used for:
- **Primary conflict detection** alongside geometric detection
- **Resolution generation** with aviation-specific prompts  
- **Decision validation** and confidence scoring
- **Reasoning explanation** for ATC integration

## API Reference

### LLMClient Methods

- `__init__(config, memory_store=None)`: Initialize client
- `detect_conflicts(context)`: Detect potential conflicts
- `generate_resolution(context, conflict_info)`: Generate resolutions
- `test_connection()`: Test connectivity and functionality
- `get_model_info()`: Get current model information

### Response Formats

**Conflict Detection Response:**
```json
{
  "conflicts_detected": true,
  "conflicts": [
    {
      "intruder_callsign": "UAL456",
      "time_to_conflict_minutes": 5.2,
      "predicted_min_separation_nm": 3.1,
      "conflict_type": "head_on"
    }
  ],
  "assessment": "Head-on conflict detected with UAL456"
}
```

**Resolution Response:**
```json
{
  "resolution_type": "heading_change",
  "parameters": {
    "new_heading_deg": 110
  },
  "reasoning": "Right turn to avoid head-on conflict",
  "confidence": 0.85
}
```
