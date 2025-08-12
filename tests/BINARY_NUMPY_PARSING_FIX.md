# Binary Numpy Data Parsing Fix for BlueSky Integration

## Problem Solved
The BlueSky simulator's `POS` command returns binary numpy data instead of plain text, which was causing incorrect position parsing (e.g., extracting 8.0, 8.0 instead of real coordinates like 41.978, -87.904).

## Root Cause
- BlueSky POS command returns structured binary numpy arrays
- Previous parsing attempted to extract text values from binary data
- IEEE 754 floating point values need proper binary extraction using struct module

## Solution Implemented

### 1. Enhanced Binary Parsing Method
```python
def _parse_binary_aircraft_data(self, callsign: str, response: str, timestamp: float) -> Optional[AircraftState]:
    """Parse binary numpy data from BlueSky POS command"""
```

**Key Features:**
- Uses `struct.unpack('<f', data)` for IEEE 754 little-endian float extraction
- Validates coordinate ranges (latitude: -90 to 90, longitude: -180 to 180)
- Extracts multiple consecutive float values from binary data
- Robust error handling with fallback to cached states

### 2. Binary Data Extraction Helper
```python
def _extract_aircraft_data_from_binary(self, binary_data: bytes) -> Dict[str, float]:
    """Extract aircraft parameters from binary numpy array"""
```

**Functionality:**
- Searches for valid IEEE 754 float patterns in binary data
- Validates coordinate ranges for latitude/longitude identification
- Returns structured dictionary with aircraft parameters
- Handles malformed or incomplete binary data gracefully

### 3. Updated get_aircraft_states Method
```python
def get_aircraft_states(self, callsigns: Optional[List[str]] = None) -> Dict[str, AircraftState]:
    """Get real-time aircraft states using POS command with proper binary numpy parsing"""
```

**Improvements:**
- Always queries BlueSky for real-time positions (no cached state reliance)
- Uses improved binary numpy parsing for accurate position extraction
- Rate limiting (100ms pause every 5 requests) to avoid overwhelming BlueSky
- Fallback to cached states only when parsing fails
- Enhanced error handling and logging

## Technical Details

### Binary Data Structure
BlueSky POS command returns data in this format:
- Binary numpy array with IEEE 754 floating point values
- Little-endian byte ordering (`'<f'` format)
- Multiple consecutive float values representing aircraft parameters
- Position data typically includes: latitude, longitude, altitude, heading, speed

### Coordinate Validation
```python
# Latitude validation: -90 to 90 degrees
if -90 <= latitude <= 90:
    params['latitude'] = latitude

# Longitude validation: -180 to 180 degrees  
if -180 <= longitude <= 180:
    params['longitude'] = longitude
```

### Error Handling
- Graceful fallback to cached aircraft states when binary parsing fails
- Comprehensive logging for debugging binary data issues
- Timeout protection (3 seconds) for BlueSky communication
- Exception handling for malformed binary data

## Files Modified

### src/cdr/simulation/bluesky_client.py
- **Enhanced**: `_parse_binary_aircraft_data()` method with proper IEEE 754 parsing
- **Added**: `_extract_aircraft_data_from_binary()` helper method
- **Updated**: `get_aircraft_states()` method for system-wide binary parsing
- **Dependencies**: Uses existing `struct` and `math` imports

## Integration Points

### CDR Pipeline Integration
The CDR pipeline (`src/cdr/pipeline/cdr_pipeline.py`) automatically benefits from the improved parsing:
- Line 255: `current_states = self.bluesky_client.get_aircraft_states()`
- Line 312: `stable_states = self.bluesky_client.get_aircraft_states()`
- Line 510: `current_states = self.bluesky_client.get_aircraft_states()`
- Line 813: `fetched = self.bluesky_client.get_aircraft_states(needed)`
- Line 1185: `aircraft_states = self.bluesky_client.get_aircraft_states()`
- Line 1285: `current_states = self.bluesky_client.get_aircraft_states()`

### Test File Compatibility
All existing test files continue to work with the enhanced parsing:
- `test_aircraft_movement.py` - Will now get real coordinates instead of 8.0, 8.0
- `test_llm_state_integration.py` - Improved accuracy for LLM state tracking
- `debug_aircraft_states.py` - Enhanced debugging with real position data

## Expected Results

### Before Fix
```
⚠️ Position for TEST001: lat=8.0, lon=8.0 (incorrect binary parsing)
```

### After Fix
```
✅ LIVE position for TEST001: lat=41.978000, lon=-87.904000 (correct binary parsing)
```

## Usage Examples

### Basic Usage
```python
# Get all tracked aircraft with real-time positions
states = bluesky_client.get_aircraft_states()

# Get specific aircraft with accurate coordinates
states = bluesky_client.get_aircraft_states(['TEST001', 'FLIGHT123'])
```

### Manual Testing
```python
# Test binary parsing directly
response = client._send_command("POS TEST001", expect_response=True)
parsed_state = client._parse_binary_aircraft_data("TEST001", response, time.time())
print(f"Real coordinates: lat={parsed_state.latitude}, lon={parsed_state.longitude}")
```

## Performance Considerations

- **Rate Limiting**: 100ms pause every 5 requests prevents TCP coalescing
- **Timeout Protection**: 3-second timeout prevents hanging on slow responses
- **Efficient Binary Parsing**: Direct struct unpack instead of string manipulation
- **Cached Fallback**: Maintains system stability when parsing fails

## System-Wide Impact

This fix addresses the user's request to "update the get state function every where basis this" by:

1. **Centralizing Binary Parsing**: All position queries now use consistent binary parsing
2. **Maintaining API Compatibility**: Existing code continues to work without changes
3. **Improving Accuracy**: Real coordinates replace incorrect 8.0, 8.0 values
4. **Enhanced Reliability**: Robust error handling prevents system crashes

The fix ensures that aircraft movement tracking, CDR pipeline processing, and LLM state integration all receive accurate real-time position data from BlueSky's binary numpy responses.
