"""Streamlined LLM client for conflict detection and resolution with enhanced multi-aircraft and destination support"""

import json
import math
import random
import requests
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class LLMProvider(Enum):
    OLLAMA = "ollama"


@dataclass
class LLMConfig:
    """Streamlined LLM configuration"""
    provider: LLMProvider = LLMProvider.OLLAMA
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    timeout: float = 60.0
    temperature: float = 0.1
    max_retries: int = 3


@dataclass
class ConflictContext:
    """Context for conflict detection and resolution"""
    ownship_callsign: str
    ownship_state: Dict[str, Any]
    intruders: List[Dict[str, Any]]
    scenario_time: float
    lookahead_minutes: float = 10.0
    constraints: Optional[Dict[str, Any]] = None
    destination: Optional[Dict[str, Any]] = None


@dataclass
class ResolutionResponse:
    """LLM resolution response"""
    conflict_id: str
    aircraft1: str
    aircraft2: str
    resolution_type: str
    parameters: Dict[str, Any]
    reasoning: str
    confidence: float
    success: bool = True
    error_message: Optional[str] = None


# JSON Schema for validation
RESOLUTION_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "conflict_id", "aircraft1", "aircraft2", "resolution_type", "parameters", "reasoning", "confidence"],
    "properties": {
        "schema_version": {"const": "cdr.v1"},
        "conflict_id": {"type": "string"},
        "aircraft1": {"type": "string"},
        "aircraft2": {"type": "string"},
        "resolution_type": {"enum": ["heading_change", "altitude_change", "speed_change", "vertical_speed_change", "no_action", "direct_to", "reroute_via"]},
        "parameters": {"type": "object"},
        "reasoning": {"type": "string", "maxLength": 300},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    }
}

COMBINED_CDR_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "conflict_id", "conflicts_detected", "conflicts", "resolution"],
    "properties": {
        "schema_version": {"const": "cdr.v1"},
        "conflict_id": {"type": "string"},
        "conflicts_detected": {"type": "boolean"},
        "conflicts": {"type": "array"},
        "resolution": {"type": "object"}
    }
}


class PromptTemplate:
    """Enhanced prompt templates for multi-aircraft scenarios with fixed destinations"""
    
    # Enhanced combined detection and resolution prompt with multi-aircraft support
    COMBINED_CDR_PROMPT = """ATC assistant. Return ONLY complete JSON matching this exact schema:

{{
  "schema_version": "cdr.v1",
  "conflict_id": "{conflict_id}",
  "conflicts_detected": true|false,
  "conflicts": [
    {{
      "intruder_callsign": "string",
      "time_to_conflict_minutes": 0,
      "predicted_min_separation_nm": 0,
      "predicted_min_vertical_separation_ft": 0,
      "conflict_type": "head_on|crossing|overtaking|vertical|vertical_rate"
    }}
  ],
  "resolution": {{
    "resolution_type": "heading_change|altitude_change|speed_change|vertical_speed_change|direct_to|reroute_via|no_action",
    "parameters": {{
      // Include ONLY ONE parameter set for your chosen resolution_type:
      // For "heading_change": {{"new_heading_deg": 120}}
      // For "altitude_change": {{"target_altitude_ft": 36000}}  
      // For "speed_change": {{"target_speed_kt": 400}}
      // For "vertical_speed_change": {{"target_vertical_speed_fpm": 1500}}
      // For "direct_to": {{"waypoint_name": "DST1"}}
      // For "reroute_via": {{"via_waypoint": {{"name":"AVOID1","lat":42.32,"lon":-87.35}}, "resume_to_destination": true}}
      // For "no_action": {{}}
    }},
    "reasoning": "Brief explanation ≤200 chars",
    "confidence": 0.8
  }}
}}

CONFLICT DETECTION - MULTI-AIRCRAFT SCENARIO:
1. OWNSHIP: {ownship_callsign} at {ownship_lat:.4f},{ownship_lon:.4f} FL{ownship_fl} hdg={ownship_hdg}° spd={ownship_spd}kt VS={ownship_vs:+d}fpm

2. TRAFFIC ({num_intruders} aircraft):
{intruders_detailed_list}

3. CONFLICT ANALYSIS: Project {lookahead_minutes} min straight-line. If ANY aircraft separation <5NM horizontal OR <1000ft vertical = CONFLICT

FIXED DESTINATION (MUST REMAIN THE SAME):
- Name: {dest_name}
- Position: {dest_lat:.4f}, {dest_lon:.4f}
- Bearing/Distance from ownship: {dest_brg}°, {dest_dist_nm:.1f} NM
- CRITICAL: Ownship MUST reach this destination. Any resolution must allow continued progress toward destination.

RESOLUTION PRIORITY:
1. SAFETY: Resolve conflicts with 5+ NM / 1000+ ft separation
2. EFFICIENCY: Minimize deviation from direct route to destination
3. COORDINATION: Consider impact on multiple intruders

RULES:
- If CONFLICT DETECTED: set conflicts_detected=true, list ALL conflicts, propose resolution for OWNSHIP only
- If NO CONFLICT: set conflicts_detected=false, conflicts=[], resolution_type="no_action"
- ECHO EXACTLY: conflict_id="{conflict_id}", aircraft1="{ownship_callsign}"
- For conflicts, aircraft2=intruder_callsign of PRIMARY conflict (closest/most urgent)
- Keep reasoning ≤200 characters
- Resolution must consider OWNSHIP progress toward {dest_name}

JSON ONLY:"""
    
    # Dedicated resolver prompt for specific conflicts
    RESOLVER_PROMPT = """You are an enroute ATC assistant. Return ONLY valid JSON per the schema cdr.v1.

CONFLICT SITUATION:
- Conflict ID: {conflict_id}
- Ownship: {ownship_callsign} at {ownship_lat:.4f},{ownship_lon:.4f} FL{ownship_fl}, heading {ownship_heading}°, speed {ownship_speed} kt
- Intruder: {intruder_callsign} at {intruder_lat:.4f},{intruder_lon:.4f} {relative_bearing}° bearing, {distance_nm} NM, FL{intruder_fl}
- Time to conflict: {time_to_cpa} minutes

CONSTRAINTS:
- Lateral separation ≥ 5 NM, Vertical ≥ 1000 ft
- Minimum heading change: ±20°
- Avoid headings within ±15° of intruder bearing ({relative_bearing}°)

Return ONLY:
{{
  "schema_version": "cdr.v1",
  "conflict_id": "{conflict_id}",
  "aircraft1": "{ownship_callsign}",
  "aircraft2": "{intruder_callsign}",
  "resolution_type": "heading_change",
  "parameters": {{"new_heading_deg": 120}},
  "reasoning": "Turn away from collision course",
  "confidence": 0.8
}}"""


class StreamlinedLLMClient:
    """Enhanced LLM client with multi-aircraft and fixed destination support"""
    
    def __init__(self, config: LLMConfig, log_dir: Optional[Path] = None):
        self.config = config
        self.log_dir = log_dir or Path("logs/llm")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup client
        self._setup_client()
        self._warmup_llm()
    
    def _setup_client(self):
        """Setup Ollama client connection"""
        if self.config.provider == LLMProvider.OLLAMA:
            self._verify_ollama_connection()
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _verify_ollama_connection(self):
        """Verify Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            print("✅ Ollama connection verified")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.config.base_url}: {e}")
    
    def _warmup_llm(self):
        """Warm up the LLM with a simple request"""
        try:
            warmup_prompt = 'Return {"warmup":true} ONLY as JSON.'
            self._call_llm(warmup_prompt)
            print("✅ LLM warmed up successfully")
        except Exception as e:
            print(f"⚠️ LLM warmup failed: {e}")
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                
                # Ollama API call
                response = requests.post(
                    f"{self.config.base_url}/api/generate",
                    json={
                        "model": self.config.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.config.temperature,
                            "num_predict": 2048
                        }
                    },
                    timeout=self.config.timeout
                )
                
                response.raise_for_status()
                result = response.json()
                
                if "response" not in result:
                    raise ValueError(f"Invalid response format: {result}")
                
                elapsed = time.time() - start_time
                print(f"🤖 LLM call completed in {elapsed:.2f}s (attempt {attempt + 1})")
                
                return result["response"].strip()
                
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️ LLM call failed (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate JSON response"""
        try:
            # Find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)
            
            return parsed
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}\nResponse: {response}")
    
    def detect_and_resolve_conflicts(self, context: ConflictContext) -> Dict[str, Any]:
        """Enhanced conflict detection and resolution with multi-aircraft support"""
        conflict_id = f"CDR_{int(time.time() * 1000)}"
        
        # Prepare context data
        ownship = context.ownship_state
        intruders_detailed = self._format_intruders_detailed(context.intruders, ownship)
        
        # Destination info (fixed throughout scenario)
        dest_info = self._format_destination(context.destination, ownship)
        
        # Format enhanced prompt with multi-aircraft information
        prompt = PromptTemplate.COMBINED_CDR_PROMPT.format(
            conflict_id=conflict_id,
            ownship_callsign=context.ownship_callsign,
            ownship_lat=ownship.get('latitude', 0),
            ownship_lon=ownship.get('longitude', 0),
            ownship_fl=int(ownship.get('altitude', 35000) / 100),
            ownship_hdg=int(ownship.get('heading', 0)),
            ownship_spd=int(ownship.get('speed', 400)),
            ownship_vs=int(ownship.get('vertical_speed_fpm', 0)),
            num_intruders=len(context.intruders),
            intruders_detailed_list=intruders_detailed,
            lookahead_minutes=int(context.lookahead_minutes),
            dest_name=dest_info['name'],
            dest_lat=dest_info['lat'],
            dest_lon=dest_info['lon'],
            dest_brg=dest_info['bearing'],
            dest_dist_nm=dest_info['distance_nm']
        )
        
        # Call LLM
        response = self._call_llm(prompt)
        parsed = self._parse_json_response(response)
        
        # Log for debugging
        self._log_interaction(conflict_id, prompt, response, parsed)
        
        return parsed
    
    def resolve_specific_conflict(self, context: ConflictContext, intruder_callsign: str) -> ResolutionResponse:
        """Resolve a specific conflict between ownship and intruder"""
        conflict_id = f"RES_{int(time.time() * 1000)}"
        
        # Find the specific intruder
        intruder = None
        for traffic in context.intruders:
            if traffic.get('callsign') == intruder_callsign:
                intruder = traffic
                break
        
        if not intruder:
            return ResolutionResponse(
                conflict_id=conflict_id,
                aircraft1=context.ownship_callsign,
                aircraft2=intruder_callsign,
                resolution_type="no_action",
                parameters={},
                reasoning="Intruder not found",
                confidence=0.0,
                success=False,
                error_message=f"Intruder {intruder_callsign} not found"
            )
        
        # Calculate relative information
        ownship = context.ownship_state
        relative_info = self._calculate_relative_info(ownship, intruder)
        
        # Format prompt
        prompt = PromptTemplate.RESOLVER_PROMPT.format(
            conflict_id=conflict_id,
            ownship_callsign=context.ownship_callsign,
            ownship_lat=ownship.get('latitude', 0),
            ownship_lon=ownship.get('longitude', 0),
            ownship_fl=int(ownship.get('altitude', 35000) / 100),
            ownship_heading=int(ownship.get('heading', 0)),
            ownship_speed=int(ownship.get('speed', 400)),
            intruder_callsign=intruder_callsign,
            intruder_lat=intruder.get('latitude', 0),
            intruder_lon=intruder.get('longitude', 0),
            intruder_fl=int(intruder.get('altitude', 35000) / 100),
            relative_bearing=int(relative_info['bearing']),
            distance_nm=relative_info['distance_nm'],
            time_to_cpa=relative_info['time_to_cpa']
        )
        
        # Call LLM
        response = self._call_llm(prompt)
        parsed = self._parse_json_response(response)
        
        # Create response object
        return ResolutionResponse(
            conflict_id=parsed.get('conflict_id', conflict_id),
            aircraft1=parsed.get('aircraft1', context.ownship_callsign),
            aircraft2=parsed.get('aircraft2', intruder_callsign),
            resolution_type=parsed.get('resolution_type', 'no_action'),
            parameters=parsed.get('parameters', {}),
            reasoning=parsed.get('reasoning', ''),
            confidence=parsed.get('confidence', 0.0)
        )
    
    def _format_intruders_detailed(self, intruders: List[Dict[str, Any]], ownship: Dict[str, Any]) -> str:
        """Format intruders with detailed relative information for multi-aircraft scenarios"""
        if not intruders:
            return "   No traffic in range"
        
        formatted = []
        own_lat = ownship.get('latitude', 42.0)
        own_lon = ownship.get('longitude', -87.0)
        
        for i, intruder in enumerate(intruders[:8]):  # Support up to 8 intruders
            callsign = intruder.get('callsign', f'TFC{i+1}')
            int_lat = intruder.get('latitude', 0)
            int_lon = intruder.get('longitude', 0)
            alt = int(intruder.get('altitude', 35000))
            hdg = int(intruder.get('heading', 0))
            spd = int(intruder.get('speed', 400))
            vs = int(intruder.get('vertical_speed_fpm', 0))
            
            # Calculate relative information
            bearing = self._calculate_bearing(own_lat, own_lon, int_lat, int_lon)
            distance = self._calculate_distance(own_lat, own_lon, int_lat, int_lon)
            
            formatted.append(
                f"   {callsign}: {int_lat:.4f},{int_lon:.4f} FL{alt//100} hdg={hdg}° spd={spd}kt VS={vs:+d}fpm "
                f"({bearing:.0f}° {distance:.1f}NM from ownship)"
            )
        
        return "\n".join(formatted)
    
    def _format_destination(self, destination: Optional[Dict[str, Any]], ownship: Dict[str, Any]) -> Dict[str, Any]:
        """Format destination information with fixed destination support from SCAT data"""
        if not destination:
            # Generate a random destination 80-100 NM from ownship starting position
            own_lat = ownship.get('latitude', 42.0)
            own_lon = ownship.get('longitude', -87.0)
            
            # Random distance between 80-100 NM (as specified for SCAT scenarios)
            distance_nm = random.uniform(80, 100)
            bearing_deg = random.uniform(0, 360)
            
            # Calculate destination coordinates
            dest_lat, dest_lon = self._calculate_destination_from_bearing(own_lat, own_lon, distance_nm, bearing_deg)
            
            return {
                'name': f'DEST{random.randint(1000, 9999)}',
                'lat': dest_lat,
                'lon': dest_lon,
                'bearing': int(bearing_deg),
                'distance_nm': round(distance_nm, 1)
            }
        
        # Use provided destination (fixed throughout scenario)
        own_lat = ownship.get('latitude', 42.0)
        own_lon = ownship.get('longitude', -87.0)
        dest_lat = destination.get('latitude', own_lat + 1.0)
        dest_lon = destination.get('longitude', own_lon + 1.0)
        
        # Calculate current bearing and distance to fixed destination
        bearing = self._calculate_bearing(own_lat, own_lon, dest_lat, dest_lon)
        distance = self._calculate_distance(own_lat, own_lon, dest_lat, dest_lon)
        
        return {
            'name': destination.get('name', 'DEST'),
            'lat': dest_lat,
            'lon': dest_lon,
            'bearing': int(bearing),
            'distance_nm': round(distance, 1)
        }
    
    def generate_destination_from_scat_start(self, start_lat: float, start_lon: float,
                                           current_heading: Optional[float] = None,
                                           min_distance_nm: float = 80, max_distance_nm: float = 100) -> Dict[str, Any]:
        """Generate a fixed destination 80-100 NM from SCAT starting position, considering current heading"""
        # Random distance within range (based on requirements)
        distance_nm = random.uniform(min_distance_nm, max_distance_nm)
        
        if current_heading is not None:
            # Generate destination in the general direction of current heading (±45° spread)
            # This makes the destination more realistic as aircraft typically continue in their current direction
            heading_spread = 45  # degrees - allows some deviation but keeps general direction
            min_bearing = (current_heading - heading_spread) % 360
            max_bearing = (current_heading + heading_spread) % 360
            
            if min_bearing <= max_bearing:
                bearing_deg = random.uniform(min_bearing, max_bearing)
            else:
                # Handle wrap-around case (e.g., heading 350°, range 305°-35°)
                if random.random() < 0.5:
                    bearing_deg = random.uniform(min_bearing, 360)
                else:
                    bearing_deg = random.uniform(0, max_bearing)
            
            print(f"🧭 Generated destination considering heading {current_heading:.0f}°: bearing {bearing_deg:.0f}°, distance {distance_nm:.1f} NM")
        else:
            # Random bearing (0-360 degrees) if no heading provided
            bearing_deg = random.uniform(0, 360)
            print(f"🎲 Generated random destination: bearing {bearing_deg:.0f}°, distance {distance_nm:.1f} NM")
        
        # Calculate destination coordinates using great circle navigation
        dest_lat, dest_lon = self._calculate_destination_from_bearing(start_lat, start_lon, distance_nm, bearing_deg)
        
        # Generate unique destination name
        dest_name = f"DEST{random.randint(1000, 9999)}"
        
        return {
            'name': dest_name,
            'latitude': dest_lat,
            'longitude': dest_lon,
            'altitude': 35000,  # Standard cruise altitude
            'original_bearing': bearing_deg,
            'original_distance_nm': distance_nm,
            'based_on_heading': current_heading is not None
        }
    
    def _calculate_destination_from_bearing(self, start_lat: float, start_lon: float, 
                                          distance_nm: float, bearing_deg: float) -> Tuple[float, float]:
        """Calculate destination coordinates from start point, distance and bearing using great circle navigation"""
        # Convert to radians
        lat1_rad = math.radians(start_lat)
        lon1_rad = math.radians(start_lon)
        bearing_rad = math.radians(bearing_deg)
        
        # Earth radius in nautical miles
        earth_radius_nm = 3440.065
        
        # Angular distance
        angular_distance = distance_nm / earth_radius_nm
        
        # Calculate destination latitude
        lat2_rad = math.asin(
            math.sin(lat1_rad) * math.cos(angular_distance) +
            math.cos(lat1_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
        )
        
        # Calculate destination longitude
        lon2_rad = lon1_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat1_rad),
            math.cos(angular_distance) - math.sin(lat1_rad) * math.sin(lat2_rad)
        )
        
        # Convert back to degrees
        dest_lat = math.degrees(lat2_rad)
        dest_lon = math.degrees(lon2_rad)
        
        return dest_lat, dest_lon
    
    def _calculate_relative_info(self, ownship: Dict[str, Any], intruder: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate relative bearing, distance, and time to CPA"""
        own_lat = ownship.get('latitude', 42.0)
        own_lon = ownship.get('longitude', -87.0)
        int_lat = intruder.get('latitude', 42.0)
        int_lon = intruder.get('longitude', -87.0)
        
        bearing = self._calculate_bearing(own_lat, own_lon, int_lat, int_lon)
        distance = self._calculate_distance(own_lat, own_lon, int_lat, int_lon)
        
        # Enhanced time to CPA calculation considering velocities
        own_spd = ownship.get('speed', 400)
        int_spd = intruder.get('speed', 400)
        own_hdg = ownship.get('heading', 0)
        int_hdg = intruder.get('heading', 0)
        
        # Simple relative speed approximation
        relative_speed = math.sqrt((own_spd * math.cos(math.radians(own_hdg)) - int_spd * math.cos(math.radians(int_hdg)))**2 + 
                                 (own_spd * math.sin(math.radians(own_hdg)) - int_spd * math.sin(math.radians(int_hdg)))**2)
        
        time_to_cpa = (distance / max(relative_speed, 1)) * 60  # Convert to minutes
        
        return {
            'bearing': bearing,
            'distance_nm': round(distance, 2),
            'time_to_cpa': round(min(time_to_cpa, 30), 1)  # Cap at 30 minutes
        }
    
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing between two points"""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        return (bearing_deg + 360) % 360
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in nautical miles using haversine formula"""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat_rad = math.radians(lat2 - lat1)
        dlon_rad = math.radians(lon2 - lon1)
        
        a = math.sin(dlat_rad/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon_rad/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        # Earth radius in nautical miles
        earth_radius_nm = 3440.065
        
        return earth_radius_nm * c
    
    def _log_interaction(self, conflict_id: str, prompt: str, response: str, parsed: Dict[str, Any]):
        """Log LLM interaction with enhanced debugging - input/output for 1500 characters"""
        try:
            # Truncate prompt and response for debugging display (1500 chars each as requested)
            prompt_debug = prompt[:1500] + "..." if len(prompt) > 1500 else prompt
            response_debug = response[:1500] + "..." if len(response) > 1500 else response
            
            # Print debug information
            print(f"\n🔍 LLM Debug - Conflict ID: {conflict_id}")
            print(f"📝 Input ({len(prompt)} chars):")
            print("-" * 80)
            print(prompt_debug)
            print("-" * 80)
            print(f"🤖 Output ({len(response)} chars):")
            print("-" * 80)
            print(response_debug)
            print("-" * 80)
            print(f"📊 Parsed: {json.dumps(parsed, indent=2)[:500]}...")
            print("=" * 80)
            
            # Create comprehensive log entry
            log_entry = {
                'timestamp': time.time(),
                'conflict_id': conflict_id,
                'prompt': prompt,  # Full prompt in log file
                'raw_response': response,  # Full response in log file
                'parsed_response': parsed,
                'response_length': len(response),
                'prompt_length': len(prompt),
                'model_config': {
                    'provider': self.config.provider.value,
                    'model': self.config.model,
                    'temperature': self.config.temperature
                },
                'debug_truncated': {
                    'prompt_debug': prompt_debug,
                    'response_debug': response_debug
                },
                'interaction_metadata': {
                    'response_valid_json': isinstance(parsed, dict),
                    'has_conflicts': parsed.get('conflicts_detected', False) if isinstance(parsed, dict) else False,
                    'resolution_type': parsed.get('resolution', {}).get('resolution_type', 'unknown') if isinstance(parsed, dict) else 'unknown'
                }
            }
            
            # Save to timestamped log file
            timestamp_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            log_file = self.log_dir / f"{timestamp_str}_{conflict_id}.json"
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Log saved to: {log_file}")
                
        except Exception as e:
            print(f"⚠️ Failed to log interaction: {e}")
            # Fallback: at least print basic debug info
            try:
                print(f"🔍 Basic Debug - ID: {conflict_id}")
                print(f"📝 Input length: {len(prompt)}")
                print(f"🤖 Output length: {len(response)}")
                print(f"📊 Parsed type: {type(parsed)}")
            except:
                print("⚠️ Complete logging failure")


# Compatibility alias
LLMClient = StreamlinedLLMClient
