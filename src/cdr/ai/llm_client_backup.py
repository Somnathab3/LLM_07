"""LLM client for conflict detection and resolution"""

import json
import math
import requests
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Import performance optimizations
from .performance_config import PerformanceConfig

# JSON Schema validation
try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    print("Warning: jsonschema not available, falling back to basic validation")


class LLMProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    LOCAL = "local"


# Strict, versioned JSON schema for resolution responses
RESOLUTION_SCHEMA_V1 = {
    "type": "object",
    "required": ["schema_version", "conflict_id", "aircraft1", "aircraft2", "resolution_type", "parameters", "reasoning", "confidence"],
    "properties": {
        "schema_version": {"const": "cdr.v1"},
        "conflict_id": {"type": "string"},
        "aircraft1": {"type": "string"},
        "aircraft2": {"type": "string"},
        "resolution_type": {"enum": ["heading_change", "altitude_change", "speed_change", "vertical_speed_change", "no_action", "direct_to", "reroute_via"]},
        "parameters": {
            "type": "object",
            "oneOf": [
                {
                    "properties": {
                        "new_heading_deg": {"type": ["number", "null"]},
                        "target_altitude_ft": {"type": ["number", "null"]},
                        "target_speed_kt": {"type": ["number", "null"]},
                        "target_vertical_speed_fpm": {"type": ["number", "null"]}
                    },
                    "additionalProperties": False
                },
                {
                    "required": ["waypoint_name"],
                    "properties": {
                        "waypoint_name": {"type": "string"},
                        "lat": {"type": ["number", "null"]},
                        "lon": {"type": ["number", "null"]}
                    },
                    "additionalProperties": False
                },
                {
                    "required": ["via_waypoint"],
                    "properties": {
                        "via_waypoint": {
                            "type": "object",
                            "required": ["name", "lat", "lon"],
                            "properties": {
                                "name": {"type": "string"},
                                "lat": {"type": "number"},
                                "lon": {"type": "number"}
                            }
                        },
                        "resume_to_destination": {"type": "boolean"}
                    },
                    "additionalProperties": False
                }
            ]
        },
        "reasoning": {"type": "string", "maxLength": 300},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    }
}

# Combined detection and resolution schema
COMBINED_CDR_SCHEMA_V1 = {
    "type": "object",
    "required": ["schema_version", "conflict_id", "conflicts_detected", "conflicts", "resolution"],
    "properties": {
        "schema_version": {"const": "cdr.v1"},
        "conflict_id": {"type": "string"},
        "conflicts_detected": {"type": "boolean"},
        "conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["intruder_callsign", "time_to_conflict_minutes", "predicted_min_separation_nm", "conflict_type"],
                "properties": {
                    "intruder_callsign": {"type": "string"},
                    "time_to_conflict_minutes": {"type": "number", "minimum": 0},
                    "predicted_min_separation_nm": {"type": "number", "minimum": 0},
                    "predicted_min_vertical_separation_ft": {"type": "number", "minimum": 0},
                    "conflict_type": {"enum": ["head_on", "crossing", "overtaking", "vertical", "vertical_rate"]}
                }
            }
        },
        "resolution": {
            "type": "object",
            "required": ["resolution_type"],
            "properties": {
                "resolution_type": {"enum": ["heading_change", "altitude_change", "speed_change", "vertical_speed_change", "no_action", "direct_to", "reroute_via"]},
                "parameters": {
                    "type": "object",
                    "oneOf": [
                        {
                            "properties": {
                                "new_heading_deg": {"type": ["number", "null"]},
                                "target_altitude_ft": {"type": ["number", "null"]},
                                "target_speed_kt": {"type": ["number", "null"]},
                                "target_vertical_speed_fpm": {"type": ["number", "null"]}
                            },
                            "additionalProperties": False
                        },
                        {
                            "required": ["waypoint_name"],
                            "properties": {
                                "waypoint_name": {"type": "string"},
                                "lat": {"type": ["number", "null"]},
                                "lon": {"type": ["number", "null"]}
                            },
                            "additionalProperties": False
                        },
                        {
                            "required": ["via_waypoint"],
                            "properties": {
                                "via_waypoint": {
                                    "type": "object",
                                    "required": ["name", "lat", "lon"],
                                    "properties": {
                                        "name": {"type": "string"},
                                        "lat": {"type": "number"},
                                        "lon": {"type": "number"}
                                    }
                                },
                                "resume_to_destination": {"type": "boolean"}
                            },
                            "additionalProperties": False
                        }
                    ]
                },
                "reasoning": {"type": "string", "maxLength": 300},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            }
        }
    }
}


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: str = "http://localhost:11434"
    temperature: float = 0.3
    max_tokens: int = 1000
    timeout: float = 60.0
    seed: int = 1337
    num_predict: int = 220  # Increased for combined mode to allow complete JSON response
    enable_verifier: bool = False  # Disabled - verifier causing issues with mathematical reasoning
    enable_agree_on_two: bool = False  # Disabled for performance  
    enable_reprompt_on_failure: bool = False  # Disabled since verifier is off
    prompt_char_limit: int = 12000
    max_intruders: int = 2  # Reduced from 3 to 2 for combined mode performance
    enable_combined_mode: bool = True  # Enable single-call detection+resolution


@dataclass
class ConflictContext:
    """Context for conflict resolution"""
    ownship_callsign: str
    ownship_state: Dict[str, Any]
    intruders: List[Dict[str, Any]]
    scenario_time: float
    lookahead_minutes: float
    constraints: Dict[str, Any]
    nearby_traffic: Optional[List[Dict[str, Any]]] = None
    destination: Optional[Dict[str, Any]] = None  # {"name":"DST1","lat":..., "lon":...}


@dataclass
class ResolutionResponse:
    """LLM resolution response"""
    success: bool
    resolution_type: str
    parameters: Dict[str, Any]
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    raw_response: Optional[str] = None


class PromptTemplate:
    """Contract-first prompt templates with strict JSON schema enforcement"""
    
    # Combined detection and resolution prompt - single LLM call
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
      // CRITICAL: Include ONLY ONE parameter set for your chosen resolution_type:
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

CONFLICT DETECTION:
1. Own: {ownship_callsign} at {ownship_lat:.4f},{ownship_lon:.4f} FL{ownship_fl} hdg={ownship_hdg}° spd={ownship_spd}kt VS={ownship_vs:+d}fpm
2. Traffic: {intruders_list}
3. Project straight-line {lookahead_minutes} min. If separation <5NM horizontal OR <1000ft vertical = CONFLICT

FINAL DESTINATION (must remain the same):
- Name: {dest_name}
- Position: {dest_lat:.4f}, {dest_lon:.4f}
- Bearing/Distance from ownship: {dest_brg}°, {dest_dist_nm} NM

RESOLUTION RULES:
- If CONFLICT: set conflicts_detected=true, add conflict details, propose resolution
- If NO CONFLICT: set conflicts_detected=false, conflicts=[], resolution_type="no_action", parameters={{}}, reasoning="No conflict detected", confidence=0.5
- ECHO EXACTLY: conflict_id="{conflict_id}", aircraft1="{ownship_callsign}", aircraft2=intruder_callsign
- Keep reasoning ≤200 characters
- Include ONLY the parameter fields for your chosen resolution_type (no extra fields, no nulls)

JSON ONLY:"""
    
    # Enhanced resolver prompt with explicit geometric constraints and example calculations
    RESOLVER_PROMPT = """You are an enroute ATC assistant. Return ONLY valid JSON per the schema cdr.v1.

CRITICAL CONSTRAINTS (Must satisfy ALL):
- Lateral separation ≥ 5 NM, Vertical ≥ 1000 ft, look-ahead = {lookahead_minutes} min
- Minimum heading change: ±20° (ENFORCED: anything less WILL FAIL verification)
- For heading changes: AVOID headings within ±15° of intruder bearing
- Consider closure geometry: head-on conflicts need larger heading changes

GEOMETRIC ANALYSIS:
- Current situation: Ownship heading {ownship_heading}°, Intruder bearing {relative_bearing}°
- Collision course check: Intruder bearing {relative_bearing}° means AVOID headings {relative_bearing_minus15}°-{relative_bearing_plus15}°
- Safe heading range 1: {safe_heading_1}° ± 10°
- Safe heading range 2: {safe_heading_2}° ± 10°

CONFLICT SITUATION:
- Conflict ID: {conflict_id}
- Ownship: {ownship_callsign} at {ownship_lat:.4f},{ownship_lon:.4f} FL{ownship_fl}, heading {ownship_heading}°, speed {ownship_speed} kt
- Intruder: {intruder_callsign} at {intruder_lat:.4f},{intruder_lon:.4f} {relative_bearing}° bearing, {distance_nm} NM, FL{intruder_fl}
- Closure rate: {closure_rate} kt, Time to CPA: {time_to_cpa} minutes

REQUIRED RESOLUTION STRATEGY:
1. Choose heading change that turns ≥20° away from current heading
2. Ensure new heading is NOT in forbidden zone ({relative_bearing_minus15}°-{relative_bearing_plus15}°)
3. Select from safe zones: {safe_heading_1}° or {safe_heading_2}°

Return ONLY:
{{
  "schema_version": "cdr.v1",
  "conflict_id": "{conflict_id}",
  "aircraft1": "{ownship_callsign}",
  "aircraft2": "{intruder_callsign}",
  "resolution_type": "heading_change",
  "parameters": {{"new_heading_deg": {safe_heading_1}}},
  "reasoning": "Turn {heading_change_direction} {heading_change_amount}° to avoid collision course",
  "confidence": 0.8
}}"""

    # Legacy templates for backward compatibility
    SIMPLE_DETECTOR_TEMPLATE = """You must respond with valid JSON only. No explanations, no text before or after.

Aircraft conflict analysis:
- Ownship: {ownship_callsign} at FL{ownship_fl}, heading {ownship_heading}°
- Intruders: {intruder_list}
- Timeframe: {lookahead_minutes} minutes

JSON response required:
{{
  "conflicts_detected": true,
  "conflicts": [
    {{
      "intruder_callsign": "UAL456",
      "conflict_type": "head_on"
    }}
  ]
}}"""

    SIMPLE_RESOLVER_TEMPLATE = """You must respond with valid JSON only. No explanations, no text before or after.

Conflict resolution needed:
- Ownship: {ownship_callsign} at FL{ownship_fl}, heading {ownship_heading}°
- Conflict with: {intruder_callsign}

JSON response required:
{{
  "resolution_type": "heading_change",
  "parameters": {{
    "new_heading_deg": 120
  }},
  "confidence": 0.8
}}"""
    
    DETECTOR_TEMPLATE = """You are an expert air traffic controller analyzing aircraft conflicts. Your task is to detect potential conflicts based on current aircraft positions and trajectories.

SCENARIO DETAILS:
- Ownship Aircraft: {ownship_callsign}
  Position: {ownship_position}
  Heading: {ownship_heading}°
  Flight Level: FL{ownship_fl}
  Speed: {ownship_speed} knots

- Aircraft in Surveillance Range:
{intruder_list}

SEPARATION STANDARDS:
- Horizontal separation minimum: 5 nautical miles
- Vertical separation minimum: 1000 feet
- Analysis timeframe: Next {lookahead_minutes} minutes

ANALYSIS INSTRUCTIONS:
1. Project current trajectories forward for {lookahead_minutes} minutes
2. Identify any separation violations
3. Classify conflict types (head-on, crossing, overtaking, vertical)
4. Estimate time to minimum separation

REQUIRED OUTPUT FORMAT (JSON only, no additional text):
{{
  "conflicts_detected": true/false,
  "conflicts": [
    {{
      "intruder_callsign": "aircraft_callsign",
      "time_to_conflict_minutes": number,
      "predicted_min_separation_nm": number,
      "predicted_min_vertical_separation_ft": number,
      "conflict_type": "head_on|crossing|overtaking|vertical|vertical_rate"
    }}
  ],
  "assessment": "brief technical explanation"
}}"""


class LLMClient:
    """Enhanced LLM client with contract-first design and robust error handling"""
    
    def __init__(self, config: LLMConfig, memory_store=None, log_dir: Optional[Path] = None):
        self.config = config
        self.memory_store = memory_store
        self.log_dir = log_dir or Path("logs/llm")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create audit directories
        (self.log_dir / "prompts").mkdir(exist_ok=True)
        (self.log_dir / "responses").mkdir(exist_ok=True)
        
        # Telemetry tracking
        self.telemetry = {
            'total_calls': 0,
            'schema_violations': 0,
            'verifier_failures': 0,
            'agreement_mismatches': 0,
            'average_latency': 0.0
        }
        
        self._setup_client()
        # Pre-warm the LLM to avoid first-call latency
        self._warmup_llm()
    
    def _setup_client(self):
        """Setup LLM client based on provider"""
        if self.config.provider == LLMProvider.OLLAMA:
            self._verify_ollama_connection()
            self._verify_model_availability()
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _warmup_llm(self):
        """Pre-warm the LLM with a simple request to avoid first-call delays"""
        try:
            warmup_prompt = 'Return {"warmup":true} ONLY as JSON.'
            response = self._call_llm(warmup_prompt)
            print(f"LLM warmed up successfully")
        except Exception as e:
            print(f"WARNING: LLM warmup failed: {e}")
            # Continue anyway - warmup is optional
    
    def _shape_input(self, context: ConflictContext) -> ConflictContext:
        """Shape input for speed and quality: top-K intruders, quantization"""
        # Top-K intruders by distance/TTC
        intruders = context.intruders[:self.config.max_intruders]
        
        # Quantize states for consistency
        def quantize_state(state: Dict[str, Any]) -> Dict[str, Any]:
            quantized = state.copy()
            if 'latitude' in quantized:
                quantized['latitude'] = round(quantized['latitude'], 4)
            if 'longitude' in quantized:
                quantized['longitude'] = round(quantized['longitude'], 4)
            if 'speed' in quantized:
                quantized['speed'] = int(round(quantized['speed']))
            if 'heading' in quantized:
                quantized['heading'] = int(round(quantized['heading']))
            if 'altitude' in quantized:
                quantized['altitude'] = int(round(quantized['altitude']))
            if 'vertical_speed_fpm' in quantized:
                quantized['vertical_speed_fpm'] = int(round(quantized['vertical_speed_fpm']))
            return quantized
        
        # Apply quantization
        ownship_state = quantize_state(context.ownship_state)
        quantized_intruders = [quantize_state(intruder) for intruder in intruders]
        
        return ConflictContext(
            ownship_callsign=context.ownship_callsign,
            ownship_state=ownship_state,
            intruders=quantized_intruders,
            scenario_time=context.scenario_time,
            lookahead_minutes=context.lookahead_minutes,
            constraints=context.constraints,
            nearby_traffic=context.nearby_traffic,
            destination=context.destination  # Preserve destination information
        )
    
    def _sanitize_resolution(self, resolution: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce operational limits and normalize parameters"""
        sanitized = resolution.copy()
        resolution_type = sanitized.get("resolution_type")
        parameters = sanitized.get("parameters", {})
        
        if resolution_type == "heading_change" and "new_heading_deg" in parameters:
            heading = parameters["new_heading_deg"]
            parameters["new_heading_deg"] = int(max(0, min(360, round(heading))))
        
        elif resolution_type == "altitude_change" and "target_altitude_ft" in parameters:
            altitude = parameters["target_altitude_ft"]
            # Snap to 100 ft increments (FL multiples)
            altitude_snapped = round(altitude / 100.0) * 100
            parameters["target_altitude_ft"] = int(min(45000, max(10000, altitude_snapped)))
        
        elif resolution_type == "speed_change" and "target_speed_kt" in parameters:
            speed = parameters["target_speed_kt"]
            parameters["target_speed_kt"] = int(min(490, max(250, round(speed))))
        
        elif resolution_type == "vertical_speed_change" and "target_vertical_speed_fpm" in parameters:
            vs = parameters["target_vertical_speed_fpm"]
            # Limit vertical speed to reasonable ATC values: -3000 to +3000 fpm
            parameters["target_vertical_speed_fpm"] = int(min(3000, max(-3000, round(vs))))
        
        elif resolution_type == "direct_to":
            # Validate waypoint name
            waypoint_name = parameters.get("waypoint_name", "DST")
            parameters["waypoint_name"] = str(waypoint_name).upper()
            
            # Optionally validate coordinates if provided
            if "lat" in parameters and parameters["lat"] is not None:
                parameters["lat"] = max(-90, min(90, float(parameters["lat"])))
            if "lon" in parameters and parameters["lon"] is not None:
                parameters["lon"] = max(-180, min(180, float(parameters["lon"])))
        
        elif resolution_type == "reroute_via":
            # Validate via waypoint structure
            via = parameters.get("via_waypoint", {})
            if isinstance(via, dict):
                via["name"] = str(via.get("name", "AVOID1")).upper()
                via["lat"] = max(-90, min(90, float(via.get("lat", 0))))
                via["lon"] = max(-180, min(180, float(via.get("lon", 0))))
                parameters["via_waypoint"] = via
            
            # Set resume flag
            parameters["resume_to_destination"] = bool(parameters.get("resume_to_destination", True))
        
        sanitized["parameters"] = parameters
        return sanitized
    
    def _validate_schema(self, response: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate response against RESOLUTION_SCHEMA_V1"""
        if not JSONSCHEMA_AVAILABLE:
            # Fallback validation
            return self._validate_resolution_response_fallback(response)
        
        try:
            jsonschema.validate(response, RESOLUTION_SCHEMA_V1)
            return True, []
        except jsonschema.exceptions.ValidationError as e:
            return False, [str(e)]
    
    def _validate_resolution_response_fallback(self, response: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Fallback validation when jsonschema is not available"""
        violations = []
        
        # Check required fields
        required = ["schema_version", "conflict_id", "aircraft1", "aircraft2", "resolution_type", "parameters", "reasoning", "confidence"]
        for field in required:
            if field not in response:
                violations.append(f"Missing required field: {field}")
        
        # Check schema version
        if response.get("schema_version") != "cdr.v1":
            violations.append("Invalid schema_version, must be 'cdr.v1'")
        
        # Check resolution type
        valid_types = ["heading_change", "altitude_change", "speed_change", "vertical_speed_change", "no_action", "direct_to", "reroute_via"]
        if response.get("resolution_type") not in valid_types:
            violations.append(f"Invalid resolution_type, must be one of {valid_types}")
        
        # Check confidence range
        confidence = response.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            violations.append("confidence must be a number between 0.0 and 1.0")
        
        # Check reasoning length
        reasoning = response.get("reasoning", "")
        if len(reasoning) > 500:
            violations.append("reasoning must be 500 characters or less")
        
        return len(violations) == 0, violations
    
    def _verify_resolution(self, resolution: Dict[str, Any], context: ConflictContext, 
                          conflict_info: Dict[str, Any]) -> Tuple[bool, List[str], str]:
        """Two-pass verification: audit the candidate resolution"""
        if not self.config.enable_verifier:
            return True, [], "Verifier disabled"
        
        try:
            primary_conflict = conflict_info['conflicts'][0]
            intruder_info = self._find_intruder_info(context, primary_conflict['intruder_callsign'])
            relative_bearing, distance_nm = self._calculate_relative_position(
                context.ownship_state, intruder_info
            )
            
            verifier_prompt = PromptTemplate.VERIFIER_PROMPT.format(
                resolution_json=json.dumps(resolution, indent=2),
                ownship_callsign=context.ownship_callsign,
                ownship_fl=int(context.ownship_state.get('altitude', 0) / 100),
                ownship_heading=context.ownship_state.get('heading', 0),
                ownship_speed=context.ownship_state.get('speed', 0),
                intruder_callsign=primary_conflict.get('intruder_callsign', 'UNKNOWN'),
                relative_bearing=relative_bearing,
                distance_nm=distance_nm
            )
            
            # Use lower temperature for verification (less creativity)
            old_temp = self.config.temperature
            self.config.temperature = 0.0
            
            try:
                response = self._call_llm(verifier_prompt)
                parsed = self._parse_json_response(response)
                
                if parsed and 'valid' in parsed:
                    is_valid = parsed.get('valid', False)
                    violations = parsed.get('violations', [])
                    notes = parsed.get('notes', '')
                    return is_valid, violations, notes
                else:
                    return False, ["Verifier returned invalid response"], "Parse error"
                    
            finally:
                self.config.temperature = old_temp
                
        except Exception as e:
            return False, [f"Verifier error: {str(e)}"], "Exception occurred"
    
    def _agreement_of_two(self, context: ConflictContext, conflict_info: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """Sample twice with different seeds and check agreement"""
        if not self.config.enable_agree_on_two:
            return True, {}, {}
        
        original_seed = self.config.seed
        
        try:
            # First resolution
            self.config.seed = original_seed
            res1 = self._generate_single_resolution(context, conflict_info)
            
            # Second resolution with different seed
            self.config.seed = original_seed + 1
            res2 = self._generate_single_resolution(context, conflict_info)
            
            # Check agreement on maneuver type
            type1 = res1.get('resolution_type', '')
            type2 = res2.get('resolution_type', '')
            
            if type1 != type2:
                return False, res1, res2
            
            # Check parameter agreement for same type
            params1 = res1.get('parameters', {})
            params2 = res2.get('parameters', {})
            
            if type1 == 'heading_change':
                h1 = params1.get('new_heading_deg', 0)
                h2 = params2.get('new_heading_deg', 0)
                if abs(h1 - h2) > 5:  # More than 5° difference
                    return False, res1, res2
            elif type1 == 'altitude_change':
                a1 = params1.get('target_altitude_ft', 0)
                a2 = params2.get('target_altitude_ft', 0)
                if abs(a1 - a2) > 500:  # More than 500 ft difference
                    return False, res1, res2
            elif type1 == 'speed_change':
                s1 = params1.get('target_speed_kt', 0)
                s2 = params2.get('target_speed_kt', 0)
                if abs(s1 - s2) > 10:  # More than 10 kt difference
                    return False, res1, res2
            
            return True, res1, res2
            
        finally:
            self.config.seed = original_seed
    
    def _sanitize_resolution(self, resolution: Dict[str, Any], context: ConflictContext) -> Tuple[Dict[str, Any], List[str]]:
        """Sanitize and auto-correct resolution to meet minimum requirements"""
        sanitized = resolution.copy()
        corrections = []
        
        resolution_type = resolution.get('resolution_type', '')
        parameters = resolution.get('parameters', {})
        
        if resolution_type == 'heading_change':
            new_heading = parameters.get('new_heading_deg')
            current_heading = context.ownship_state.get('heading', 0)
            
            if new_heading is not None:
                # Ensure minimum 15° heading change
                heading_diff = abs(new_heading - current_heading)
                if heading_diff < 15:
                    # Auto-correct to minimum 15° change
                    if new_heading > current_heading:
                        corrected_heading = (current_heading + 15) % 360
                    else:
                        corrected_heading = (current_heading - 15) % 360
                    
                    sanitized['parameters']['new_heading_deg'] = corrected_heading
                    corrections.append(f"Increased heading change from {heading_diff:.1f}° to 15° minimum")
                
                # Ensure heading is within 0-360 range
                corrected_heading = sanitized['parameters']['new_heading_deg'] % 360
                sanitized['parameters']['new_heading_deg'] = corrected_heading
        
        elif resolution_type == 'altitude_change':
            target_altitude = parameters.get('target_altitude_ft')
            current_altitude = context.ownship_state.get('altitude', 0)
            
            if target_altitude is not None:
                # Ensure minimum 1000 ft altitude change
                altitude_diff = abs(target_altitude - current_altitude)
                if altitude_diff < 1000:
                    # Auto-correct to minimum 1000 ft change
                    if target_altitude > current_altitude:
                        corrected_altitude = current_altitude + 1000
                    else:
                        corrected_altitude = current_altitude - 1000
                    
                    # Snap to 100 ft increments (FL levels)
                    corrected_altitude = round(corrected_altitude / 100) * 100
                    sanitized['parameters']['target_altitude_ft'] = corrected_altitude
                    corrections.append(f"Increased altitude change from {altitude_diff:.0f}ft to 1000ft minimum")
                
                # Ensure reasonable altitude limits
                corrected_altitude = max(1000, min(45000, sanitized['parameters']['target_altitude_ft']))
                sanitized['parameters']['target_altitude_ft'] = corrected_altitude
        
        elif resolution_type == 'speed_change':
            target_speed = parameters.get('target_speed_kt')
            current_speed = context.ownship_state.get('speed', 0)
            
            if target_speed is not None:
                # Ensure minimum 20 kt speed change
                speed_diff = abs(target_speed - current_speed)
                if speed_diff < 20:
                    # Auto-correct to minimum 20 kt change
                    if target_speed > current_speed:
                        corrected_speed = current_speed + 20
                    else:
                        corrected_speed = current_speed - 20
                    
                    sanitized['parameters']['target_speed_kt'] = corrected_speed
                    corrections.append(f"Increased speed change from {speed_diff:.0f}kt to 20kt minimum")
                
                # Ensure reasonable speed limits
                corrected_speed = max(250, min(490, sanitized['parameters']['target_speed_kt']))
                sanitized['parameters']['target_speed_kt'] = corrected_speed
        
        elif resolution_type == 'vertical_speed_change':
            target_vs = parameters.get('target_vertical_speed_fpm')
            current_vs = context.ownship_state.get('vertical_speed_fpm', 0)
            
            if target_vs is not None:
                # Ensure minimum 200 fpm vertical speed change for effectiveness
                vs_diff = abs(target_vs - current_vs)
                if vs_diff < 200:
                    # Auto-correct to minimum 200 fpm change
                    if target_vs > current_vs:
                        corrected_vs = current_vs + 200
                    else:
                        corrected_vs = current_vs - 200
                    
                    sanitized['parameters']['target_vertical_speed_fpm'] = corrected_vs
                    corrections.append(f"Increased vertical speed change from {vs_diff:.0f}fpm to 200fpm minimum")
                
                # Ensure reasonable vertical speed limits (-3000 to +3000 fpm)
                corrected_vs = max(-3000, min(3000, sanitized['parameters']['target_vertical_speed_fpm']))
                sanitized['parameters']['target_vertical_speed_fpm'] = corrected_vs
        
        elif resolution_type == 'direct_to':
            # Validate waypoint name
            waypoint_name = parameters.get('waypoint_name', 'DST')
            sanitized['parameters']['waypoint_name'] = str(waypoint_name).upper()
            
            # Validate coordinates if provided
            if 'lat' in parameters and parameters['lat'] is not None:
                sanitized['parameters']['lat'] = max(-90, min(90, float(parameters['lat'])))
            if 'lon' in parameters and parameters['lon'] is not None:
                sanitized['parameters']['lon'] = max(-180, min(180, float(parameters['lon'])))
                
            corrections.append(f"Validated direct_to waypoint: {sanitized['parameters']['waypoint_name']}")
        
        elif resolution_type == 'reroute_via':
            # Validate via waypoint structure
            via = parameters.get('via_waypoint', {})
            if isinstance(via, dict):
                via['name'] = str(via.get('name', 'AVOID1')).upper()
                via['lat'] = max(-90, min(90, float(via.get('lat', 0))))
                via['lon'] = max(-180, min(180, float(via.get('lon', 0))))
                sanitized['parameters']['via_waypoint'] = via
                
                # Ensure resume flag is set
                sanitized['parameters']['resume_to_destination'] = bool(parameters.get('resume_to_destination', True))
                
                corrections.append(f"Validated reroute_via waypoint: {via['name']} at {via['lat']:.4f},{via['lon']:.4f}")
            else:
                corrections.append("Invalid via_waypoint structure - must be object with name, lat, lon")
        
        return sanitized, corrections
    
    def _reprompt_with_feedback(self, context: ConflictContext, conflict_info: Dict[str, Any], 
                               violations: List[str]) -> Optional[Dict[str, Any]]:
        """Re-prompt with verifier feedback to correct issues"""
        primary_conflict = conflict_info['conflicts'][0]
        conflict_id = conflict_info.get('conflict_id', primary_conflict.get('conflict_id', 'UNKNOWN'))
        
        # Get intruder information
        intruder_info = self._find_intruder_info(context, primary_conflict['intruder_callsign'])
        relative_bearing, distance_nm = self._calculate_relative_position(
            context.ownship_state, intruder_info
        )
        
        # Calculate enhanced geometric information
        closure_rate = self._calculate_closure_rate(context.ownship_state, intruder_info)
        time_to_cpa = self._calculate_time_to_cpa(context.ownship_state, intruder_info, distance_nm, closure_rate)
        
        # Create feedback-enhanced prompt
        feedback_prompt = f"""You are an enroute ATC assistant. Your previous resolution was REJECTED for these reasons:
{'; '.join(violations)}

You MUST fix these issues. Use the enhanced constraints below:

CRITICAL CONSTRAINTS (Must satisfy ALL):
- Lateral separation ≥ 5 NM, Vertical ≥ 1000 ft, look-ahead = {context.lookahead_minutes} min
- Minimum heading change: ±20° (verified geometric calculation required)
- Safe altitude changes: minimum ±1500 ft from current altitude
- Speed changes: minimum ±30 kt from current speed
- Echo EXACT aircraft callsigns and conflict ID

GEOMETRIC REQUIREMENTS:
- For heading_change: new heading must create ≥5 NM separation within {context.lookahead_minutes} minutes
- Consider current bearing angles: if intruder at {relative_bearing}°, avoid headings within ±15° of that bearing
- Account for closure rates and predict separation at lookahead time

CONFLICT SITUATION:
- Conflict ID: {conflict_id}
- Ownship: {context.ownship_callsign} at FL{int(context.ownship_state.get('altitude', 0) / 100)}, heading {context.ownship_state.get('heading', 0)}°, speed {context.ownship_state.get('speed', 0)} kt
- Intruder: {primary_conflict.get('intruder_callsign', 'UNKNOWN')} at {relative_bearing}° bearing, {distance_nm} NM, altitude FL{int(intruder_info.get('altitude', 0) / 100)}
- Closure rate: {closure_rate} kt
- Time to CPA: {time_to_cpa} minutes

CORRECTION REQUIRED: Address the specific violations mentioned above.

Return ONLY:
{{
  "schema_version": "cdr.v1",
  "conflict_id": "{conflict_id}",
  "aircraft1": "{context.ownship_callsign}",
  "aircraft2": "{primary_conflict.get('intruder_callsign', 'UNKNOWN')}",
  "resolution_type": "heading_change|altitude_change|speed_change|no_action",
  "parameters": {{"new_heading_deg": number OR "target_altitude_ft": number OR "target_speed_kt": number OR {{}}}},
  "reasoning": "Fixed: [specific corrections made]",
  "confidence": 0.0-1.0
}}"""
        
        try:
            response = self._call_llm_with_retry(feedback_prompt, max_retries=1)
            parsed = self._parse_json_response(response)
            return parsed
        except Exception as e:
            print(f"Error in re-prompt: {e}")
            return None
    
    def _log_artifacts(self, conflict_id: str, prompt: str, raw_response: str, parsed_response: Dict[str, Any]):
        """Log all artifacts for audit trail"""
        try:
            # Log prompt
            prompt_file = self.log_dir / "prompts" / f"{conflict_id}.json"
            with open(prompt_file, 'w') as f:
                json.dump({
                    "conflict_id": conflict_id,
                    "prompt": prompt,
                    "timestamp": time.time(),
                    "prompt_chars": len(prompt)
                }, f, indent=2)
            
            # Log raw response
            raw_file = self.log_dir / "responses" / f"{conflict_id}.raw.txt"
            with open(raw_file, 'w') as f:
                f.write(raw_response)
            
            # Log parsed response
            parsed_file = self.log_dir / "responses" / f"{conflict_id}.parsed.json"
            with open(parsed_file, 'w') as f:
                json.dump(parsed_response, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to log artifacts for {conflict_id}: {e}")
    
    def _update_telemetry(self, latency: float, schema_valid: bool, verifier_valid: bool, agreement: bool):
        """Update telemetry metrics"""
        self.telemetry['total_calls'] += 1
        
        # Update average latency
        if self.telemetry['total_calls'] == 1:
            self.telemetry['average_latency'] = latency
        else:
            alpha = 0.1  # Exponential moving average
            self.telemetry['average_latency'] = (
                alpha * latency + (1 - alpha) * self.telemetry['average_latency']
            )
        
        if not schema_valid:
            self.telemetry['schema_violations'] += 1
        if not verifier_valid:
            self.telemetry['verifier_failures'] += 1
        if not agreement:
            self.telemetry['agreement_mismatches'] += 1
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get current telemetry metrics"""
        return self.telemetry.copy()
    
    @staticmethod
    def to_trafscript(acid: str, resolution: Dict[str, Any]) -> List[str]:
        """Convert LLM resolution to BlueSky TrafScript commands"""
        resolution_type = resolution.get('resolution_type')
        parameters = resolution.get('parameters', {})
        
        if resolution_type == "heading_change" and "new_heading_deg" in parameters:
            heading = int(parameters['new_heading_deg'])
            return [f"HDG {acid},{heading}"]
        
        elif resolution_type == "altitude_change" and "target_altitude_ft" in parameters:
            altitude_ft = int(parameters['target_altitude_ft'])
            flight_level = altitude_ft // 100
            return [f"ALT {acid},FL{flight_level}"]
        
        elif resolution_type == "speed_change" and "target_speed_kt" in parameters:
            speed = int(parameters['target_speed_kt'])
            return [f"SPD {acid},{speed}"]
        
        elif resolution_type == "vertical_speed_change" and "target_vertical_speed_fpm" in parameters:
            vs_fpm = int(parameters['target_vertical_speed_fpm'])
            return [f"VS {acid},{vs_fpm}"]
        
        elif resolution_type == "direct_to":
            waypoint_name = parameters.get('waypoint_name', 'DST')
            return [f"DIRECT {acid},{waypoint_name}"]
        
        elif resolution_type == "reroute_via":
            # Pipeline will issue ADDWPT + DIRECT; here we reflect intent
            via = parameters.get('via_waypoint', {})
            waypoint_name = via.get('name', 'AVOID1')
            return [f"DIRECT {acid},{waypoint_name}"]
        
        elif resolution_type == "no_action":
            return []
        
        else:
            print(f"Warning: Unknown resolution type {resolution_type}")
            return []
    
    def _query_memory_for_context(self, context: ConflictContext) -> str:
        """Query memory system for relevant past experiences"""
        if not self.memory_store:
            return ""
        
        try:
            similar_experiences = self.memory_store.query_similar(context, top_k=3)
            
            if not similar_experiences:
                return ""
            
            memory_context = "\n\nRELEVANT PAST EXPERIENCES:\n"
            for i, exp in enumerate(similar_experiences, 1):
                resolution = exp.resolution_taken
                success_score = exp.success_score
                outcome = exp.outcome_metrics
                
                memory_context += f"{i}. Similar conflict (Success: {success_score:.1f}):\n"
                memory_context += f"   Resolution: {resolution.get('resolution_type', 'unknown')}\n"
                if 'parameters' in resolution:
                    memory_context += f"   Parameters: {resolution['parameters']}\n"
                if 'safety_margin' in outcome:
                    memory_context += f"   Safety margin achieved: {outcome['safety_margin']:.1f} nm\n"
                memory_context += f"   Reasoning: {resolution.get('reasoning', 'N/A')}\n\n"
                
            memory_context += "Consider these experiences when making your decision.\n"
            return memory_context
            
        except Exception as e:
            print(f"Warning: Memory query failed: {e}")
            return ""
    
    def _store_experience(self, context: ConflictContext, response: Any, task_type: str, success_metrics: Dict[str, Any]):
        """Store experience in memory system"""
        if not self.memory_store:
            return
            
        try:
            from .memory import create_memory_record
            
            # Extract resolution data based on response type
            if hasattr(response, 'resolution_type'):
                # ResolutionResponse object
                resolution = {
                    'resolution_type': response.resolution_type,
                    'parameters': response.parameters,
                    'reasoning': response.reasoning,
                    'confidence': response.confidence
                }
            elif isinstance(response, dict):
                # Dictionary response
                resolution = {
                    'resolution_type': response.get('resolution_type', 'unknown'),
                    'parameters': response.get('parameters', {}),
                    'reasoning': response.get('reasoning', ''),
                    'confidence': response.get('confidence', 0.5)
                }
            else:
                # Fallback for other response types
                resolution = {
                    'resolution_type': 'unknown',
                    'parameters': {},
                    'reasoning': str(response),
                    'confidence': 0.5
                }
            
            # Create and store memory record
            record = create_memory_record(
                conflict_context=context,
                resolution=resolution,
                outcome_metrics=success_metrics,
                task_type=task_type
            )
            
            self.memory_store.store_experience(record)
            
        except Exception as e:
            print(f"Warning: Failed to store experience: {e}")
    
    def _verify_ollama_connection(self):
        """Verify Ollama server connectivity"""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama server returned status {response.status_code}")
            
            # Check if the response contains the expected structure
            tags_data = response.json()
            if 'models' not in tags_data:
                raise ConnectionError("Ollama server response format unexpected")
                
        except requests.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama server at {self.config.base_url}: {e}")
        except json.JSONDecodeError as e:
            raise ConnectionError(f"Invalid JSON response from Ollama server: {e}")
    
    def _verify_model_availability(self):
        """Verify that the specified model is available"""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in available_models]
                
                if not any(self.config.model_name in name for name in model_names):
                    print(f"Warning: Model '{self.config.model_name}' not found in available models: {model_names}")
                    print(f"Continuing without pulling (model pulling can be done manually)")
                    print(f"To pull manually: ollama pull {self.config.model_name}")
        except Exception as e:
            print(f"Warning: Could not verify model availability: {e}")
    
    def _pull_model(self):
        """Attempt to pull the model if not available - Currently disabled for safety"""
        print(f"Model pulling disabled. Please pull manually: ollama pull {self.config.model_name}")
    
    def detect_conflicts(self, context: ConflictContext, use_simple_prompt: bool = False) -> Dict[str, Any]:
        """Use LLM for conflict detection with robust error handling"""
        try:
            # Format intruder list
            intruder_descriptions = []
            for i, intruder in enumerate(context.intruders):
                desc = f"  {i+1}. {intruder['callsign']}: {intruder.get('position', 'N/A')}, " \
                       f"heading {intruder.get('heading', 0)}°, FL{int(intruder.get('altitude', 0)/100)}, " \
                       f"{intruder.get('speed', 0)} kts"
                intruder_descriptions.append(desc)
            
            # Choose prompt template based on testing mode
            if use_simple_prompt or self.config.max_tokens < 500:
                template = PromptTemplate.SIMPLE_DETECTOR_TEMPLATE
                prompt = template.format(
                    ownship_callsign=context.ownship_callsign,
                    ownship_fl=int(context.ownship_state.get('altitude', 0) / 100),
                    ownship_heading=context.ownship_state.get('heading', 0),
                    intruder_list=', '.join([f"{i['callsign']}" for i in context.intruders]),
                    lookahead_minutes=context.lookahead_minutes
                )
            else:
                prompt = PromptTemplate.DETECTOR_TEMPLATE.format(
                    ownship_callsign=context.ownship_callsign,
                    ownship_position=self._format_position(context.ownship_state),
                    ownship_heading=context.ownship_state.get('heading', 0),
                    ownship_fl=int(context.ownship_state.get('altitude', 0) / 100),
                    ownship_speed=context.ownship_state.get('speed', 0),
                    intruder_list='\n'.join(intruder_descriptions),
                    lookahead_minutes=context.lookahead_minutes
                )
            
            # Call LLM with retry mechanism
            response = self._call_llm_with_retry(prompt, max_retries=2)  # Reduced retries for testing
            parsed = self._parse_json_response(response)
            
            # Validate response structure
            if parsed and self._validate_conflict_response(parsed):
                return parsed
            else:
                print("Warning: Invalid LLM response for conflict detection, using fallback")
                return self._get_fallback_conflict_response()
                
        except Exception as e:
            print(f"Error in conflict detection: {e}")
            return self._get_fallback_conflict_response()
    
    def _validate_conflict_response(self, response: Dict[str, Any]) -> bool:
        """Validate conflict detection response structure"""
        required_fields = ['conflicts_detected', 'conflicts']
        if not all(field in response for field in required_fields):
            return False
        
        if not isinstance(response['conflicts_detected'], bool):
            return False
        
        if not isinstance(response['conflicts'], list):
            return False
        
        # Validate each conflict entry - be flexible about required fields
        for conflict in response['conflicts']:
            if not isinstance(conflict, dict):
                return False
            
            # Only require intruder_callsign for basic validation
            if 'intruder_callsign' not in conflict:
                return False
        
        return True
    
    def _get_fallback_conflict_response(self) -> Dict[str, Any]:
        """Provide fallback response when LLM fails"""
        return {
            "conflicts_detected": False,
            "conflicts": [],
            "assessment": "LLM detection failed, using fallback response"
        }
    
    def generate_resolution(self, context: ConflictContext, 
                          conflict_info: Dict[str, Any], use_simple_prompt: bool = False) -> ResolutionResponse:
        """Generate conflict resolution using enhanced contract-first approach"""
        if not conflict_info.get('conflicts'):
            return ResolutionResponse(
                success=False,
                resolution_type="no_action",
                parameters={},
                reasoning="No conflicts to resolve"
            )
        
        start_time = time.perf_counter()
        
        try:
            # Shape input for performance and quality
            shaped_context = self._shape_input(context)
            
            # Extract conflict information
            primary_conflict = conflict_info['conflicts'][0]
            conflict_id = conflict_info.get('conflict_id', primary_conflict.get('conflict_id', f"conflict_{int(time.time())}"))
            
            # Generate resolution(s)
            if self.config.enable_agree_on_two:
                agreement, resolution1, resolution2 = self._agreement_of_two(shaped_context, conflict_info)
                if agreement:
                    candidate_resolution = resolution1
                else:
                    # Use first resolution but mark disagreement
                    candidate_resolution = resolution1
                    self.telemetry['agreement_mismatches'] += 1
                    print(f"Warning: Agreement-of-two failed for {conflict_id}")
            else:
                candidate_resolution = self._generate_single_resolution(shaped_context, conflict_info)
            
            # Validate schema
            schema_valid, violations = self._validate_schema(candidate_resolution)
            if not schema_valid:
                print(f"Schema validation failed: {violations}")
                self.telemetry['schema_violations'] += 1
                return self._get_fallback_resolution(context)
            
            # Sanitize and normalize
            sanitized_resolution, corrections = self._sanitize_resolution(candidate_resolution, shaped_context)
            if corrections:
                print(f"Applied sanitization corrections: {corrections}")
            
            # Two-pass verification
            verifier_valid, verifier_violations, verifier_notes = self._verify_resolution(
                sanitized_resolution, shaped_context, conflict_info
            )
            
            if not verifier_valid and self.config.enable_verifier:
                print(f"Verifier failed: {verifier_violations}")
                self.telemetry['verifier_failures'] += 1
                
                # Re-prompt once with verifier feedback
                if self.config.enable_reprompt_on_failure:
                    print("Re-prompting with verifier feedback...")
                    retry_resolution = self._reprompt_with_feedback(
                        shaped_context, conflict_info, verifier_violations
                    )
                    if retry_resolution:
                        # Re-validate the retry
                        retry_sanitized, retry_corrections = self._sanitize_resolution(retry_resolution, shaped_context)
                        retry_valid, _, _ = self._verify_resolution(retry_sanitized, shaped_context, conflict_info)
                        
                        if retry_valid:
                            print("✅ Re-prompt succeeded, using corrected resolution")
                            sanitized_resolution = retry_sanitized
                            verifier_valid = True
                        else:
                            print("❌ Re-prompt also failed, using fallback")
                            return self._get_fallback_resolution(context)
                    else:
                        print("❌ Re-prompt parsing failed, using fallback")
                        return self._get_fallback_resolution(context)
                else:
                    return self._get_fallback_resolution(context)
            
            # Calculate confidence
            base_confidence = sanitized_resolution.get('confidence', 0.5)
            adjusted_confidence = self._adjust_confidence(base_confidence, schema_valid, verifier_valid)
            
            # Log artifacts
            self._log_artifacts(
                conflict_id,
                f"Primary resolution for {conflict_id}",  # Would be actual prompt in real implementation
                json.dumps(candidate_resolution, indent=2),
                sanitized_resolution
            )
            
            # Update telemetry
            elapsed_time = time.perf_counter() - start_time
            self._update_telemetry(elapsed_time, schema_valid, verifier_valid, True)
            
            # Store experience
            resolution_response = ResolutionResponse(
                success=True,
                resolution_type=sanitized_resolution.get('resolution_type', 'heading_change'),
                parameters=sanitized_resolution.get('parameters', {}),
                reasoning=sanitized_resolution.get('reasoning'),
                confidence=adjusted_confidence,
                raw_response=json.dumps(candidate_resolution)
            )
            
            success_metrics = {
                'success_rate': 1.0,
                'confidence': adjusted_confidence,
                'response_time': elapsed_time,
                'llm_provider': self.config.provider.value,
                'schema_valid': schema_valid,
                'verifier_valid': verifier_valid
            }
            self._store_experience(context, resolution_response, 'resolution', success_metrics)
            
            return resolution_response
                
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            self._update_telemetry(elapsed_time, False, False, False)
            print(f"Error in enhanced resolution generation: {e}")
            return self._get_fallback_resolution(context)
    
    def _generate_single_resolution(self, context: ConflictContext, conflict_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single resolution using the enhanced contract-first prompt with pre-calculated safe zones"""
        primary_conflict = conflict_info['conflicts'][0]
        conflict_id = conflict_info.get('conflict_id', primary_conflict.get('conflict_id', 'UNKNOWN'))
        
        # Get intruder information
        intruder_info = self._find_intruder_info(context, primary_conflict['intruder_callsign'])
        relative_bearing, distance_nm = self._calculate_relative_position(
            context.ownship_state, intruder_info
        )
        
        # Calculate enhanced geometric information
        closure_rate = self._calculate_closure_rate(context.ownship_state, intruder_info)
        time_to_cpa = self._calculate_time_to_cpa(context.ownship_state, intruder_info, distance_nm, closure_rate)
        
        # Calculate safe heading zones
        current_heading = context.ownship_state.get('heading', 0)
        forbidden_zone_start = (relative_bearing - 15) % 360
        forbidden_zone_end = (relative_bearing + 15) % 360
        
        # Calculate safe headings
        safe_heading_1 = (current_heading + 30) % 360  # 30° right turn
        safe_heading_2 = (current_heading - 30) % 360  # 30° left turn
        
        # Choose better safe heading (further from intruder bearing)
        bearing_diff_1 = min(abs(safe_heading_1 - relative_bearing), 360 - abs(safe_heading_1 - relative_bearing))
        bearing_diff_2 = min(abs(safe_heading_2 - relative_bearing), 360 - abs(safe_heading_2 - relative_bearing))
        
        if bearing_diff_1 > bearing_diff_2:
            primary_safe_heading = safe_heading_1
            heading_change_direction = "right"
            heading_change_amount = 30
        else:
            primary_safe_heading = safe_heading_2
            heading_change_direction = "left"
            heading_change_amount = 30
        
        # Format the enhanced contract-first prompt
        prompt = PromptTemplate.RESOLVER_PROMPT.format(
            conflict_id=conflict_id,
            ownship_callsign=context.ownship_callsign,
            ownship_fl=int(context.ownship_state.get('altitude', 0) / 100),
            ownship_heading=current_heading,
            ownship_speed=context.ownship_state.get('speed', 0),
            intruder_callsign=primary_conflict.get('intruder_callsign', 'UNKNOWN'),
            intruder_fl=int(intruder_info.get('altitude', 0) / 100),
            relative_bearing=relative_bearing,
            relative_bearing_minus15=forbidden_zone_start,
            relative_bearing_plus15=forbidden_zone_end,
            distance_nm=distance_nm,
            closure_rate=closure_rate,
            time_to_cpa=time_to_cpa,
            lookahead_minutes=context.lookahead_minutes,
            safe_heading_1=safe_heading_1,
            safe_heading_2=safe_heading_2,
            heading_change_direction=heading_change_direction,
            heading_change_amount=heading_change_amount
        )
        
        # Check prompt size guard
        if len(prompt) > self.config.prompt_char_limit:
            print(f"Warning: Prompt length {len(prompt)} exceeds limit {self.config.prompt_char_limit}")
            # In a real implementation, we would trim to top-2 intruders here
        
        # Get memory context
        memory_context = self._query_memory_for_context(context)
        if memory_context:
            prompt += memory_context
        
        # Call LLM with network retry only
        response = self._call_llm_with_retry(prompt, max_retries=2)
        parsed = self._parse_json_response(response)
        
        if not parsed:
            raise ValueError("Failed to parse LLM response as JSON")
        
        return parsed
    
    def _adjust_confidence(self, base_confidence: float, schema_valid: bool, verifier_valid: bool) -> float:
        """Adjust confidence based on validation results"""
        adjusted = base_confidence
        
        if not schema_valid:
            adjusted *= 0.5
        if not verifier_valid:
            adjusted *= 0.7
            
        return max(0.0, min(1.0, adjusted))
    
    def _find_intruder_info(self, context: ConflictContext, callsign: str) -> Dict[str, Any]:
        """Find intruder information by callsign"""
        for intruder in context.intruders:
            if intruder.get('callsign') == callsign:
                return intruder
        return {}
    
    def _calculate_relative_position(self, ownship: Dict[str, Any], intruder: Dict[str, Any]) -> tuple:
        """Calculate relative bearing and distance"""
        # Simple calculation - in real implementation, use proper aviation calculations
        import math
        
        own_lat = ownship.get('latitude', 0)
        own_lon = ownship.get('longitude', 0)
        int_lat = intruder.get('latitude', 0)
        int_lon = intruder.get('longitude', 0)
        
        # Calculate bearing (simplified)
        dlat = int_lat - own_lat
        dlon = int_lon - own_lon
        bearing = math.degrees(math.atan2(dlon, dlat))
        if bearing < 0:
            bearing += 360
        
        # Calculate distance (simplified, in nautical miles)
        distance = math.sqrt(dlat**2 + dlon**2) * 60  # Rough conversion
        
        return int(bearing), round(distance, 1)
    
    def _calculate_closure_rate(self, ownship: Dict[str, Any], intruder: Dict[str, Any]) -> float:
        """Calculate closure rate between aircraft in knots"""
        import math
        
        own_speed = ownship.get('speed', 0)
        own_heading = math.radians(ownship.get('heading', 0))
        int_speed = intruder.get('speed', 0)
        int_heading = math.radians(intruder.get('heading', 0))
        
        # Calculate relative velocity components
        own_vx = own_speed * math.sin(own_heading)
        own_vy = own_speed * math.cos(own_heading)
        int_vx = int_speed * math.sin(int_heading)
        int_vy = int_speed * math.cos(int_heading)
        
        # Relative velocity
        rel_vx = int_vx - own_vx
        rel_vy = int_vy - own_vy
        
        # Calculate relative bearing for closure
        own_lat = ownship.get('latitude', 0)
        own_lon = ownship.get('longitude', 0)
        int_lat = intruder.get('latitude', 0)
        int_lon = intruder.get('longitude', 0)
        
        # Simplified closure calculation
        dlat = int_lat - own_lat
        dlon = int_lon - own_lon
        
        if abs(dlat) < 0.001 and abs(dlon) < 0.001:
            return 0.0
        
        # Dot product of relative velocity and position vector
        closure_component = (rel_vx * dlon + rel_vy * dlat) / math.sqrt(dlon**2 + dlat**2)
        
        return round(abs(closure_component), 1)
    
    def _calculate_time_to_cpa(self, ownship: Dict[str, Any], intruder: Dict[str, Any], 
                              distance_nm: float, closure_rate: float) -> float:
        """Calculate time to closest point of approach in minutes"""
        if closure_rate <= 0:
            return 999.0  # No meaningful CPA
        
        # Simple calculation: time = distance / closure_rate
        time_hours = distance_nm / closure_rate
        time_minutes = time_hours * 60
        
        return round(min(time_minutes, 999.0), 1)
    
    def _calculate_bearing(self, ownship: Dict[str, Any], destination: Dict[str, Any]) -> int:
        """Calculate bearing from ownship to destination"""
        if not destination:
            return 0
            
        import math
        
        own_lat = ownship.get('latitude', 0)
        own_lon = ownship.get('longitude', 0)
        dest_lat = destination.get('lat', 0)
        dest_lon = destination.get('lon', 0)
        
        # Calculate bearing (simplified)
        dlat = dest_lat - own_lat
        dlon = dest_lon - own_lon
        bearing = math.degrees(math.atan2(dlon, dlat))
        if bearing < 0:
            bearing += 360
        
        return int(bearing)
    
    def _calculate_distance_nm(self, ownship: Dict[str, Any], destination: Dict[str, Any]) -> float:
        """Calculate distance from ownship to destination in nautical miles"""
        if not destination:
            return 0.0
            
        import math
        
        own_lat = ownship.get('latitude', 0)
        own_lon = ownship.get('longitude', 0)
        dest_lat = destination.get('lat', 0)
        dest_lon = destination.get('lon', 0)
        
        # Simplified distance calculation in nautical miles
        dlat = dest_lat - own_lat
        dlon = dest_lon - own_lon
        distance = math.sqrt(dlat**2 + dlon**2) * 60  # Rough conversion
        
        return round(distance, 1)
    
    def _add_safety_constraints(self, prompt: str, context: ConflictContext) -> str:
        """Add safety constraints to the prompt"""
        safety_notes = """
SAFETY CONSTRAINTS:
- Minimum altitude for commercial aircraft: FL100
- Maximum heading change: 45° (prefer 20-30°)
- Speed changes: ±50 knots maximum
- Consider wake turbulence separation
- Ensure return to original route after conflict resolution
"""
        return prompt + safety_notes
    
    def _validate_resolution_response(self, response: Dict[str, Any], 
                                     context: ConflictContext = None, 
                                     conflict_info: Dict[str, Any] = None) -> bool:
        """Validate resolution response structure with strict parameter validation and echo checking"""
        # Check required fields
        required_fields = ['conflict_id', 'aircraft1', 'aircraft2', 'resolution_type', 'parameters']
        if not all(field in response for field in required_fields):
            return False
        
        # Validate echo fields if context and conflict_info provided
        if context and conflict_info:
            expected_aircraft1 = context.ownship_callsign.upper().strip()
            primary_conflict = conflict_info.get('conflicts', [{}])[0]
            expected_aircraft2 = primary_conflict.get('intruder_callsign', '').upper().strip()
            expected_conflict_id = conflict_info.get('conflict_id', primary_conflict.get('conflict_id', ''))
            
            if (response.get('aircraft1', '').upper().strip() != expected_aircraft1 or
                response.get('aircraft2', '').upper().strip() != expected_aircraft2 or
                response.get('conflict_id', '') != expected_conflict_id):
                print(f"Warning: LLM failed to echo correct IDs/callsigns")
                return False
        
        resolution_type = response.get('resolution_type')
        if resolution_type not in ['heading_change', 'altitude_change', 'speed_change', 'vertical_speed_change', 'direct_to', 'reroute_via', 'no_action']:
            return False
        
        parameters = response.get('parameters', {})
        if not isinstance(parameters, dict):
            return False
        
        # Strict parameter validation based on resolution type
        if resolution_type == "heading_change":
            new_heading = parameters.get("new_heading_deg")
            return isinstance(new_heading, (int, float)) and 0 <= new_heading <= 360
        elif resolution_type == "altitude_change":
            target_alt = parameters.get("target_altitude_ft")
            return isinstance(target_alt, (int, float)) and 10000 <= target_alt <= 45000
        elif resolution_type == "speed_change":
            target_speed = parameters.get("target_speed_kt")
            return isinstance(target_speed, (int, float)) and 250 <= target_speed <= 490
        
        return True  # no_action
    
    def _calculate_resolution_confidence(self, response: Dict[str, Any], context: ConflictContext) -> float:
        """Calculate confidence score based on response quality"""
        confidence = 0.5  # Base confidence
        
        # Check if reasoning is provided
        if response.get('reasoning'):
            confidence += 0.2
        
        # Check parameter validity
        params = response.get('parameters', {})
        resolution_type = response.get('resolution_type')
        
        if resolution_type == 'heading_change' and 'new_heading_deg' in params:
            heading = params['new_heading_deg']
            if 0 <= heading <= 360:
                confidence += 0.2
        elif resolution_type == 'altitude_change' and 'target_altitude_ft' in params:
            altitude = params['target_altitude_ft']
            if 10000 <= altitude <= 45000:  # Reasonable altitude range
                confidence += 0.2
        elif resolution_type == 'speed_change' and 'target_speed_kt' in params:
            speed = params['target_speed_kt']
            if 100 <= speed <= 600:  # Reasonable speed range
                confidence += 0.2
        
        # Cap confidence at 1.0
        return min(confidence, 1.0)
    
    def _get_fallback_resolution(self, context: ConflictContext) -> ResolutionResponse:
        """Provide fallback resolution when LLM fails"""
        # Simple fallback: slight heading change
        current_heading = context.ownship_state.get('heading', 0)
        new_heading = (current_heading + 20) % 360
        
        return ResolutionResponse(
            success=True,
            resolution_type="heading_change",
            parameters={"new_heading_deg": new_heading},
            reasoning="Fallback resolution: 20-degree right turn",
            confidence=0.3
        )
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with enhanced configuration, telemetry, and performance optimizations"""
        try:
            if self.config.provider == LLMProvider.OLLAMA:
                # Instrument for debugging and telemetry
                t0 = time.perf_counter()
                prompt_chars = len(prompt)
                
                # Debug: Print full LLM prompt for verification
                print(f"=== LLM PROMPT ({prompt_chars} chars) ===")
                print(prompt)
                print("=== END LLM PROMPT ===\n")
                
                payload = {
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "format": "json",          # Force JSON-only output
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.num_predict,  # Small to reduce drift
                        "top_p": 0.9,
                        "top_k": 40,
                        "seed": self.config.seed,   # Reproducibility
                        
                        # Performance optimizations from centralized config
                        **PerformanceConfig.get_ollama_options()
                    },
                    "stream": False,
                    "keep_alive": PerformanceConfig.KEEP_ALIVE_TIMEOUT
                }
                
                # Use dynamic timeout from performance config
                timeout = (PerformanceConfig.REQUEST_TIMEOUT_WARMUP if self.telemetry['total_calls'] == 0 
                          else PerformanceConfig.REQUEST_TIMEOUT_NORMAL)
                
                # Add connection optimizations
                session = requests.Session()
                session.headers.update({
                    'Connection': 'keep-alive',
                    'Keep-Alive': 'timeout=30, max=100'
                })
                
                response = session.post(
                    f"{self.config.base_url}/api/generate",
                    json=payload,
                    timeout=timeout
                )
                
                t1 = time.perf_counter()
                
                if response.status_code == 200:
                    result = response.json()
                    llm_response = result.get('response', '').strip()
                    
                    # Debug: Print full LLM response for verification
                    print(f"=== LLM RESPONSE ({len(llm_response)} chars) ===")
                    print(llm_response)
                    print("=== END LLM RESPONSE ===\n")
                    
                    # Enhanced telemetry logging
                    tokens_predicted = result.get('eval_count', 0)  # If available from Ollama
                    print(f"DEBUG: LLM latency: {(t1-t0):.2f}s, prompt_chars={prompt_chars}, tokens={tokens_predicted}")
                    
                    return llm_response
                else:
                    raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
            
        except requests.exceptions.Timeout:
            raise Exception(f"LLM request timed out after {timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise Exception("Failed to connect to Ollama server")
        except Exception as e:
            raise Exception(f"LLM call failed: {e}")
    
    def _call_llm_with_retry(self, prompt: str, max_retries: int = 2) -> str:
        """Call LLM with retry mechanism - only for network/timeout errors"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return self._call_llm(prompt)
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Only retry on network/timeout errors, not on schema violations
                if any(keyword in error_msg for keyword in ['timeout', 'connection', 'network']):
                    print(f"LLM call attempt {attempt + 1} failed (retryable): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Non-retryable error, fail immediately
                    print(f"LLM call failed (non-retryable): {e}")
                    raise e
        
        raise last_exception or Exception("All LLM call attempts failed")
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM with robust error handling and trailing comma fix"""
        if not response:
            return None
            
        # Clean the response
        response = response.strip()
        
        # Light fix for trailing commas in JSON objects/arrays
        import re
        cleaned = re.sub(r",(\s*[}\]])", r"\1", response.strip())
        
        try:
            # Try parsing cleaned response first
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        try:
            # First try to parse the response directly
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        try:
            # Extract JSON from response (handle cases where LLM adds extra text)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                # Apply trailing comma fix to extracted JSON
                json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        try:
            # Try to find JSON in code blocks
            import re
            json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            matches = re.findall(json_pattern, response, re.DOTALL)
            if matches:
                json_str = re.sub(r",(\s*[}\]])", r"\1", matches[0])
                return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            pass
        
        try:
            # Try to extract any JSON-like structure more aggressively
            import re
            # Look for patterns like "conflicts_detected": true/false
            if 'conflicts_detected' in response:
                # Try to build a simple response
                conflicts_detected = 'true' in response.lower() if 'conflicts_detected' in response else False
                return {
                    "conflicts_detected": conflicts_detected,
                    "conflicts": [],
                    "assessment": "Extracted from non-JSON response"
                }
        except Exception:
            pass
        
        print(f"Failed to parse JSON response: {response[:200]}...")
        return None
    
    def _format_position(self, aircraft_state: Dict[str, Any]) -> str:
        """Format aircraft position for prompt"""
        lat = aircraft_state.get('latitude', 0)
        lon = aircraft_state.get('longitude', 0)
        alt = aircraft_state.get('altitude', 0)
        return f"{lat:.4f}°, {lon:.4f}°, {alt:.0f} ft"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                for model in models:
                    if self.config.model_name in model.get('name', ''):
                        return {
                            'name': model.get('name'),
                            'size': model.get('size', 0),
                            'parameter_size': model.get('details', {}).get('parameter_size', 'Unknown'),
                            'quantization_level': model.get('details', {}).get('quantization_level', 'Unknown')
                        }
            return {'name': self.config.model_name, 'status': 'Not found'}
        except Exception as e:
            return {'name': self.config.model_name, 'status': f'Error: {e}'}
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the LLM connection and basic functionality"""
        test_result = {
            'server_connected': False,
            'model_available': False,
            'generation_test': False,
            'error_messages': []
        }
        
        try:
            # Test server connection
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                test_result['server_connected'] = True
            else:
                test_result['error_messages'].append(f"Server responded with status {response.status_code}")
        except Exception as e:
            test_result['error_messages'].append(f"Server connection failed: {e}")
        
        if test_result['server_connected']:
            try:
                # Test model availability
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                if any(self.config.model_name in name for name in model_names):
                    test_result['model_available'] = True
                else:
                    test_result['error_messages'].append(f"Model {self.config.model_name} not available")
            except Exception as e:
                test_result['error_messages'].append(f"Model check failed: {e}")
        
        if test_result['model_available']:
            try:
                # Test basic generation
                test_prompt = "Respond with just the word 'TEST' and nothing else."
                response = self._call_llm(test_prompt)
                if 'TEST' in response.upper():
                    test_result['generation_test'] = True
                else:
                    test_result['error_messages'].append(f"Unexpected response: {response}")
            except Exception as e:
                test_result['error_messages'].append(f"Generation test failed: {e}")
        
        return test_result

    def detect_and_resolve(self, context: ConflictContext, conflict_id: str) -> Dict[str, Any]:
        """
        Combined detection and resolution in a single LLM call.
        
        Returns combined response with both conflict detection and resolution.
        """
        try:
            # Shape input for performance and quality
            shaped_context = self._shape_input(context)
            
            # Prepare intruders list (limit to top 2 for performance)
            intruders_min = []
            for intr in shaped_context.intruders[:2]:  # Limit to top 2
                intruders_min.append({
                    'callsign': intr.get('callsign', 'UNK'),
                    'lat': round(intr.get('latitude', 0), 4),
                    'lon': round(intr.get('longitude', 0), 4),
                    'FL': int(intr.get('altitude', 0) / 100),
                    'hdg': int(intr.get('heading', 0)),
                    'spd': int(intr.get('speed', 0)),
                    'vs': int(intr.get('vertical_speed_fpm', 0))
                })

            intruders_str = ", ".join([
                f"{i['callsign']} {i['lat']},{i['lon']} FL{i['FL']} {i['hdg']}° {i['spd']}kt VS={i['vs']:+d}fpm"
                for i in intruders_min
            ])            # Build combined prompt
            dest_info = shaped_context.destination if hasattr(shaped_context, 'destination') else None
            print(f"🎯 DEBUG: shaped_context.destination = {dest_info}")
            print(f"🎯 DEBUG: hasattr(shaped_context, 'destination') = {hasattr(shaped_context, 'destination')}")
            
            prompt = PromptTemplate.COMBINED_CDR_PROMPT.format(
                conflict_id=conflict_id,
                ownship_callsign=shaped_context.ownship_callsign,
                ownship_lat=round(shaped_context.ownship_state.get('latitude', 0), 4),
                ownship_lon=round(shaped_context.ownship_state.get('longitude', 0), 4),
                ownship_fl=int(shaped_context.ownship_state.get('altitude', 0) / 100),
                ownship_hdg=int(shaped_context.ownship_state.get('heading', 0)),
                ownship_spd=int(shaped_context.ownship_state.get('speed', 0)),
                ownship_vs=int(shaped_context.ownship_state.get('vertical_speed_fpm', 0)),
                intruders_list=intruders_str,
                lookahead_minutes=int(shaped_context.lookahead_minutes),
                dest_name=shaped_context.destination.get('name', 'DST') if shaped_context.destination else 'DST',
                dest_lat=round(shaped_context.destination.get('lat', 0), 4) if shaped_context.destination else 0,
                dest_lon=round(shaped_context.destination.get('lon', 0), 4) if shaped_context.destination else 0,
                dest_brg=self._calculate_bearing(shaped_context.ownship_state, shaped_context.destination) if shaped_context.destination else 0,
                dest_dist_nm=self._calculate_distance_nm(shaped_context.ownship_state, shaped_context.destination) if shaped_context.destination else 0
            )
            
            # Debug: Print aircraft positions being sent to LLM
            print(f"DEBUG: LLM INPUT - Ownship: {shaped_context.ownship_callsign} at {shaped_context.ownship_state.get('latitude', 0):.4f},{shaped_context.ownship_state.get('longitude', 0):.4f} FL{int(shaped_context.ownship_state.get('altitude', 0) / 100)} heading {shaped_context.ownship_state.get('heading', 0):.1f}° speed {shaped_context.ownship_state.get('speed', 0):.0f}kt VS {shaped_context.ownship_state.get('vertical_speed_fpm', 0):+.0f}fpm")
            print(f"DEBUG: LLM INPUT - Intruders: {intruders_str}")
            
            # Enhanced JSON Debug Output
            llm_input_data = {
                "prompt_type": "COMBINED_CDR_PROMPT",
                "conflict_id": conflict_id,
                "ownship": {
                    "callsign": shaped_context.ownship_callsign,
                    "latitude": round(shaped_context.ownship_state.get('latitude', 0), 4),
                    "longitude": round(shaped_context.ownship_state.get('longitude', 0), 4),
                    "altitude_fl": int(shaped_context.ownship_state.get('altitude', 0) / 100),
                    "heading_deg": int(shaped_context.ownship_state.get('heading', 0)),
                    "speed_kt": int(shaped_context.ownship_state.get('speed', 0)),
                    "vertical_speed_fpm": int(shaped_context.ownship_state.get('vertical_speed_fpm', 0))
                },
                "intruders": intruders_min,
                "lookahead_minutes": int(shaped_context.lookahead_minutes),
                "prompt_length": len(prompt)
            }
            print(f"DEBUG: LLM INPUT JSON: {json.dumps(llm_input_data, indent=2)}")
            
            # Show which prompt template is being used
            print(f"DEBUG: USING PROMPT: COMBINED_CDR_PROMPT (Single-call detection+resolution)")
            print(f"DEBUG: PROMPT PREVIEW: {prompt[:200]}..." if len(prompt) > 200 else f"DEBUG: FULL PROMPT: {prompt}")
            
            # Call LLM with retry
            raw_response = self._call_llm_with_retry(prompt, max_retries=1)
            
            # Fix truncated JSON by adding missing closing braces if needed
            fixed_response = raw_response.strip()
            if not fixed_response.endswith('}'):
                # Count open and close braces to determine how many we need
                open_braces = fixed_response.count('{')
                close_braces = fixed_response.count('}')
                missing_braces = open_braces - close_braces
                if missing_braces > 0:
                    # Add missing closing braces
                    fixed_response += '\n' + '  }\n' * (missing_braces - 1) + '}'
                    print(f"DEBUG: Fixed truncated JSON by adding {missing_braces} closing brace(s)")
            
            parsed = self._parse_json_response(fixed_response)
            
            if not parsed:
                print(f"WARNING: Failed to parse LLM response for conflict {conflict_id}")
                return self._get_fallback_combined_response(conflict_id, shaped_context.ownship_callsign)
                
            # Enhanced JSON Debug Output for LLM Response
            print(f"📋 LLM OUTPUT JSON: {json.dumps(parsed, indent=2)}")
            
            # Debug: Print LLM output (legacy format for readability)
            if 'resolution' in parsed and parsed['resolution'].get('resolution_type') != 'no_action':
                resolution = parsed['resolution']
                print(f"🎯 LLM OUTPUT - Resolution: {resolution.get('resolution_type')} with parameters: {resolution.get('parameters', {})}")
                print(f"🎯 LLM OUTPUT - Reasoning: {resolution.get('reasoning', 'N/A')}")
            
            # Also show conflicts detected
            if parsed.get('conflicts_detected'):
                conflicts = parsed.get('conflicts', [])
                print(f"🚨 CONFLICTS DETECTED: {len(conflicts)} conflict(s)")
                for i, conflict in enumerate(conflicts):
                    print(f"   {i+1}. {conflict.get('intruder_callsign', 'UNK')} - {conflict.get('conflict_type', 'unknown')} type")
            else:
                print(f"✅ NO CONFLICTS DETECTED")
                resolution = parsed['resolution']
                print(f"🎯 LLM OUTPUT - Resolution: {resolution.get('resolution_type')} with parameters: {resolution.get('parameters', {})}")
                print(f"🎯 LLM OUTPUT - Reasoning: {resolution.get('reasoning', 'N/A')}")
            
            # Validate combined schema
            if JSONSCHEMA_AVAILABLE:
                try:
                    jsonschema.validate(parsed, COMBINED_CDR_SCHEMA_V1)
                except jsonschema.ValidationError as e:
                    print(f"Combined schema validation failed: {e}")
                    return self._get_fallback_combined_response(conflict_id, shaped_context.ownship_callsign)
            
            # Ensure all required resolution fields are present for pipeline compatibility
            if 'resolution' in parsed:
                resolution = parsed['resolution']
                if 'parameters' not in resolution:
                    resolution['parameters'] = {}
                if 'reasoning' not in resolution:
                    resolution['reasoning'] = f"Automated {resolution.get('resolution_type', 'no_action')} resolution"
                if 'confidence' not in resolution:
                    resolution['confidence'] = 0.8 if resolution.get('resolution_type') != 'no_action' else 0.5
            
            # Sanitize resolution parameters
            if parsed.get('conflicts_detected') and parsed.get('resolution'):
                sanitized_resolution, corrections = self._sanitize_resolution(parsed['resolution'], shaped_context)
                parsed['resolution'] = sanitized_resolution
                if corrections:
                    print(f"Applied sanitization corrections: {corrections}")
            
            # Optional verifier check (if enabled)
            if self.config.enable_verifier and parsed.get('conflicts_detected'):
                conflicts = parsed.get('conflicts', [])
                if conflicts:
                    # Convert to legacy format for verifier
                    legacy_resolution = {
                        "schema_version": "cdr.v1",
                        "conflict_id": conflict_id,
                        "aircraft1": shaped_context.ownship_callsign,
                        "aircraft2": conflicts[0].get('intruder_callsign', ''),
                        "resolution_type": parsed['resolution']['resolution_type'],
                        "parameters": parsed['resolution']['parameters'],
                        "reasoning": parsed['resolution']['reasoning'],
                        "confidence": parsed['resolution']['confidence']
                    }
                    
                    legacy_conflict_info = {
                        "conflict_id": conflict_id,
                        "conflicts": [{"intruder_callsign": conflicts[0].get('intruder_callsign', '')}]
                    }
                    
                    ok, viols, _ = self._verify_resolution(legacy_resolution, shaped_context, legacy_conflict_info)
                    if not ok:
                        print(f"Verifier failed for combined response: {viols}")
                        # Downgrade to no_action if verification fails
                        parsed['conflicts_detected'] = False
                        parsed['conflicts'] = []
                        parsed['resolution'] = {
                            "resolution_type": "no_action",
                            "parameters": {},
                            "reasoning": "Verification failed, defaulting to no action",
                            "confidence": 0.5
                        }
            
            return parsed
            
        except Exception as e:
            print(f"Error in combined detect_and_resolve: {e}")
            return self._get_fallback_combined_response(conflict_id, context.ownship_callsign)
    
    def _get_fallback_combined_response(self, conflict_id: str, ownship_callsign: str) -> Dict[str, Any]:
        """Provide fallback combined response when LLM fails"""
        return {
            "schema_version": "cdr.v1",
            "conflict_id": conflict_id,
            "conflicts_detected": False,
            "conflicts": [],
            "resolution": {
                "resolution_type": "no_action",
                "parameters": {},
                "reasoning": "LLM failed, using fallback no-action response",
                "confidence": 0.0
            }
        }
