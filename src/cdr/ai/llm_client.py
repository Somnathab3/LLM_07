"""LLM client for conflict detection and resolution"""

import json
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: float = 30.0


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
    """Prompt template management"""
    
    DETECTOR_TEMPLATE = """
You are an expert air traffic controller analyzing aircraft conflicts.

SCENARIO:
- Ownship: {ownship_callsign} at {ownship_position}, heading {ownship_heading}°, FL{ownship_fl}, {ownship_speed} kts
- Intruders within surveillance range:
{intruder_list}

SEPARATION STANDARDS:
- Horizontal: 5 nautical miles minimum
- Vertical: 1000 feet minimum

TASK: Analyze for potential conflicts within the next {lookahead_minutes} minutes.

OUTPUT FORMAT (JSON only):
{{
  "conflicts_detected": boolean,
  "conflicts": [
    {{
      "intruder_callsign": "string",
      "time_to_conflict_minutes": number,
      "predicted_min_separation_nm": number,
      "predicted_min_vertical_separation_ft": number,
      "conflict_type": "head_on|crossing|overtaking|vertical"
    }}
  ],
  "assessment": "brief explanation"
}}
"""

    RESOLVER_TEMPLATE = """
You are an expert air traffic controller providing conflict resolution.

SITUATION:
- Ownship: {ownship_callsign} at {ownship_position}
- Current: heading {ownship_heading}°, FL{ownship_fl}, {ownship_speed} kts
- Conflict with: {intruder_callsign} at {relative_bearing}° and {distance_nm} NM

RESOLUTION PREFERENCES:
1. Heading change (temporary deviation)
2. Altitude change (if horizontal not feasible)
3. Speed adjustment (if minor conflict)

OUTPUT FORMAT (JSON only):
{{
  "resolution_type": "heading_change|altitude_change|speed_change",
  "parameters": {{
    "new_heading_deg": number,
    "target_altitude_ft": number,
    "target_speed_kt": number
  }},
  "reasoning": "brief explanation",
  "confidence": number
}}
"""


class LLMClient:
    """LLM client for conflict detection and resolution"""
    
    def __init__(self, config: LLMConfig, memory_store=None):
        self.config = config
        self.memory_store = memory_store
        self._setup_client()
    
    def _setup_client(self):
        """Setup LLM client based on provider"""
        if self.config.provider == LLMProvider.OPENAI:
            if not self.config.api_key:
                raise ValueError("OpenAI API key required")
        elif self.config.provider == LLMProvider.OLLAMA:
            try:
                response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
                if response.status_code != 200:
                    raise ConnectionError("Ollama server not responding")
            except requests.RequestException as e:
                raise ConnectionError(f"Cannot connect to Ollama: {e}")
    
    def detect_conflicts(self, context: ConflictContext) -> Dict[str, Any]:
        """Use LLM for conflict detection"""
        # Format intruder list
        intruder_descriptions = []
        for i, intruder in enumerate(context.intruders):
            desc = f"  {i+1}. {intruder['callsign']}: {intruder.get('position', 'N/A')}, " \
                   f"heading {intruder.get('heading', 0)}°, FL{int(intruder.get('altitude', 0)/100)}, " \
                   f"{intruder.get('speed', 0)} kts"
            intruder_descriptions.append(desc)
        
        prompt = PromptTemplate.DETECTOR_TEMPLATE.format(
            ownship_callsign=context.ownship_callsign,
            ownship_position=self._format_position(context.ownship_state),
            ownship_heading=context.ownship_state.get('heading', 0),
            ownship_fl=int(context.ownship_state.get('altitude', 0) / 100),
            ownship_speed=context.ownship_state.get('speed', 0),
            intruder_list='\n'.join(intruder_descriptions),
            lookahead_minutes=context.lookahead_minutes
        )
        
        response = self._call_llm(prompt)
        parsed = self._parse_json_response(response)
        
        return parsed or {"conflicts_detected": False, "conflicts": []}
    
    def generate_resolution(self, context: ConflictContext, 
                          conflict_info: Dict[str, Any]) -> ResolutionResponse:
        """Generate conflict resolution using LLM"""
        if not conflict_info.get('conflicts'):
            return ResolutionResponse(
                success=False,
                resolution_type="no_action",
                parameters={},
                reasoning="No conflicts to resolve"
            )
        
        primary_conflict = conflict_info['conflicts'][0]
        
        # Mock implementation for demonstration
        prompt = PromptTemplate.RESOLVER_TEMPLATE.format(
            ownship_callsign=context.ownship_callsign,
            ownship_position=self._format_position(context.ownship_state),
            ownship_heading=context.ownship_state.get('heading', 0),
            ownship_fl=int(context.ownship_state.get('altitude', 0) / 100),
            ownship_speed=context.ownship_state.get('speed', 0),
            intruder_callsign=primary_conflict.get('intruder_callsign', 'UNKNOWN'),
            relative_bearing=90,  # Mock value
            distance_nm=10  # Mock value
        )
        
        response = self._call_llm(prompt)
        parsed = self._parse_json_response(response)
        
        if parsed:
            return ResolutionResponse(
                success=True,
                resolution_type=parsed.get('resolution_type', 'heading_change'),
                parameters=parsed.get('parameters', {}),
                reasoning=parsed.get('reasoning'),
                confidence=parsed.get('confidence'),
                raw_response=response
            )
        
        return ResolutionResponse(
            success=False,
            resolution_type="no_action",
            parameters={},
            reasoning="Failed to parse LLM response"
        )
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt"""
        try:
            if self.config.provider == LLMProvider.OLLAMA:
                payload = {
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens
                    },
                    "stream": False
                }
                
                response = requests.post(
                    f"{self.config.base_url}/api/generate",
                    json=payload,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    return response.json()['response'].strip()
                else:
                    raise Exception(f"Ollama API error: {response.status_code}")
            
            # Mock response for other providers
            return '{"conflicts_detected": false, "conflicts": []}'
            
        except Exception as e:
            print(f"LLM call failed: {e}")
            return ""
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM"""
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            
            return json.loads(response)
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            return None
    
    def _format_position(self, aircraft_state: Dict[str, Any]) -> str:
        """Format aircraft position for prompt"""
        lat = aircraft_state.get('latitude', 0)
        lon = aircraft_state.get('longitude', 0)
        alt = aircraft_state.get('altitude', 0)
        return f"{lat:.4f}°, {lon:.4f}°, {alt:.0f} ft"
