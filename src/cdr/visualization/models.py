"""Data models for visualization module."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


@dataclass
class Position:
    """Aircraft position"""
    latitude: float
    longitude: float
    altitude_ft: float
    heading_deg: float
    speed_kt: float
    timestamp: float = 0.0


@dataclass
class Aircraft:
    """Aircraft data model"""
    callsign: str
    aircraft_type: str = "B738"
    position: Optional[Position] = None
    status: str = "active"  # active, conflict, resolved
    color: Tuple[int, int, int] = (255, 255, 255)  # White by default
    
    def __post_init__(self):
        if self.position is None:
            self.position = Position(0, 0, 0, 0, 0)


@dataclass
class TrackPoint:
    """Single track point in trajectory"""
    timestamp: float
    latitude: float
    longitude: float
    altitude_ft: float
    heading_deg: float
    speed_kt: float
    callsign: str = ""


@dataclass
class Conflict:
    """Conflict data model"""
    aircraft1: str
    aircraft2: str
    detection_time: float
    cpa_time: float
    cpa_distance_nm: float
    severity: float
    status: str = "active"  # active, resolved, missed
    resolution_type: Optional[str] = None


@dataclass
class Resolution:
    """Resolution data model"""
    conflict_id: str
    aircraft_callsign: str
    resolution_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    success: bool = False


@dataclass
class Scenario:
    """Complete scenario data"""
    scenario_id: str
    aircraft: List[Aircraft] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    resolutions: List[Resolution] = field(default_factory=list)
    trajectory_points: List[TrackPoint] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_aircraft_by_callsign(self, callsign: str) -> Optional[Aircraft]:
        """Get aircraft by callsign"""
        for aircraft in self.aircraft:
            if aircraft.callsign == callsign:
                return aircraft
        return None
    
    def get_active_conflicts_at_time(self, timestamp: float) -> List[Conflict]:
        """Get conflicts active at given timestamp"""
        active_conflicts = []
        for conflict in self.conflicts:
            if (conflict.detection_time <= timestamp <= conflict.cpa_time and 
                conflict.status == "active"):
                active_conflicts.append(conflict)
        return active_conflicts
    
    def get_trajectory_for_aircraft(self, callsign: str) -> List[TrackPoint]:
        """Get trajectory points for specific aircraft"""
        return [tp for tp in self.trajectory_points if tp.callsign == callsign]


@dataclass 
class VisualizationFrame:
    """Single frame of visualization data"""
    timestamp: float
    aircraft_states: List[Aircraft] = field(default_factory=list)
    active_conflicts: List[Conflict] = field(default_factory=list)
    recent_resolutions: List[Resolution] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_aircraft_from_dict(data: Dict[str, Any]) -> Aircraft:
    """Create Aircraft object from dictionary data"""
    position = None
    if all(key in data for key in ['latitude', 'longitude', 'altitude_ft', 'heading_deg', 'speed_kt']):
        position = Position(
            latitude=data['latitude'],
            longitude=data['longitude'], 
            altitude_ft=data['altitude_ft'],
            heading_deg=data['heading_deg'],
            speed_kt=data['speed_kt'],
            timestamp=data.get('timestamp', 0.0)
        )
    
    return Aircraft(
        callsign=data['callsign'],
        aircraft_type=data.get('aircraft_type', 'B738'),
        position=position,
        status=data.get('status', 'active')
    )


def create_track_point_from_dict(data: Dict[str, Any]) -> TrackPoint:
    """Create TrackPoint object from dictionary data"""
    return TrackPoint(
        timestamp=data.get('timestamp', 0.0),
        latitude=data['latitude'],
        longitude=data['longitude'],
        altitude_ft=data['altitude_ft'],
        heading_deg=data['heading_deg'],
        speed_kt=data['speed_kt'],
        callsign=data.get('callsign', '')
    )
