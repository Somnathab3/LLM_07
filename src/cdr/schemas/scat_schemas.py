"""SCAT data schemas and validation classes"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import json


@dataclass
class ASTERIX_I062_010:
    """System Area Code and System Identification Code"""
    sac: int  # System Area Code
    sic: int  # System Identification Code


@dataclass
class ASTERIX_I062_015:
    """Service Identification"""
    service_identification: int


@dataclass
class ASTERIX_I062_070:
    """Time of Track (seconds since midnight)"""
    time_of_track: float


@dataclass
class ASTERIX_I062_105:
    """Geodetic Position (WGS-84)"""
    latitude: float   # degrees
    longitude: float  # degrees
    lat: Optional[float] = None  # Alternative field name
    lon: Optional[float] = None  # Alternative field name


@dataclass
class ASTERIX_I062_100:
    """Cartesian Position"""
    x_coordinate: float  # meters
    y_coordinate: float  # meters


@dataclass
class ASTERIX_I062_185:
    """Cartesian Velocity"""
    vx: float  # m/s
    vy: float  # m/s


@dataclass
class ASTERIX_I062_200:
    """Mode of Movement and Track Status"""
    track_status: Optional[str] = None
    adf: Optional[bool] = None
    long: Optional[int] = None
    trans: Optional[int] = None
    vert: Optional[int] = None


@dataclass
class ASTERIX_I062_295:
    """Track Data Ages"""
    track_data_ages: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ASTERIX_I062_136:
    """Measured Flight Level"""
    measured_flight_level: float  # FL (flight level)


@dataclass
class ASTERIX_I062_130:
    """Calculated Track Geometric Altitude"""
    altitude: float  # feet


@dataclass
class ASTERIX_I062_220:
    """Rate of Climb/Descent"""
    rocd: float  # feet per minute


@dataclass
class ASTERIX_I062_380_Subitem:
    """Individual subitem within I062/380"""
    altitude: Optional[float] = None
    sas: Optional[bool] = None
    source: Optional[int] = None
    ah: Optional[bool] = None
    am: Optional[bool] = None
    mv: Optional[bool] = None
    baro_vert_rate: Optional[float] = None
    ias: Optional[float] = None  # Indicated Airspeed
    mach: Optional[float] = None  # Mach number
    mag_hdg: Optional[float] = None  # Magnetic heading


@dataclass
class ASTERIX_I062_380:
    """Aircraft Derived Data"""
    aircraft_address: Optional[str] = None
    aircraft_identification: Optional[str] = None
    magnetic_heading: Optional[float] = None
    indicated_airspeed: Optional[float] = None
    true_airspeed: Optional[float] = None
    mach_number: Optional[float] = None
    barometric_pressure_setting: Optional[float] = None
    # Subitems (alternative format in actual data)
    subitem3: Optional[ASTERIX_I062_380_Subitem] = None
    subitem6: Optional[ASTERIX_I062_380_Subitem] = None
    subitem7: Optional[ASTERIX_I062_380_Subitem] = None
    subitem13: Optional[ASTERIX_I062_380_Subitem] = None
    subitem26: Optional[ASTERIX_I062_380_Subitem] = None
    subitem27: Optional[ASTERIX_I062_380_Subitem] = None


@dataclass
class ASTERIX_I062_040:
    """Track Number"""
    track_number: int


@dataclass
class ASTERIX_I062_080:
    """Mode 3/A Code"""
    mode_3a_code: str


@dataclass
class ASTERIX_I062_060:
    """Mode C Code"""
    mode_c_code: Union[int, str]


@dataclass
class ASTERIX_I062_245:
    """Target Identification"""
    target_identification: str


@dataclass
class SCATPlot:
    """Individual SCAT surveillance plot with full I062 support"""
    time_of_track: Union[str, float]
    source_id: Optional[str] = None
    
    # Core ASTERIX I062 fields
    i062_010: Optional[ASTERIX_I062_010] = None
    i062_015: Optional[ASTERIX_I062_015] = None
    i062_070: Optional[ASTERIX_I062_070] = None
    i062_105: Optional[ASTERIX_I062_105] = None
    i062_100: Optional[ASTERIX_I062_100] = None
    i062_185: Optional[ASTERIX_I062_185] = None
    i062_200: Optional[ASTERIX_I062_200] = None
    i062_295: Optional[ASTERIX_I062_295] = None
    i062_136: Optional[ASTERIX_I062_136] = None
    i062_130: Optional[ASTERIX_I062_130] = None
    i062_220: Optional[ASTERIX_I062_220] = None
    i062_380: Optional[ASTERIX_I062_380] = None
    i062_040: Optional[ASTERIX_I062_040] = None
    i062_080: Optional[ASTERIX_I062_080] = None
    i062_060: Optional[ASTERIX_I062_060] = None
    i062_245: Optional[ASTERIX_I062_245] = None


@dataclass
class Waypoint:
    """Flight plan waypoint"""
    name: str
    lat: float
    lon: float
    altitude: Optional[float] = None


@dataclass
class FlightPlanUpdate:
    """Flight plan update information"""
    callsign: str
    route: str
    rfl: int  # Requested Flight Level
    tas: int  # True Airspeed
    departure_time: Optional[str] = None
    waypoints: List[Waypoint] = field(default_factory=list)


@dataclass
class ATCClearanceUpdate:
    """Individual ATC clearance update"""
    time: Union[int, float, str]
    type: str  # altitude, direct, heading, speed
    value: Union[int, float, str]
    reason: Optional[str] = None


@dataclass
class ATCClearance:
    """Flight plan clearance information"""
    clearance_limit: Optional[str] = None
    cleared_flight_level: Optional[int] = None
    updates: List[ATCClearanceUpdate] = field(default_factory=list)


@dataclass
class PerformanceData:
    """Aircraft performance data"""
    aircraft_type: str
    mtow: Optional[float] = None  # Maximum Takeoff Weight (kg)
    oew: Optional[float] = None   # Operating Empty Weight (kg)
    max_fuel: Optional[float] = None  # Maximum fuel capacity (kg)
    wing_area: Optional[float] = None  # Wing area (m²)
    wing_span: Optional[float] = None  # Wing span (m)


@dataclass
class FlightPlanBase:
    """Basic flight plan information"""
    callsign: str
    aircraft_type: str
    adep: Optional[str] = None  # Departure airport
    ades: Optional[str] = None  # Destination airport
    flight_rules: Optional[str] = None
    wtc: Optional[str] = None  # Wake turbulence category
    equip_status_rvsm: Optional[bool] = None
    time_stamp: Optional[str] = None


@dataclass
class SCATData:
    """Complete SCAT dataset structure"""
    flight_id: Optional[str] = None
    callsign: Optional[str] = None
    aircraft_type: Optional[str] = None
    departure: Optional[str] = None
    arrival: Optional[str] = None
    id: Optional[int] = None
    
    # Core data sections
    plots: List[SCATPlot] = field(default_factory=list)
    fpl_plan_update: Optional[FlightPlanUpdate] = None
    fpl_clearance: Optional[ATCClearance] = None
    performance_data: Optional[PerformanceData] = None
    
    # Legacy format support
    fpl: Optional[Dict[str, Any]] = None
    centre_ctrl: Optional[List[Dict[str, Any]]] = None


@dataclass
class TrackPoint:
    """Individual trajectory point (enhanced version)"""
    timestamp: float
    latitude: float
    longitude: float
    altitude_ft: float
    heading_deg: float
    speed_kt: float
    vertical_rate_fpm: Optional[float] = None
    mach: Optional[float] = None
    
    # Additional derived fields
    x_coordinate: Optional[float] = None
    y_coordinate: Optional[float] = None
    true_airspeed_kt: Optional[float] = None
    flight_level: Optional[float] = None
    track_status: Optional[str] = None


@dataclass
class FlightPlan:
    """Enhanced flight plan information"""
    callsign: str
    route_string: str
    requested_flight_level: int
    cruise_tas: int
    waypoints: List[Waypoint] = field(default_factory=list)
    clearance_updates: List[ATCClearanceUpdate] = field(default_factory=list)
    
    # Additional metadata
    departure_airport: Optional[str] = None
    arrival_airport: Optional[str] = None
    aircraft_type: Optional[str] = None
    departure_time: Optional[str] = None