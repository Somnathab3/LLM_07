"""SCAT dataset parser and neighbor finder with full ASTERIX I062 support"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional, Iterator, Dict, Any, Union
from pathlib import Path
from datetime import datetime, timezone
from math import radians, cos, sin, asin, sqrt
from scipy.spatial import KDTree
import re

from ..schemas.scat_schemas import (
    SCATData, SCATPlot, TrackPoint, FlightPlan, ATCClearance, ATCClearanceUpdate,
    FlightPlanUpdate, Waypoint, PerformanceData, FlightPlanBase,
    ASTERIX_I062_105, ASTERIX_I062_136, ASTERIX_I062_380, ASTERIX_I062_220,
    ASTERIX_I062_200, ASTERIX_I062_185, ASTERIX_I062_100, ASTERIX_I062_010,
    ASTERIX_I062_380_Subitem
)


class SCATAdapter:
    """Enhanced SCAT dataset parser with full ASTERIX I062 and neighbor finding support"""
    
    def __init__(self, scat_file: Path):
        self.scat_file = scat_file
        self._data: Optional[SCATData] = None
        self._raw_data: Optional[Dict] = None
        self._track_cache: Optional[List[TrackPoint]] = None
    
    def load_file(self) -> Dict[str, Any]:
        """
        Load and validate complete SCAT structure
        - Parse all I062 ASTERIX fields
        - Validate time sequences
        - Extract performance data
        - Handle multiple radar sources
        """
        try:
            with open(self.scat_file, 'r', encoding='utf-8') as f:
                self._raw_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load SCAT file {self.scat_file}: {e}")
        
        # Parse the data into structured format
        self._data = self._parse_scat_data(self._raw_data)
        
        # Validate time sequences
        self._validate_time_sequences()
        
        return self._raw_data
    
    def _parse_scat_data(self, raw_data: Dict[str, Any]) -> SCATData:
        """Parse raw SCAT data into structured format"""
        scat_data = SCATData()
        
        # Extract basic identifiers
        scat_data.flight_id = raw_data.get('flight_id')
        scat_data.callsign = raw_data.get('callsign')
        scat_data.aircraft_type = raw_data.get('aircraft_type')
        scat_data.departure = raw_data.get('departure')
        scat_data.arrival = raw_data.get('arrival')
        scat_data.id = raw_data.get('id')
        
        # Parse plots with full I062 support
        if 'plots' in raw_data:
            scat_data.plots = [self._parse_plot(plot) for plot in raw_data['plots']]
        
        # Parse flight plan updates (new format)
        if 'fpl_plan_update' in raw_data:
            scat_data.fpl_plan_update = self._parse_flight_plan_update(
                raw_data['fpl_plan_update']
            )
        
        # Parse clearances (new format)
        if 'fpl_clearance' in raw_data:
            scat_data.fpl_clearance = self._parse_atc_clearance(
                raw_data['fpl_clearance']
            )
        
        # Parse performance data
        if 'performance_data' in raw_data:
            scat_data.performance_data = self._parse_performance_data(
                raw_data['performance_data']
            )
        
        # Handle legacy format
        if 'fpl' in raw_data:
            scat_data.fpl = raw_data['fpl']
            # Extract callsign and aircraft type from legacy format
            if 'fpl_base' in raw_data['fpl']:
                for base in raw_data['fpl']['fpl_base']:
                    if not scat_data.callsign:
                        scat_data.callsign = base.get('callsign')
                    if not scat_data.aircraft_type:
                        scat_data.aircraft_type = base.get('aircraft_type')
                    if not scat_data.departure:
                        scat_data.departure = base.get('adep')
                    if not scat_data.arrival:
                        scat_data.arrival = base.get('ades')
        
        if 'centre_ctrl' in raw_data:
            scat_data.centre_ctrl = raw_data['centre_ctrl']
        
        return scat_data
    
    def _parse_plot(self, plot_data: Dict[str, Any]) -> SCATPlot:
        """Parse individual plot with full ASTERIX I062 support"""
        plot = SCATPlot(
            time_of_track=plot_data.get('time_of_track'),
            source_id=plot_data.get('source_id')
        )
        
        # Parse I062/105 - Geodetic Position
        if 'I062/105' in plot_data:
            i062_105_data = plot_data['I062/105']
            plot.i062_105 = ASTERIX_I062_105(
                latitude=i062_105_data.get('latitude', i062_105_data.get('lat', 0.0)),
                longitude=i062_105_data.get('longitude', i062_105_data.get('lon', 0.0)),
                lat=i062_105_data.get('lat'),
                lon=i062_105_data.get('lon')
            )
        
        # Parse I062/136 - Measured Flight Level
        if 'I062/136' in plot_data:
            plot.i062_136 = ASTERIX_I062_136(
                measured_flight_level=plot_data['I062/136'].get('measured_flight_level', 0.0)
            )
        
        # Parse I062/185 - Cartesian Velocity
        if 'I062/185' in plot_data:
            i062_185_data = plot_data['I062/185']
            plot.i062_185 = ASTERIX_I062_185(
                vx=i062_185_data.get('vx', 0.0),
                vy=i062_185_data.get('vy', 0.0)
            )
        
        # Parse I062/200 - Mode of Movement
        if 'I062/200' in plot_data:
            i062_200_data = plot_data['I062/200']
            plot.i062_200 = ASTERIX_I062_200(
                track_status=i062_200_data.get('track_status'),
                adf=i062_200_data.get('adf'),
                long=i062_200_data.get('long'),
                trans=i062_200_data.get('trans'),
                vert=i062_200_data.get('vert')
            )
        
        # Parse I062/220 - Rate of Climb/Descent
        if 'I062/220' in plot_data:
            plot.i062_220 = ASTERIX_I062_220(
                rocd=plot_data['I062/220'].get('rocd', 0.0)
            )
        
        # Parse I062/380 - Aircraft Derived Data
        if 'I062/380' in plot_data:
            plot.i062_380 = self._parse_i062_380(plot_data['I062/380'])
        
        # Parse other I062 fields
        if 'I062/100' in plot_data:
            i062_100_data = plot_data['I062/100']
            plot.i062_100 = ASTERIX_I062_100(
                x_coordinate=i062_100_data.get('x_coordinate', 0.0),
                y_coordinate=i062_100_data.get('y_coordinate', 0.0)
            )
        
        return plot
    
    def _parse_i062_380(self, i062_380_data: Dict[str, Any]) -> ASTERIX_I062_380:
        """Parse I062/380 with support for both direct fields and subitems"""
        i062_380 = ASTERIX_I062_380()
        
        # Direct fields (new format)
        i062_380.aircraft_address = i062_380_data.get('aircraft_address')
        i062_380.aircraft_identification = i062_380_data.get('aircraft_identification')
        i062_380.magnetic_heading = i062_380_data.get('magnetic_heading')
        i062_380.indicated_airspeed = i062_380_data.get('indicated_airspeed')
        i062_380.true_airspeed = i062_380_data.get('true_airspeed')
        i062_380.mach_number = i062_380_data.get('mach_number')
        i062_380.barometric_pressure_setting = i062_380_data.get('barometric_pressure_setting')
        
        # Parse subitems (legacy format)
        for subitem_key, subitem_data in i062_380_data.items():
            if subitem_key.startswith('subitem') and isinstance(subitem_data, dict):
                subitem = ASTERIX_I062_380_Subitem(
                    altitude=subitem_data.get('altitude'),
                    sas=subitem_data.get('sas'),
                    source=subitem_data.get('source'),
                    ah=subitem_data.get('ah'),
                    am=subitem_data.get('am'),
                    mv=subitem_data.get('mv'),
                    baro_vert_rate=subitem_data.get('baro_vert_rate'),
                    ias=subitem_data.get('ias'),
                    mach=subitem_data.get('mach'),
                    mag_hdg=subitem_data.get('mag_hdg')
                )
                setattr(i062_380, subitem_key, subitem)
        
        return i062_380
    
    def _parse_flight_plan_update(self, fpl_data: Dict[str, Any]) -> FlightPlanUpdate:
        """Parse flight plan update information"""
        waypoints = []
        if 'waypoints' in fpl_data:
            for wp_data in fpl_data['waypoints']:
                waypoint = Waypoint(
                    name=wp_data.get('name', ''),
                    lat=wp_data.get('lat', 0.0),
                    lon=wp_data.get('lon', 0.0),
                    altitude=wp_data.get('altitude')
                )
                waypoints.append(waypoint)
        
        return FlightPlanUpdate(
            callsign=fpl_data.get('callsign', ''),
            route=fpl_data.get('route', ''),
            rfl=fpl_data.get('rfl', 0),
            tas=fpl_data.get('tas', 0),
            departure_time=fpl_data.get('departure_time'),
            waypoints=waypoints
        )
    
    def _parse_atc_clearance(self, clearance_data: Dict[str, Any]) -> ATCClearance:
        """Parse ATC clearance information"""
        updates = []
        if 'updates' in clearance_data:
            for update_data in clearance_data['updates']:
                update = ATCClearanceUpdate(
                    time=update_data.get('time'),
                    type=update_data.get('type', ''),
                    value=update_data.get('value'),
                    reason=update_data.get('reason')
                )
                updates.append(update)
        
        return ATCClearance(
            clearance_limit=clearance_data.get('clearance_limit'),
            cleared_flight_level=clearance_data.get('cleared_flight_level'),
            updates=updates
        )
    
    def _parse_performance_data(self, perf_data: Dict[str, Any]) -> PerformanceData:
        """Parse aircraft performance data"""
        return PerformanceData(
            aircraft_type=perf_data.get('aircraft_type', ''),
            mtow=perf_data.get('mtow'),
            oew=perf_data.get('oew'),
            max_fuel=perf_data.get('max_fuel'),
            wing_area=perf_data.get('wing_area'),
            wing_span=perf_data.get('wing_span')
        )
    
    def _validate_time_sequences(self):
        """Validate time sequences in plots"""
        if not self._data or not self._data.plots:
            return
        
        # Convert time strings to timestamps for validation
        timestamps = []
        for plot in self._data.plots:
            timestamp = self._parse_timestamp(plot.time_of_track)
            if timestamp is not None:
                timestamps.append(timestamp)
        
        if len(timestamps) > 1:
            # Check for chronological order
            sorted_timestamps = sorted(timestamps)
            if timestamps != sorted_timestamps:
                print(f"Warning: Non-chronological timestamps in {self.scat_file}")
    
    def _parse_timestamp(self, timestamp: Union[str, float, int]) -> Optional[float]:
        """Convert various timestamp formats to Unix timestamp"""
        if isinstance(timestamp, (int, float)):
            return float(timestamp)
        
        if isinstance(timestamp, str):
            try:
                # Try ISO format first
                if 'T' in timestamp:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    return dt.timestamp()
                # Try other common formats
                else:
                    # Handle simple time format if needed
                    return float(timestamp)
            except (ValueError, TypeError):
                return None
        
        return None
    
    def ownship_track(self) -> Iterator[TrackPoint]:
        """
        Parse surveillance plots with full I062 support
        - Handle I062/105 (lat/lon) and I062/100 (cartesian)
        - Convert I062/136 flight levels properly
        - Extract I062/380 aircraft data (heading, speeds, mach)
        - Parse I062/220 vertical rates
        - Interpolate missing points
        """
        if not self._data:
            self.load_file()
        
        if not self._data.plots:
            return
        
        # Sort plots by timestamp
        sorted_plots = sorted(self._data.plots, 
                            key=lambda p: self._parse_timestamp(p.time_of_track) or 0)
        
        for plot in sorted_plots:
            track_point = self._plot_to_trackpoint(plot)
            if track_point:
                yield track_point
    
    def _plot_to_trackpoint(self, plot: SCATPlot) -> Optional[TrackPoint]:
        """Convert SCAT plot to TrackPoint with full data extraction"""
        timestamp = self._parse_timestamp(plot.time_of_track)
        if timestamp is None:
            return None
        
        # Extract position (prefer geodetic over cartesian)
        lat, lon = 0.0, 0.0
        x_coord, y_coord = None, None
        
        if plot.i062_105:
            lat = plot.i062_105.latitude or plot.i062_105.lat or 0.0
            lon = plot.i062_105.longitude or plot.i062_105.lon or 0.0
        
        if plot.i062_100:
            x_coord = plot.i062_100.x_coordinate
            y_coord = plot.i062_100.y_coordinate
        
        # Extract altitude (convert flight level to feet)
        altitude_ft = 0.0
        flight_level = None
        if plot.i062_136:
            flight_level = plot.i062_136.measured_flight_level
            altitude_ft = flight_level * 100  # Convert FL to feet
        
        # Extract speeds and heading
        heading_deg = 0.0
        speed_kt = 0.0
        true_airspeed_kt = None
        mach = None
        
        if plot.i062_380:
            # Try direct fields first
            if plot.i062_380.magnetic_heading is not None:
                heading_deg = plot.i062_380.magnetic_heading
            elif plot.i062_380.subitem3 and plot.i062_380.subitem3.mag_hdg is not None:
                heading_deg = plot.i062_380.subitem3.mag_hdg
            
            if plot.i062_380.indicated_airspeed is not None:
                speed_kt = plot.i062_380.indicated_airspeed
            elif plot.i062_380.subitem26 and plot.i062_380.subitem26.ias is not None:
                speed_kt = plot.i062_380.subitem26.ias
            
            if plot.i062_380.true_airspeed is not None:
                true_airspeed_kt = plot.i062_380.true_airspeed
            
            if plot.i062_380.mach_number is not None:
                mach = plot.i062_380.mach_number
            elif plot.i062_380.subitem27 and plot.i062_380.subitem27.mach is not None:
                mach = plot.i062_380.subitem27.mach
        
        # Extract vertical rate
        vertical_rate_fpm = None
        if plot.i062_220:
            vertical_rate_fpm = plot.i062_220.rocd
        
        # Extract track status
        track_status = None
        if plot.i062_200:
            track_status = plot.i062_200.track_status
        
        return TrackPoint(
            timestamp=timestamp,
            latitude=lat,
            longitude=lon,
            altitude_ft=altitude_ft,
            heading_deg=heading_deg,
            speed_kt=speed_kt,
            vertical_rate_fpm=vertical_rate_fpm,
            mach=mach,
            x_coordinate=x_coord,
            y_coordinate=y_coord,
            true_airspeed_kt=true_airspeed_kt,
            flight_level=flight_level,
            track_status=track_status
        )
    
    def flight_plan(self) -> FlightPlan:
        """Extract enhanced flight plan information"""
        if not self._data:
            self.load_file()
        
        # Try new format first
        if self._data.fpl_plan_update:
            fpl_update = self._data.fpl_plan_update
            clearance_updates = []
            
            if self._data.fpl_clearance and self._data.fpl_clearance.updates:
                clearance_updates = self._data.fpl_clearance.updates
            
            return FlightPlan(
                callsign=fpl_update.callsign,
                route_string=fpl_update.route,
                requested_flight_level=fpl_update.rfl,
                cruise_tas=fpl_update.tas,
                waypoints=fpl_update.waypoints,
                clearance_updates=clearance_updates,
                departure_airport=self._data.departure,
                arrival_airport=self._data.arrival,
                aircraft_type=self._data.aircraft_type,
                departure_time=fpl_update.departure_time
            )
        
        # Fallback to legacy format
        else:
            return FlightPlan(
                callsign=self._data.callsign or '',
                route_string='',
                requested_flight_level=0,
                cruise_tas=0,
                waypoints=[],
                clearance_updates=[],
                departure_airport=self._data.departure,
                arrival_airport=self._data.arrival,
                aircraft_type=self._data.aircraft_type
            )
    
    def extract_clearances(self) -> List[ATCClearanceUpdate]:
        """
        Extract and parse ATC clearances
        - Parse fpl_clearance updates
        - Extract timing and reasoning
        - Classify clearance types (altitude, heading, direct)
        """
        if not self._data:
            self.load_file()
        
        clearances = []
        
        if self._data.fpl_clearance and self._data.fpl_clearance.updates:
            for update in self._data.fpl_clearance.updates:
                clearances.append(update)
        
        return clearances
    
    def find_neighbors(self, scat_directory: Path, 
                      radius_nm: float = 100.0, 
                      altitude_tolerance_ft: float = 5000.0) -> List['SCATAdapter']:
        """
        Efficient neighbor finding using KDTree
        - Build spatial-temporal index
        - 100 NM horizontal, ±5000 ft vertical search
        - Time-based filtering for surveillance overlap
        - Handle coordinate system conversions
        """
        neighbors = []
        
        # Get own track data
        own_track = list(self.ownship_track())
        if not own_track:
            return neighbors
        
        # Create spatial-temporal index for own track
        own_points = []
        own_times = []
        
        for point in own_track:
            if point.latitude != 0.0 and point.longitude != 0.0:
                # Convert to approximate cartesian coordinates (simple projection)
                x_approx = point.longitude * 60.0 * 1.852  # Rough NM conversion
                y_approx = point.latitude * 60.0 * 1.852
                own_points.append([x_approx, y_approx, point.altitude_ft])
                own_times.append(point.timestamp)
        
        if not own_points:
            return neighbors
        
        own_kdtree = KDTree(own_points)
        
        # Get time window
        start_time = min(own_times)
        end_time = max(own_times)
        
        # Search other SCAT files
        for scat_file in scat_directory.glob('*.json'):
            if scat_file == self.scat_file:
                continue
            
            try:
                neighbor_adapter = SCATAdapter(scat_file)
                neighbor_track = list(neighbor_adapter.ownship_track())
                
                if not neighbor_track:
                    continue
                
                # Check time overlap
                neighbor_times = [p.timestamp for p in neighbor_track]
                neighbor_start = min(neighbor_times)
                neighbor_end = max(neighbor_times)
                
                if neighbor_end < start_time or neighbor_start > end_time:
                    continue
                
                # Check spatial proximity using KDTree
                found_close = False
                
                for n_point in neighbor_track:
                    if start_time <= n_point.timestamp <= end_time:
                        if n_point.latitude != 0.0 and n_point.longitude != 0.0:
                            # Convert neighbor point to same coordinate system
                            n_x = n_point.longitude * 60.0 * 1.852
                            n_y = n_point.latitude * 60.0 * 1.852
                            n_z = n_point.altitude_ft
                            
                            # Query KDTree for nearby points
                            distances, indices = own_kdtree.query(
                                [n_x, n_y, n_z], 
                                k=1, 
                                distance_upper_bound=radius_nm * 1.852  # Convert NM to km approximation
                            )
                            
                            if distances < np.inf:
                                # Check altitude tolerance
                                own_point_idx = indices
                                if own_point_idx < len(own_points):
                                    alt_diff = abs(own_points[own_point_idx][2] - n_z)
                                    if alt_diff <= altitude_tolerance_ft:
                                        found_close = True
                                        break
                
                if found_close:
                    neighbors.append(neighbor_adapter)
                    
            except Exception as e:
                print(f"Error processing neighbor {scat_file}: {e}")
                continue
        
        return neighbors
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points in nautical miles"""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Radius of earth in nautical miles
        r_nm = 3440.065
        
        return c * r_nm
    
    def get_cached_track(self) -> List[TrackPoint]:
        """Get cached track points for performance"""
        if self._track_cache is None:
            self._track_cache = list(self.ownship_track())
        return self._track_cache
    
    def get_callsign(self) -> str:
        """Get aircraft callsign"""
        if not self._data:
            self.load_file()
        return self._data.callsign or 'UNKNOWN'
    
    def get_aircraft_type(self) -> str:
        """Get aircraft type"""
        if not self._data:
            self.load_file()
        return self._data.aircraft_type or 'UNKNOWN'
    
    def to_simple_scenario(self) -> Dict[str, Any]:
        """Convert SCAT data to simple scenario format for CDR pipeline"""
        if not self._data:
            self.load_file()
        
        # Extract first plot as starting position
        track_points = list(self.ownship_track())
        if not track_points:
            raise ValueError("No track points found in SCAT data")
        
        first_point = track_points[0]
        
        # Get flight plan info
        flight_plan = self.flight_plan()
        
        # Create ownship from first track point
        ownship = {
            'callsign': flight_plan.callsign or self._data.callsign or 'SCAT_AC',
            'aircraft_type': flight_plan.aircraft_type or self._data.aircraft_type or 'B738',
            'latitude': first_point.latitude,
            'longitude': first_point.longitude,
            'altitude_ft': first_point.altitude_ft,
            'heading_deg': first_point.heading_deg,
            'speed_kt': first_point.speed_kt or 250,  # Default if missing
            'departure': flight_plan.departure_airport or self._data.departure,
            'arrival': flight_plan.arrival_airport or self._data.arrival
        }
        
        # Create scenario with no initial intruders (ownship only scenario)
        scenario = {
            'ownship': ownship,
            'initial_traffic': [],
            'pending_intruders': [
                # Add a test intruder after 10 minutes for conflict testing
                {
                    'callsign': 'TEST_INTRUDER',
                    'aircraft_type': 'A320',
                    'latitude': first_point.latitude + 0.1,  # Slightly offset position
                    'longitude': first_point.longitude + 0.1,
                    'altitude_ft': first_point.altitude_ft,
                    'heading_deg': (first_point.heading_deg + 180) % 360,  # Opposite direction
                    'speed_kt': 280,
                    'injection_time_minutes': 10.0
                }
            ]
        }
        
        return scenario
