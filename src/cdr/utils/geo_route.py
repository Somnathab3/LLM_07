"""Geometric route planning utilities for conflict-aware navigation"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class RoutePoint:
    """A point along a flight route"""
    name: str
    lat: float
    lon: float
    altitude_ft: Optional[float] = None


@dataclass
class DogleggingResult:
    """Result of dogleg route planning"""
    via_waypoint: RoutePoint
    total_distance_nm: float
    lateral_offset_nm: float
    intruders_cleared: List[str]
    safety_margin_nm: float


class GeoRouteUtils:
    """Geometric utilities for aviation route planning"""
    
    @staticmethod
    def great_circle_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points in nautical miles"""
        # Convert degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in nautical miles
        earth_radius_nm = 3440.065
        distance = earth_radius_nm * c
        
        return distance
    
    @staticmethod
    def great_circle_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate initial bearing from point 1 to point 2 in degrees"""
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        
        y = math.sin(dlon) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon))
        
        bearing = math.degrees(math.atan2(y, x))
        
        # Normalize to 0-360 degrees
        return (bearing + 360) % 360
    
    @staticmethod
    def offset_point_on_great_circle(lat1: float, lon1: float, lat2: float, lon2: float,
                                   along_track_nm: float, lateral_offset_nm: float,
                                   side: int = 1) -> Tuple[float, float]:
        """
        Calculate a point offset from the great circle route
        
        Args:
            lat1, lon1: Starting point coordinates
            lat2, lon2: Ending point coordinates  
            along_track_nm: Distance along the track from start point
            lateral_offset_nm: Perpendicular distance from track
            side: +1 for right side, -1 for left side of track
            
        Returns:
            Tuple of (latitude, longitude) for the offset point
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Calculate the initial bearing
        bearing = GeoRouteUtils.great_circle_bearing(lat1, lon1, lat2, lon2)
        bearing_rad = math.radians(bearing)
        
        # Earth radius in nautical miles
        R = 3440.065
        
        # Calculate point along track
        angular_distance = along_track_nm / R
        
        lat_along = math.asin(
            math.sin(lat1_rad) * math.cos(angular_distance) +
            math.cos(lat1_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
        )
        
        lon_along = lon1_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat1_rad),
            math.cos(angular_distance) - math.sin(lat1_rad) * math.sin(lat_along)
        )
        
        # Calculate perpendicular bearing (90 degrees to the right/left)
        perp_bearing_rad = bearing_rad + side * math.pi / 2
        
        # Calculate offset point
        offset_angular_distance = lateral_offset_nm / R
        
        lat_offset = math.asin(
            math.sin(lat_along) * math.cos(offset_angular_distance) +
            math.cos(lat_along) * math.sin(offset_angular_distance) * math.cos(perp_bearing_rad)
        )
        
        lon_offset = lon_along + math.atan2(
            math.sin(perp_bearing_rad) * math.sin(offset_angular_distance) * math.cos(lat_along),
            math.cos(offset_angular_distance) - math.sin(lat_along) * math.sin(lat_offset)
        )
        
        # Convert back to degrees
        return math.degrees(lat_offset), math.degrees(lon_offset)
    
    @staticmethod
    def point_to_track_distance(point_lat: float, point_lon: float,
                              track_lat1: float, track_lon1: float,
                              track_lat2: float, track_lon2: float) -> float:
        """Calculate perpendicular distance from point to great circle track"""
        # This is a simplified calculation - for production use proper spherical geometry
        
        # Convert track to a line segment for approximation
        # Find the closest point on the track to the given point
        
        # Vector from track start to point
        dp_lat = point_lat - track_lat1
        dp_lon = point_lon - track_lon1
        
        # Track vector
        dt_lat = track_lat2 - track_lat1
        dt_lon = track_lon2 - track_lon1
        
        # Project point onto track (dot product)
        if dt_lat**2 + dt_lon**2 == 0:
            # Track has zero length
            return GeoRouteUtils.great_circle_distance_nm(point_lat, point_lon, track_lat1, track_lon1)
        
        t = max(0, min(1, (dp_lat * dt_lat + dp_lon * dt_lon) / (dt_lat**2 + dt_lon**2)))
        
        # Closest point on track
        closest_lat = track_lat1 + t * dt_lat
        closest_lon = track_lon1 + t * dt_lon
        
        # Distance from point to closest point on track
        return GeoRouteUtils.great_circle_distance_nm(point_lat, point_lon, closest_lat, closest_lon)


class DeterministicDogleggingPlanner:
    """Deterministic conflict-aware route planning"""
    
    def __init__(self, separation_min_nm: float = 5.0, safety_buffer_nm: float = 2.0):
        self.separation_min_nm = separation_min_nm
        self.safety_buffer_nm = safety_buffer_nm
        self.corridor_radius_nm = separation_min_nm + safety_buffer_nm
    
    def plan_dogleg_route(self, ownship: Dict[str, Any], destination: Dict[str, Any],
                         intruders: List[Dict[str, Any]], 
                         lookahead_minutes: float = 10.0) -> Optional[DogleggingResult]:
        """
        Plan a dogleg route around traffic conflicts
        
        Args:
            ownship: Current ownship state with lat, lon, heading, speed
            destination: Destination with name, lat, lon
            intruders: List of intruder aircraft states
            lookahead_minutes: Time horizon for conflict prediction
            
        Returns:
            DogleggingResult with via waypoint or None if no conflict resolution needed
        """
        ownship_lat = ownship.get('latitude', 0)
        ownship_lon = ownship.get('longitude', 0)
        ownship_speed_kt = ownship.get('speed', 400)
        
        dest_lat = destination.get('lat', 0)
        dest_lon = destination.get('lon', 0)
        dest_name = destination.get('name', 'DST')
        
        # Calculate baseline great-circle route
        direct_distance_nm = GeoRouteUtils.great_circle_distance_nm(
            ownship_lat, ownship_lon, dest_lat, dest_lon
        )
        
        direct_bearing = GeoRouteUtils.great_circle_bearing(
            ownship_lat, ownship_lon, dest_lat, dest_lon
        )
        
        # Check for conflicts with direct route
        blockers = []
        for intruder in intruders:
            if self._intruder_blocks_route(ownship, destination, intruder, lookahead_minutes):
                blockers.append(intruder)
        
        if not blockers:
            # No conflicts detected, direct route is clear
            return None
        
        # Choose side to avoid primary blocker
        primary_blocker = blockers[0]
        side = self._choose_avoidance_side(ownship, destination, primary_blocker)
        
        # Calculate via waypoint position
        along_track_distance = max(15, min(30, 0.5 * direct_distance_nm))
        lateral_offset = self.corridor_radius_nm + 3  # Extra margin for safety
        
        via_lat, via_lon = GeoRouteUtils.offset_point_on_great_circle(
            ownship_lat, ownship_lon, dest_lat, dest_lon,
            along_track_distance, lateral_offset, side
        )
        
        # Create via waypoint
        via_waypoint = RoutePoint(
            name="AVOID1",
            lat=via_lat,
            lon=via_lon,
            altitude_ft=ownship.get('altitude')
        )
        
        # Calculate total route distance
        leg1_distance = GeoRouteUtils.great_circle_distance_nm(
            ownship_lat, ownship_lon, via_lat, via_lon
        )
        leg2_distance = GeoRouteUtils.great_circle_distance_nm(
            via_lat, via_lon, dest_lat, dest_lon
        )
        total_distance = leg1_distance + leg2_distance
        
        # Calculate safety margin achieved
        safety_margin = self._calculate_safety_margin(via_waypoint, blockers)
        
        return DogleggingResult(
            via_waypoint=via_waypoint,
            total_distance_nm=total_distance,
            lateral_offset_nm=lateral_offset,
            intruders_cleared=[blocker.get('callsign', 'UNK') for blocker in blockers],
            safety_margin_nm=safety_margin
        )
    
    def _intruder_blocks_route(self, ownship: Dict[str, Any], destination: Dict[str, Any],
                              intruder: Dict[str, Any], lookahead_minutes: float) -> bool:
        """Check if intruder will penetrate the route corridor within lookahead time"""
        
        # Predict intruder position at lookahead time
        intruder_lat = intruder.get('latitude', 0)
        intruder_lon = intruder.get('longitude', 0)
        intruder_heading = intruder.get('heading', 0)
        intruder_speed_kt = intruder.get('speed', 400)
        
        # Simple straight-line projection
        distance_traveled_nm = intruder_speed_kt * (lookahead_minutes / 60.0)
        
        # Calculate future position
        heading_rad = math.radians(intruder_heading)
        dlat = (distance_traveled_nm / 60.0) * math.cos(heading_rad)  # Rough approximation
        dlon = (distance_traveled_nm / 60.0) * math.sin(heading_rad) / math.cos(math.radians(intruder_lat))
        
        future_lat = intruder_lat + dlat
        future_lon = intruder_lon + dlon
        
        # Check current and future positions against route corridor
        current_distance = GeoRouteUtils.point_to_track_distance(
            intruder_lat, intruder_lon,
            ownship.get('latitude', 0), ownship.get('longitude', 0),
            destination.get('lat', 0), destination.get('lon', 0)
        )
        
        future_distance = GeoRouteUtils.point_to_track_distance(
            future_lat, future_lon,
            ownship.get('latitude', 0), ownship.get('longitude', 0),
            destination.get('lat', 0), destination.get('lon', 0)
        )
        
        # Consider it a blocker if either current or future position is too close
        return min(current_distance, future_distance) < self.corridor_radius_nm
    
    def _choose_avoidance_side(self, ownship: Dict[str, Any], destination: Dict[str, Any],
                              intruder: Dict[str, Any]) -> int:
        """Choose which side (left=-1, right=+1) to avoid the intruder"""
        
        # Calculate cross product to determine which side intruder is on
        ownship_lat = ownship.get('latitude', 0)
        ownship_lon = ownship.get('longitude', 0)
        dest_lat = destination.get('lat', 0)
        dest_lon = destination.get('lon', 0)
        intruder_lat = intruder.get('latitude', 0)
        intruder_lon = intruder.get('longitude', 0)
        
        # Route vector
        route_lat = dest_lat - ownship_lat
        route_lon = dest_lon - ownship_lon
        
        # Vector to intruder
        intruder_lat_rel = intruder_lat - ownship_lat
        intruder_lon_rel = intruder_lon - ownship_lon
        
        # Cross product (simplified for lat/lon)
        cross_product = route_lat * intruder_lon_rel - route_lon * intruder_lat_rel
        
        # Choose opposite side from intruder
        return -1 if cross_product > 0 else 1
    
    def _calculate_safety_margin(self, via_waypoint: RoutePoint, 
                                intruders: List[Dict[str, Any]]) -> float:
        """Calculate minimum safety margin achieved by the dogleg"""
        
        min_margin = float('inf')
        
        for intruder in intruders:
            distance = GeoRouteUtils.great_circle_distance_nm(
                via_waypoint.lat, via_waypoint.lon,
                intruder.get('latitude', 0), intruder.get('longitude', 0)
            )
            min_margin = min(min_margin, distance)
        
        return min_margin if min_margin != float('inf') else 0.0


def create_deterministic_reroute(ownship: Dict[str, Any], destination: Dict[str, Any],
                               intruders: List[Dict[str, Any]], 
                               lookahead_minutes: float = 10.0) -> Optional[Dict[str, Any]]:
    """
    Convenience function to create a deterministic reroute resolution
    
    Returns:
        Dictionary in LLM resolution format or None if no rerouting needed
    """
    planner = DeterministicDogleggingPlanner()
    result = planner.plan_dogleg_route(ownship, destination, intruders, lookahead_minutes)
    
    if not result:
        return None
    
    return {
        "resolution_type": "reroute_via",
        "parameters": {
            "via_waypoint": {
                "name": result.via_waypoint.name,
                "lat": result.via_waypoint.lat,
                "lon": result.via_waypoint.lon
            },
            "resume_to_destination": True
        },
        "reasoning": f"Dogleg {result.lateral_offset_nm:.1f} NM around traffic; safety margin {result.safety_margin_nm:.1f} NM",
        "confidence": 0.85
    }
