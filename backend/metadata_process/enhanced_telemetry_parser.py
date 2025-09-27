#!/usr/bin/env python3
"""
Enhanced telemetry parser for drone CSV data with geospatial analytics.

This module implements the recommendations from ChatGPT for comprehensive
telemetry analysis including camera footprint calculation, ground projection,
and coverage analysis.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np


@dataclass
class TelemetryPoint:
    """Single telemetry data point with all sensor readings."""
    timestamp: int  # microseconds since epoch
    vfov: float  # vertical field of view (degrees)
    hfov: float  # horizontal field of view (degrees)
    roll: float  # degrees
    pitch: float  # degrees
    yaw: float  # degrees
    latitude: float  # drone position
    longitude: float  # drone position
    altitude: float  # meters
    gimbal_elevation: float  # degrees
    gimbal_azimuth: float  # degrees
    center_latitude: float  # what camera is looking at
    center_longitude: float  # what camera is looking at
    center_elevation: float  # meters
    slant_range: float  # meters from drone to target


@dataclass
class CameraFootprint:
    """Camera footprint on ground with corner coordinates."""
    top_left: Tuple[float, float]  # (lat, lon)
    top_right: Tuple[float, float]
    bottom_left: Tuple[float, float]
    bottom_right: Tuple[float, float]
    center: Tuple[float, float]
    coverage_area: float  # square meters
    
    def to_polygon_coords(self) -> List[List[float]]:
        """Convert to polygon coordinates for mapping libraries."""
        return [
            [self.top_left[1], self.top_left[0]],  # [lon, lat] for GeoJSON
            [self.top_right[1], self.top_right[0]],
            [self.bottom_right[1], self.bottom_right[0]],
            [self.bottom_left[1], self.bottom_left[0]],
            [self.top_left[1], self.top_left[0]]  # Close polygon
        ]


@dataclass
class FlightAnalytics:
    """Flight stability and performance analytics."""
    total_distance: float  # meters
    max_altitude: float  # meters
    min_altitude: float  # meters
    avg_speed: float  # m/s
    max_speed: float  # m/s
    roll_variance: float  # stability metric
    pitch_variance: float  # stability metric
    yaw_variance: float  # stability metric
    flight_duration: float  # seconds
    hover_segments: List[Tuple[int, int]]  # [(start_idx, end_idx), ...]
    coverage_area: float  # total area covered in square meters


class EnhancedTelemetryParser:
    """Parser for drone telemetry CSV files with geospatial analytics."""
    
    def __init__(self):
        self.telemetry_points: List[TelemetryPoint] = []
        self.footprints: List[CameraFootprint] = []
        
    def parse_csv(self, csv_path: Path) -> List[TelemetryPoint]:
        """Parse telemetry CSV file into structured data points."""
        telemetry_points = []
        
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    point = TelemetryPoint(
                        timestamp=int(row['timestamp']),
                        vfov=float(row['vfov']),
                        hfov=float(row['hfov']),
                        roll=float(row['roll']),
                        pitch=float(row['pitch']),
                        yaw=float(row['yaw']),
                        latitude=float(row['latitude']),
                        longitude=float(row['longitude']),
                        altitude=float(row['altitude']),
                        gimbal_elevation=float(row['gimbal_elevation']),
                        gimbal_azimuth=float(row['gimbal_azimuth']),
                        center_latitude=float(row['center_latitude']),
                        center_longitude=float(row['center_longitude']),
                        center_elevation=float(row['center_elevation']),
                        slant_range=float(row['slant_range'])
                    )
                    telemetry_points.append(point)
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping invalid telemetry row: {e}")
                    continue
        
        self.telemetry_points = telemetry_points
        return telemetry_points
    
    def calculate_camera_footprint(self, point: TelemetryPoint) -> CameraFootprint:
        """
        Calculate the ground footprint of the camera for a given telemetry point.
        Uses FOV, altitude, and orientation to project camera view to ground.
        """
        # Convert angles to radians
        hfov_rad = math.radians(point.hfov)
        vfov_rad = math.radians(point.vfov)
        
        # Ground distance from center to edges (rough approximation)
        # This uses the slant range for more accurate calculation
        ground_width = 2 * point.slant_range * math.tan(hfov_rad / 2)
        ground_height = 2 * point.slant_range * math.tan(vfov_rad / 2)
        
        # Calculate corner offsets from center point in meters
        half_width = ground_width / 2
        half_height = ground_height / 2
        
        # Convert gimbal azimuth to proper heading (0° = North, clockwise)
        heading_rad = math.radians(point.gimbal_azimuth)
        
        # Calculate corner positions relative to center
        corners_local = [
            (-half_width, half_height),   # top-left
            (half_width, half_height),    # top-right
            (-half_width, -half_height),  # bottom-left
            (half_width, -half_height)    # bottom-right
        ]
        
        # Rotate corners based on gimbal azimuth
        rotated_corners = []
        for x, y in corners_local:
            rotated_x = x * math.cos(heading_rad) - y * math.sin(heading_rad)
            rotated_y = x * math.sin(heading_rad) + y * math.cos(heading_rad)
            rotated_corners.append((rotated_x, rotated_y))
        
        # Convert meter offsets to lat/lon using approximation
        # 1 degree latitude ≈ 111,111 meters
        # 1 degree longitude ≈ 111,111 * cos(latitude) meters
        lat_factor = 1.0 / 111111.0
        lon_factor = 1.0 / (111111.0 * math.cos(math.radians(point.center_latitude)))
        
        # Calculate actual corner coordinates
        corner_coords = []
        for dx, dy in rotated_corners:
            lat = point.center_latitude + (dy * lat_factor)
            lon = point.center_longitude + (dx * lon_factor)
            corner_coords.append((lat, lon))
        
        footprint = CameraFootprint(
            top_left=corner_coords[0],
            top_right=corner_coords[1],
            bottom_left=corner_coords[2],
            bottom_right=corner_coords[3],
            center=(point.center_latitude, point.center_longitude),
            coverage_area=ground_width * ground_height
        )
        
        return footprint
    
    def calculate_all_footprints(self) -> List[CameraFootprint]:
        """Calculate camera footprints for all telemetry points."""
        footprints = []
        for point in self.telemetry_points:
            footprint = self.calculate_camera_footprint(point)
            footprints.append(footprint)
        
        self.footprints = footprints
        return footprints
    
    def analyze_flight_performance(self) -> FlightAnalytics:
        """Analyze flight stability, efficiency, and coverage."""
        if len(self.telemetry_points) < 2:
            return FlightAnalytics(0, 0, 0, 0, 0, 0, 0, 0, 0, [], 0)
        
        points = self.telemetry_points
        
        # Calculate distances and speeds
        total_distance = 0.0
        speeds = []
        
        for i in range(1, len(points)):
            prev_point = points[i-1]
            curr_point = points[i]
            
            # Calculate distance using Haversine formula
            distance = self._haversine_distance(
                prev_point.latitude, prev_point.longitude,
                curr_point.latitude, curr_point.longitude
            )
            total_distance += distance
            
            # Calculate speed (distance / time)
            time_diff = (curr_point.timestamp - prev_point.timestamp) / 1_000_000.0  # seconds
            if time_diff > 0:
                speed = distance / time_diff
                speeds.append(speed)
        
        # Altitude statistics
        altitudes = [p.altitude for p in points]
        max_altitude = max(altitudes)
        min_altitude = min(altitudes)
        
        # Speed statistics
        avg_speed = np.mean(speeds) if speeds else 0.0
        max_speed = max(speeds) if speeds else 0.0
        
        # Stability metrics (variance in orientation)
        rolls = [p.roll for p in points]
        pitches = [p.pitch for p in points]
        yaws = [p.yaw for p in points]
        
        roll_variance = np.var(rolls)
        pitch_variance = np.var(pitches)
        yaw_variance = np.var(yaws)
        
        # Flight duration
        flight_duration = (points[-1].timestamp - points[0].timestamp) / 1_000_000.0
        
        # Detect hover segments (low speed periods)
        hover_segments = self._detect_hover_segments(speeds)
        
        # Total coverage area
        if self.footprints:
            coverage_area = sum(fp.coverage_area for fp in self.footprints)
        else:
            coverage_area = 0.0
        
        return FlightAnalytics(
            total_distance=total_distance,
            max_altitude=max_altitude,
            min_altitude=min_altitude,
            avg_speed=avg_speed,
            max_speed=max_speed,
            roll_variance=roll_variance,
            pitch_variance=pitch_variance,
            yaw_variance=yaw_variance,
            flight_duration=flight_duration,
            hover_segments=hover_segments,
            coverage_area=coverage_area
        )
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two lat/lon points using Haversine formula."""
        R = 6371000  # Earth radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _detect_hover_segments(self, speeds: List[float], hover_threshold: float = 1.0) -> List[Tuple[int, int]]:
        """Detect segments where drone is hovering (speed below threshold)."""
        hover_segments = []
        in_hover = False
        hover_start = 0
        
        for i, speed in enumerate(speeds):
            if speed < hover_threshold and not in_hover:
                in_hover = True
                hover_start = i
            elif speed >= hover_threshold and in_hover:
                in_hover = False
                if i - hover_start > 5:  # Only count hover segments > 5 points
                    hover_segments.append((hover_start, i))
        
        # Handle case where flight ends in hover
        if in_hover and len(speeds) - hover_start > 5:
            hover_segments.append((hover_start, len(speeds)))
        
        return hover_segments
    
    def normalize_timestamps_to_video(self, video_duration_seconds: float) -> List[float]:
        """
        Convert telemetry timestamps to video timeline (0 to video_duration).
        This aligns telemetry data with video playback time.
        """
        if not self.telemetry_points:
            return []
        
        # Get timestamp range
        start_timestamp = self.telemetry_points[0].timestamp
        end_timestamp = self.telemetry_points[-1].timestamp
        timestamp_duration = (end_timestamp - start_timestamp) / 1_000_000.0  # seconds
        
        # Convert each timestamp to video time
        video_timestamps = []
        for point in self.telemetry_points:
            # Calculate relative timestamp (0 to 1)
            relative_time = (point.timestamp - start_timestamp) / (end_timestamp - start_timestamp)
            # Map to video duration
            video_time = relative_time * video_duration_seconds
            video_timestamps.append(video_time)
        
        return video_timestamps
    
    def export_for_frontend(self, video_duration_seconds: float) -> Dict[str, Any]:
        """Export telemetry data in format suitable for frontend consumption."""
        if not self.telemetry_points:
            return {"telemetry": [], "footprints": [], "analytics": None}
        
        # Calculate footprints if not already done
        if not self.footprints:
            self.calculate_all_footprints()
        
        # Normalize timestamps to video timeline
        video_timestamps = self.normalize_timestamps_to_video(video_duration_seconds)
        
        # Create simplified telemetry data for frontend
        telemetry_data = []
        for i, (point, video_time) in enumerate(zip(self.telemetry_points, video_timestamps)):
            telemetry_data.append({
                "timestamp": video_time,
                "latitude": point.latitude,
                "longitude": point.longitude,
                "altitude": point.altitude,
                "roll": point.roll,
                "pitch": point.pitch,
                "yaw": point.yaw,
                "gimbal_elevation": point.gimbal_elevation,
                "gimbal_azimuth": point.gimbal_azimuth,
                "center_latitude": point.center_latitude,
                "center_longitude": point.center_longitude,
                "slant_range": point.slant_range,
                "hfov": point.hfov,
                "vfov": point.vfov,
                "footprint": self.footprints[i].to_polygon_coords() if i < len(self.footprints) else None
            })
        
        # Get flight analytics
        analytics = self.analyze_flight_performance()
        
        return {
            "telemetry": telemetry_data,
            "analytics": {
                "total_distance": analytics.total_distance,
                "max_altitude": analytics.max_altitude,
                "min_altitude": analytics.min_altitude,
                "avg_speed": analytics.avg_speed,
                "max_speed": analytics.max_speed,
                "flight_duration": analytics.flight_duration,
                "coverage_area": analytics.coverage_area,
                "stability_metrics": {
                    "roll_variance": analytics.roll_variance,
                    "pitch_variance": analytics.pitch_variance,
                    "yaw_variance": analytics.yaw_variance
                }
            }
        }


def parse_telemetry_csv(csv_path: Path, video_duration_seconds: float = None) -> Dict[str, Any]:
    """
    Convenience function to parse telemetry CSV and return frontend-ready data.
    
    Args:
        csv_path: Path to the telemetry CSV file
        video_duration_seconds: Duration of corresponding video for timestamp alignment
        
    Returns:
        Dictionary with telemetry data, footprints, and analytics
    """
    parser = EnhancedTelemetryParser()
    parser.parse_csv(csv_path)
    
    if video_duration_seconds is None:
        # Estimate duration from telemetry timestamps
        if parser.telemetry_points:
            duration = (parser.telemetry_points[-1].timestamp - 
                       parser.telemetry_points[0].timestamp) / 1_000_000.0
            video_duration_seconds = duration
        else:
            video_duration_seconds = 30.0  # fallback
    
    return parser.export_for_frontend(video_duration_seconds)


if __name__ == "__main__":
    # Test the parser with sample data
    import sys
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
        if csv_path.exists():
            result = parse_telemetry_csv(csv_path)
            print(f"Parsed {len(result['telemetry'])} telemetry points")
            if result['analytics']:
                analytics = result['analytics']
                print(f"Flight duration: {analytics['flight_duration']:.1f}s")
                print(f"Total distance: {analytics['total_distance']:.1f}m")
                print(f"Coverage area: {analytics['coverage_area']:.1f}m²")
        else:
            print(f"CSV file not found: {csv_path}")
    else:
        print("Usage: python enhanced_telemetry_parser.py <csv_path>")
