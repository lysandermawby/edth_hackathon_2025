#!/usr/bin/env python3
"""
3D Position Calculator for Object Tracking

This module converts 2D pixel coordinates from object detection to 3D world coordinates
using camera parameters, drone telemetry, and object size estimation.

The 3D positioning is essential for kinematic re-identification as it allows
the system to predict where objects will be in real-world space, not just
in the 2D image plane.
"""

import math
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class CameraParameters:
    """Camera intrinsic and extrinsic parameters"""
    width: int  # Image width in pixels
    height: int  # Image height in pixels
    hfov: float  # Horizontal field of view in degrees
    vfov: float  # Vertical field of view in degrees
    focal_length: Optional[float] = None  # Focal length in pixels (calculated if None)
    
    def __post_init__(self):
        if self.focal_length is None:
            # Calculate focal length from FOV
            self.focal_length = self.width / (2 * math.tan(math.radians(self.hfov) / 2))


@dataclass
class DronePose:
    """Drone position and orientation"""
    latitude: float
    longitude: float
    altitude: float  # meters above ground
    roll: float  # degrees
    pitch: float  # degrees
    yaw: float  # degrees
    gimbal_elevation: float  # degrees
    gimbal_azimuth: float  # degrees


@dataclass
class Object3DPosition:
    """3D position of an object in world coordinates"""
    latitude: float
    longitude: float
    altitude: float  # meters above ground
    distance: float  # distance from drone in meters
    bearing: float  # bearing from drone in degrees
    elevation_angle: float  # elevation angle from drone in degrees
    confidence: float  # confidence in position estimate (0-1)


class Position3DCalculator:
    """Calculates 3D world positions from 2D pixel coordinates"""
    
    def __init__(self, camera_params: CameraParameters):
        self.camera_params = camera_params
        
        # Object size estimates for different classes (in meters)
        self.object_sizes = {
            'person': {'width': 0.6, 'height': 1.7, 'length': 0.4},
            'car': {'width': 1.8, 'height': 1.5, 'length': 4.5},
            'truck': {'width': 2.5, 'height': 3.0, 'length': 8.0},
            'bus': {'width': 2.5, 'height': 3.5, 'length': 12.0},
            'bicycle': {'width': 0.6, 'height': 1.2, 'length': 1.8},
            'motorcycle': {'width': 0.8, 'height': 1.4, 'length': 2.2},
            'boat': {'width': 2.0, 'height': 1.5, 'length': 6.0},
            'airplane': {'width': 3.0, 'height': 2.5, 'length': 15.0},
            'helicopter': {'width': 2.0, 'height': 2.0, 'length': 8.0},
        }
        
        # Default size for unknown objects
        self.default_size = {'width': 1.0, 'height': 1.0, 'length': 1.0}
    
    def pixel_to_3d(self, 
                   bbox: Tuple[float, float, float, float],  # (x1, y1, x2, y2)
                   class_name: str,
                   drone_pose: DronePose) -> Object3DPosition:
        """
        Convert 2D pixel coordinates to 3D world coordinates
        
        Args:
            bbox: Bounding box in pixels (x1, y1, x2, y2)
            class_name: Object class name for size estimation
            drone_pose: Current drone position and orientation
            
        Returns:
            Object3DPosition with 3D world coordinates
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate object center in pixels
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate object size in pixels
        width_pixels = x2 - x1
        height_pixels = y2 - y1
        
        # Get estimated object size in meters
        object_size = self.object_sizes.get(class_name.lower(), self.default_size)
        
        # Calculate distance using object size
        distance = self._calculate_distance_from_size(
            width_pixels, height_pixels, object_size, drone_pose.altitude
        )
        
        # Calculate 3D position
        lat, lon, alt = self._pixel_to_world_coordinates(
            center_x, center_y, distance, drone_pose
        )
        
        # Calculate bearing and elevation
        bearing = self._calculate_bearing(drone_pose, lat, lon)
        elevation_angle = self._calculate_elevation_angle(
            drone_pose, lat, lon, alt
        )
        
        # Calculate confidence based on object size and distance
        confidence = self._calculate_confidence(
            distance, object_size, width_pixels, height_pixels
        )
        
        return Object3DPosition(
            latitude=lat,
            longitude=lon,
            altitude=alt,
            distance=distance,
            bearing=bearing,
            elevation_angle=elevation_angle,
            confidence=confidence
        )
    
    def _calculate_distance_from_size(self, 
                                    width_pixels: float, 
                                    height_pixels: float, 
                                    object_size: Dict[str, float],
                                    altitude: float) -> float:
        """Calculate distance to object using its apparent size"""
        
        # Use the larger dimension for distance calculation
        # This is more robust than using just width or height
        object_width_m = object_size['width']
        object_height_m = object_size['height']
        
        # Calculate distance using both width and height, take the average
        distance_from_width = (object_width_m * self.camera_params.focal_length) / width_pixels
        distance_from_height = (object_height_m * self.camera_params.focal_length) / height_pixels
        
        # Average the two estimates, but weight by the more reliable dimension
        if width_pixels > height_pixels:
            # Width is more reliable
            distance = 0.7 * distance_from_width + 0.3 * distance_from_height
        else:
            # Height is more reliable
            distance = 0.3 * distance_from_width + 0.7 * distance_from_height
        
        # Ensure distance is reasonable (not too close or too far)
        distance = max(1.0, min(distance, altitude * 2))  # Between 1m and 2x altitude
        
        return distance
    
    def _pixel_to_world_coordinates(self, 
                                  center_x: float, 
                                  center_y: float, 
                                  distance: float,
                                  drone_pose: DronePose) -> Tuple[float, float, float]:
        """Convert pixel coordinates to world coordinates"""
        
        # Convert pixel coordinates to normalized camera coordinates
        # Camera center is at (width/2, height/2)
        norm_x = (center_x - self.camera_params.width / 2) / self.camera_params.focal_length
        norm_y = (center_y - self.camera_params.height / 2) / self.camera_params.focal_length
        
        # Convert to 3D camera coordinates
        # Note: Camera coordinates have Z pointing forward, Y pointing down
        camera_x = norm_x * distance
        camera_y = norm_y * distance
        camera_z = distance
        
        # Convert to world coordinates using drone orientation
        world_x, world_y, world_z = self._camera_to_world_coordinates(
            camera_x, camera_y, camera_z, drone_pose
        )
        
        # Convert to lat/lon/alt
        lat, lon, alt = self._world_to_geographic(
            world_x, world_y, world_z, drone_pose
        )
        
        return lat, lon, alt
    
    def _camera_to_world_coordinates(self, 
                                   camera_x: float, 
                                   camera_y: float, 
                                   camera_z: float,
                                   drone_pose: DronePose) -> Tuple[float, float, float]:
        """Convert camera coordinates to world coordinates"""
        
        # Convert angles to radians
        roll_rad = math.radians(drone_pose.roll)
        pitch_rad = math.radians(drone_pose.pitch)
        yaw_rad = math.radians(drone_pose.yaw)
        gimbal_elevation_rad = math.radians(drone_pose.gimbal_elevation)
        gimbal_azimuth_rad = math.radians(drone_pose.gimbal_azimuth)
        
        # Create rotation matrices
        # Roll rotation (around X axis)
        roll_matrix = np.array([
            [1, 0, 0],
            [0, math.cos(roll_rad), -math.sin(roll_rad)],
            [0, math.sin(roll_rad), math.cos(roll_rad)]
        ])
        
        # Pitch rotation (around Y axis)
        pitch_matrix = np.array([
            [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
            [0, 1, 0],
            [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
        ])
        
        # Yaw rotation (around Z axis)
        yaw_matrix = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
            [math.sin(yaw_rad), math.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        
        # Gimbal rotation
        gimbal_elevation_matrix = np.array([
            [1, 0, 0],
            [0, math.cos(gimbal_elevation_rad), -math.sin(gimbal_elevation_rad)],
            [0, math.sin(gimbal_elevation_rad), math.cos(gimbal_elevation_rad)]
        ])
        
        gimbal_azimuth_matrix = np.array([
            [math.cos(gimbal_azimuth_rad), -math.sin(gimbal_azimuth_rad), 0],
            [math.sin(gimbal_azimuth_rad), math.cos(gimbal_azimuth_rad), 0],
            [0, 0, 1]
        ])
        
        # Combine all rotations
        camera_point = np.array([camera_x, camera_y, camera_z])
        
        # Apply rotations in order: gimbal -> drone orientation
        rotated = gimbal_azimuth_matrix @ gimbal_elevation_matrix @ camera_point
        rotated = yaw_matrix @ pitch_matrix @ roll_matrix @ rotated
        
        return rotated[0], rotated[1], rotated[2]
    
    def _world_to_geographic(self, 
                           world_x: float, 
                           world_y: float, 
                           world_z: float,
                           drone_pose: DronePose) -> Tuple[float, float, float]:
        """Convert world coordinates to geographic coordinates"""
        
        # Convert meter offsets to lat/lon using approximation
        # 1 degree latitude ≈ 111,111 meters
        # 1 degree longitude ≈ 111,111 * cos(latitude) meters
        lat_factor = 1.0 / 111111.0
        lon_factor = 1.0 / (111111.0 * math.cos(math.radians(drone_pose.latitude)))
        
        # Calculate geographic coordinates
        lat = drone_pose.latitude + (world_y * lat_factor)
        lon = drone_pose.longitude + (world_x * lon_factor)
        alt = drone_pose.altitude + world_z
        
        return lat, lon, alt
    
    def _calculate_bearing(self, 
                         drone_pose: DronePose, 
                         target_lat: float, 
                         target_lon: float) -> float:
        """Calculate bearing from drone to target"""
        
        # Convert to radians
        lat1_rad = math.radians(drone_pose.latitude)
        lat2_rad = math.radians(target_lat)
        dlon_rad = math.radians(target_lon - drone_pose.longitude)
        
        # Calculate bearing
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        # Normalize to 0-360 degrees
        return (bearing_deg + 360) % 360
    
    def _calculate_elevation_angle(self, 
                                 drone_pose: DronePose, 
                                 target_lat: float, 
                                 target_lon: float,
                                 target_alt: float) -> float:
        """Calculate elevation angle from drone to target"""
        
        # Calculate horizontal distance
        horizontal_distance = self._haversine_distance(
            drone_pose.latitude, drone_pose.longitude, target_lat, target_lon
        )
        
        # Calculate elevation angle
        vertical_distance = target_alt - drone_pose.altitude
        elevation_rad = math.atan2(vertical_distance, horizontal_distance)
        elevation_deg = math.degrees(elevation_rad)
        
        return elevation_deg
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in meters
        earth_radius = 6371000
        distance = earth_radius * c
        
        return distance
    
    def _calculate_confidence(self, 
                            distance: float, 
                            object_size: Dict[str, float],
                            width_pixels: float, 
                            height_pixels: float) -> float:
        """Calculate confidence in position estimate"""
        
        # Base confidence on distance (closer objects are more accurate)
        distance_confidence = max(0.1, 1.0 - (distance / 1000.0))  # Decreases with distance
        
        # Confidence based on object size (larger objects are more accurate)
        size_confidence = min(1.0, (object_size['width'] + object_size['height']) / 2.0)
        
        # Confidence based on pixel size (larger in pixels is more accurate)
        pixel_confidence = min(1.0, (width_pixels + height_pixels) / 100.0)
        
        # Combine confidences
        total_confidence = (distance_confidence * 0.4 + 
                          size_confidence * 0.3 + 
                          pixel_confidence * 0.3)
        
        return min(1.0, max(0.1, total_confidence))


def create_camera_parameters_from_metadata(metadata: Dict[str, Any]) -> CameraParameters:
    """Create camera parameters from metadata"""
    
    # Default values
    width = metadata.get('width', 1280)
    height = metadata.get('height', 720)
    hfov = metadata.get('hfov', 90.0)
    vfov = metadata.get('vfov', 60.0)
    
    return CameraParameters(width=width, height=height, hfov=hfov, vfov=vfov)


def create_drone_pose_from_metadata(metadata: Dict[str, Any]) -> DronePose:
    """Create drone pose from metadata"""
    
    return DronePose(
        latitude=metadata.get('latitude', 0.0),
        longitude=metadata.get('longitude', 0.0),
        altitude=metadata.get('altitude', 100.0),
        roll=metadata.get('roll', 0.0),
        pitch=metadata.get('pitch', 0.0),
        yaw=metadata.get('yaw', 0.0),
        gimbal_elevation=metadata.get('gimbal_elevation', 0.0),
        gimbal_azimuth=metadata.get('gimbal_azimuth', 0.0)
    )


if __name__ == "__main__":
    # Test the 3D position calculator
    camera_params = CameraParameters(width=1280, height=720, hfov=90.0, vfov=60.0)
    calculator = Position3DCalculator(camera_params)
    
    # Test with a car detection
    bbox = (640, 360, 800, 500)  # Center of image, car-sized
    class_name = "car"
    drone_pose = DronePose(
        latitude=48.1351,
        longitude=11.5820,
        altitude=100.0,
        roll=0.0,
        pitch=-15.0,
        yaw=45.0,
        gimbal_elevation=-30.0,
        gimbal_azimuth=0.0
    )
    
    position_3d = calculator.pixel_to_3d(bbox, class_name, drone_pose)
    print(f"3D Position: {position_3d.latitude:.6f}, {position_3d.longitude:.6f}")
    print(f"Distance: {position_3d.distance:.2f}m")
    print(f"Bearing: {position_3d.bearing:.1f}°")
    print(f"Elevation: {position_3d.elevation_angle:.1f}°")
    print(f"Confidence: {position_3d.confidence:.2f}")
