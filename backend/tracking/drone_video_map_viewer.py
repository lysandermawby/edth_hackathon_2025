#!/usr/bin/env python3
"""
Drone Video and Map Data Processor

Processes drone video and GPS metadata to extract flight path information.
This is a headless version that doesn't require GUI components.

Usage:
    python drone_video_map_viewer.py <video_path> <metadata_csv_path>
"""

import os
import sys
import csv
import cv2
import numpy as np
import math
from datetime import datetime


class DroneVideoMapProcessor:
    def __init__(self, video_path, metadata_path):
        self.video_path = video_path
        self.metadata_path = metadata_path

        # Video properties
        self.cap = None
        self.frame_rate = 25.0
        self.frame_count = 0

        # Metadata
        self.metadata = []
        self.current_metadata_index = 0

    def load_metadata(self):
        """Load GPS metadata from CSV file"""
        print(f"Loading metadata from {self.metadata_path}")
        
        try:
            with open(self.metadata_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Convert string values to appropriate types
                    entry = {}
                    for key, value in row.items():
                        if key in ['timestamp', 'frame_number']:
                            entry[key] = int(float(value)) if value else 0
                        elif key in ['latitude', 'longitude', 'altitude', 'roll', 'pitch', 'yaw', 
                                   'gimbal_elevation', 'gimbal_azimuth', 'vfov', 'hfov']:
                            entry[key] = float(value) if value else 0.0
                        else:
                            entry[key] = value
                    self.metadata.append(entry)
            
            print(f"Loaded {len(self.metadata)} metadata entries")
            
            if self.metadata:
                # Calculate GPS bounds
                lats = [m.get('latitude', 0) for m in self.metadata if m.get('latitude', 0) != 0]
                lons = [m.get('longitude', 0) for m in self.metadata if m.get('longitude', 0) != 0]
                
                if lats and lons:
                    print(f"Debug: lats range: {min(lats)} to {max(lats)}")
                    print(f"Debug: lons range: {min(lons)} to {max(lons)}")
                    
                    center_lat = (min(lats) + max(lats)) / 2
                    center_lon = (min(lons) + max(lons)) / 2
                    print(f"Debug: map center: {center_lat}, {center_lon}")
                    
                    # Calculate zoom level based on GPS bounds
                    lat_range = max(lats) - min(lats)
                    lon_range = max(lons) - min(lons)
                    max_range = max(lat_range, lon_range)
                    
                    if max_range > 0.1:
                        zoom = 10
                    elif max_range > 0.01:
                        zoom = 14
                    else:
                        zoom = 16
                    
                    print(f"Map center: {center_lat:.6f}, {center_lon:.6f}")
                    print(f"Map zoom: {zoom}")
                    
                    # Count valid GPS entries
                    valid_gps = sum(1 for m in self.metadata if m.get('latitude', 0) != 0 and m.get('longitude', 0) != 0)
                    print(f"Valid GPS entries: {valid_gps}/{len(self.metadata)}")
                    
        except Exception as e:
            print(f"Error loading metadata: {e}")
            self.metadata = []

    def load_video(self):
        """Load video file and get properties"""
        print(f"Loading video: {self.video_path}")
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return False
        
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video loaded: {self.frame_count} frames at {self.frame_rate:.1f} FPS")
        print(f"Resolution: {width}x{height}")
        print(f"Duration: {self.frame_count / self.frame_rate:.2f} seconds")
        
        return True

    def get_metadata_for_frame(self, frame_number):
        """Get metadata for a specific frame number"""
        if not self.metadata:
            return None
        
        # Simple linear mapping - could be improved with timestamp matching
        if len(self.metadata) <= 1:
            return self.metadata[0] if self.metadata else None
        
        # Map frame number to metadata index
        metadata_index = int((frame_number / self.frame_count) * len(self.metadata))
        metadata_index = max(0, min(metadata_index, len(self.metadata) - 1))
        
        return self.metadata[metadata_index]

    def process_video(self):
        """Process video frames and extract flight path data"""
        if not self.cap or not self.cap.isOpened():
            print("Error: Video not loaded")
            return
        
        print("\n=== Processing Video for Flight Path Data ===")
        print("Extracting GPS coordinates and camera orientations...")
        
        flight_path = []
        frame_times = []
        
        frame_number = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Get metadata for this frame
            metadata = self.get_metadata_for_frame(frame_number)
            if metadata:
                lat = metadata.get('latitude', 0)
                lon = metadata.get('longitude', 0)
                alt = metadata.get('altitude', 0)
                yaw = metadata.get('yaw', 0)
                pitch = metadata.get('pitch', 0)
                roll = metadata.get('roll', 0)
                
                if lat != 0 and lon != 0:  # Valid GPS data
                    flight_path.append({
                        'frame': frame_number,
                        'timestamp': frame_number / self.frame_rate,
                        'latitude': lat,
                        'longitude': lon,
                        'altitude': alt,
                        'yaw': yaw,
                        'pitch': pitch,
                        'roll': roll
                    })
                    
                    frame_times.append(frame_number / self.frame_rate)
            
            frame_number += 1
            
            # Progress indicator
            if frame_number % 1000 == 0:
                progress = (frame_number / self.frame_count) * 100
                print(f"Processed {frame_number}/{self.frame_count} frames ({progress:.1f}%)")
        
        print(f"\nFlight path extraction complete!")
        print(f"Total frames processed: {frame_number}")
        print(f"Frames with GPS data: {len(flight_path)}")
        
        if flight_path:
            # Calculate flight statistics
            lats = [p['latitude'] for p in flight_path]
            lons = [p['longitude'] for p in flight_path]
            alts = [p['altitude'] for p in flight_path]
            
            print(f"\nFlight Statistics:")
            print(f"  Start position: {lats[0]:.6f}, {lons[0]:.6f} (alt: {alts[0]:.1f}m)")
            print(f"  End position: {lats[-1]:.6f}, {lons[-1]:.6f} (alt: {alts[-1]:.1f}m)")
            print(f"  Altitude range: {min(alts):.1f}m - {max(alts):.1f}m")
            print(f"  Flight duration: {frame_times[-1]:.1f} seconds")
            
            # Calculate total distance (rough approximation)
            total_distance = 0
            for i in range(1, len(flight_path)):
                lat1, lon1 = lats[i-1], lons[i-1]
                lat2, lon2 = lats[i], lons[i]
                
                # Haversine formula for distance calculation
                R = 6371000  # Earth's radius in meters
                dlat = math.radians(lat2 - lat1)
                dlon = math.radians(lon2 - lon1)
                a = (math.sin(dlat/2) * math.sin(dlat/2) + 
                     math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                     math.sin(dlon/2) * math.sin(dlon/2))
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                distance = R * c
                total_distance += distance
            
            print(f"  Total distance: {total_distance:.0f}m ({total_distance/1000:.2f}km)")
            print(f"  Average speed: {total_distance/frame_times[-1]:.1f}m/s ({total_distance/frame_times[-1]*3.6:.1f}km/h)")

    def run(self):
        """Main processing function"""
        print("Starting Drone Video & Map Data Processor...")
        print(f"Video: {self.video_path}")
        print(f"Metadata: {self.metadata_path}")
        
        # Load metadata
        self.load_metadata()
        
        # Load video
        if not self.load_video():
            return
        
        # Process video
        self.process_video()
        
        print("\nProcessing complete!")


def main():
    if len(sys.argv) != 3:
        print("Usage: python drone_video_map_viewer.py <video_path> <metadata_csv_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    metadata_path = sys.argv[2]

    # Check if files exist
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found: {metadata_path}")
        sys.exit(1)

    try:
        processor = DroneVideoMapProcessor(video_path, metadata_path)
        processor.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()