#!/usr/bin/env python3
"""
Video and Data File Cropping Script

This script reads time configuration from time_config.json and crops videos and their 
corresponding data files (CSV/JSON) based on specified time ranges. Each time range 
creates a separate output directory with organized files.

Usage:
    python video_crop.py <video_file_path>
    
The script will:
1. Look up the video file in time_config.json
2. Find the corresponding data file (CSV or JSON)
3. Crop both video and data file for each time range specified
4. Output organized files in data/cropped_videos/{video_name}/segment_{i}_{start}s-{end}s/

Time ranges are specified in SECONDS (not frames) in the configuration file.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import pandas as pd
import numpy as np


class VideoCSVCropper:
    def __init__(self, config_path: str = "time_config.json"):
        """Initialize the cropper with configuration file."""
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load the time configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print(f"Error: Configuration file {self.config_path} not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {self.config_path}: {e}")
            sys.exit(1)
    
    def _get_video_info(self, video_path: str) -> Tuple[float, int, int, int]:
        """Get video information including FPS, frame count, width, height."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        return fps, frame_count, width, height
    
    def _time_to_frame(self, time_seconds: float, fps: float) -> int:
        """Convert time in seconds to frame number."""
        return int(time_seconds * fps)
    
    def _crop_video(self, input_path: str, output_path: str, start_time: float, end_time: float) -> bool:
        """Crop video to specified time range."""
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"Error: Cannot open video {input_path}")
                return False
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate frame numbers
            start_frame = self._time_to_frame(start_time, fps)
            end_frame = self._time_to_frame(end_time, fps)
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Read and write frames
            frame_num = start_frame
            while frame_num <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frame_num += 1
            
            cap.release()
            out.release()
            return True
            
        except Exception as e:
            print(f"Error cropping video: {e}")
            return False
    
    def _crop_data_file(self, input_path: str, output_path: str, start_time: float, end_time: float, fps: float) -> bool:
        """Crop data file (CSV or JSON) to specified time range based on frame numbers."""
        try:
            file_ext = Path(input_path).suffix.lower()
            
            if file_ext == '.json':
                return self._crop_json(input_path, output_path, start_time, end_time, fps)
            elif file_ext == '.csv':
                return self._crop_csv(input_path, output_path, start_time, end_time, fps)
            else:
                print(f"Error: Unsupported file format {file_ext}")
                return False
                
        except Exception as e:
            print(f"Error cropping data file: {e}")
            return False
    
    def _crop_csv(self, input_path: str, output_path: str, start_time: float, end_time: float, fps: float) -> bool:
        """Crop CSV file to specified time range based on frame numbers."""
        try:
            # Read CSV file
            df = pd.read_csv(input_path)
            
            # Calculate frame numbers
            start_frame = self._time_to_frame(start_time, fps)
            end_frame = self._time_to_frame(end_time, fps)
            
            # Assume first column is frame number or index represents frames
            if 'frame' in df.columns:
                # If there's a 'frame' column, use it
                cropped_df = df[(df['frame'] >= start_frame) & (df['frame'] <= end_frame)].copy()
            else:
                # Otherwise, assume rows correspond to frames sequentially
                cropped_df = df.iloc[start_frame:end_frame + 1].copy()
                # Reset index to start from 0
                cropped_df.reset_index(drop=True, inplace=True)
            
            # Save cropped CSV
            cropped_df.to_csv(output_path, index=False)
            return True
            
        except Exception as e:
            print(f"Error cropping CSV: {e}")
            return False
    
    def _crop_json(self, input_path: str, output_path: str, start_time: float, end_time: float, fps: float) -> bool:
        """Crop JSON tracking data file to specified time range based on frame numbers."""
        try:
            import json
            
            # Read JSON file
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            # Calculate frame numbers
            start_frame = self._time_to_frame(start_time, fps)
            end_frame = self._time_to_frame(end_time, fps)
            
            # Filter data based on frame numbers
            if isinstance(data, list):
                # Handle list of frame data (like tracking_data.json)
                cropped_data = []
                for frame_data in data:
                    frame_num = frame_data.get('frame_number', 0)
                    if start_frame <= frame_num <= end_frame:
                        # Adjust frame numbers to start from 0
                        adjusted_frame = frame_data.copy()
                        adjusted_frame['frame_number'] = frame_num - start_frame
                        cropped_data.append(adjusted_frame)
            else:
                print("Error: JSON format not supported")
                return False
            
            # Save cropped JSON
            with open(output_path, 'w') as f:
                json.dump(cropped_data, f, indent=2)
            return True
            
        except Exception as e:
            print(f"Error cropping JSON: {e}")
            return False
    
    def crop_video_csv_pair(self, video_path: str) -> bool:
        """Crop video and corresponding CSV based on configuration."""
        # Normalize path for lookup
        video_path_abs = os.path.abspath(video_path)
        
        # Find video in config by comparing file paths
        video_config = None
        for config_entry in self.config.get('videos', []):
            config_video_path = config_entry['video_path']
            try:
                # Use os.path.samefile to handle different path representations
                if os.path.samefile(video_path, config_video_path):
                    video_config = config_entry
                    break
            except (OSError, FileNotFoundError):
                # If samefile fails, try absolute path comparison
                if os.path.abspath(config_video_path) == video_path_abs:
                    video_config = config_entry
                    break
        
        if not video_config:
            print(f"Error: Video {video_path} not found in configuration.")
            return False
        
        data_path = video_config.get('csv_path') or video_config.get('data_path')
        if not data_path or not os.path.exists(data_path):
            print(f"Error: Data file {data_path} not found or not specified.")
            return False
        
        time_ranges = video_config.get('time_ranges', [])
        if not time_ranges:
            print(f"Error: No time ranges specified for video {video_path}")
            return False
        
        # Get video info
        try:
            fps, frame_count, width, height = self._get_video_info(video_path)
            print(f"Video info: {fps} FPS, {frame_count} frames, {width}x{height}")
        except Exception as e:
            print(f"Error getting video info: {e}")
            return False
        
        # Create organized output directory structure in data/
        video_name = Path(video_path).stem
        # Create base directory: data/cropped_videos/{video_name}/
        # Use absolute path to ensure we're in the project root's data directory
        project_root = Path(__file__).parent.parent.parent  # Go up from backend/video_crop/ to project root
        base_output_dir = project_root / "data" / "cropped_videos" / video_name
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        
        # Process each time range
        for i, time_range in enumerate(time_ranges):
            start_time = time_range.get('start_time')
            end_time = time_range.get('end_time')
            
            if start_time is None or end_time is None:
                print(f"Warning: Skipping time range {i+1} - missing start_time or end_time")
                continue
            
            if start_time >= end_time:
                print(f"Warning: Skipping time range {i+1} - start_time ({start_time}) >= end_time ({end_time})")
                continue
            
            # Create organized output structure:
            # data/cropped_videos/{video_name}/segment_{i+1}_{start}s-{end}s/
            segment_name = f"segment_{i+1}_{start_time:.1f}s-{end_time:.1f}s"
            segment_dir = base_output_dir / segment_name
            segment_dir.mkdir(exist_ok=True)
            
            # Create output filenames within the segment directory
            video_output = segment_dir / f"{video_name}_{segment_name}.mp4"
            data_ext = Path(data_path).suffix
            data_output = segment_dir / f"{video_name}_{segment_name}{data_ext}"
            
            print(f"Processing segment {i+1}: {start_time}s - {end_time}s")
            
            # Crop video
            if self._crop_video(video_path, str(video_output), start_time, end_time):
                print(f"  ✓ Video cropped: {video_output}")
            else:
                print(f"  ✗ Failed to crop video")
                continue
            
            # Crop data file
            if self._crop_data_file(data_path, str(data_output), start_time, end_time, fps):
                print(f"  ✓ Data file cropped: {data_output}")
                success_count += 1
            else:
                print(f"  ✗ Failed to crop data file")
                # Remove the video file if data cropping failed
                if os.path.exists(video_output):
                    os.remove(video_output)
        
        print(f"\nCompleted: {success_count}/{len(time_ranges)} segments processed successfully")
        print(f"Output directory: {base_output_dir}")
        return success_count > 0


def main():
    parser = argparse.ArgumentParser(description='Crop videos and CSV files based on time ranges')
    parser.add_argument('video_path', help='Path to the video file to crop')
    parser.add_argument('--config', default='time_config.json', help='Path to time configuration file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file {args.video_path} not found.")
        sys.exit(1)
    
    cropper = VideoCSVCropper(args.config)
    success = cropper.crop_video_csv_pair(args.video_path)
    
    if success:
        print("Cropping completed successfully!")
    else:
        print("Cropping failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
