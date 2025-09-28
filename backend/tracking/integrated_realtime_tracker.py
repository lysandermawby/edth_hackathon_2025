#!/usr/bin/env python3
"""
Integrated real-time video tracker with database storage.

Combines real-time video playback with live tracking and database storage
for real-time processing and decision making.
"""

import os
import argparse
import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
import time
import math
from datetime import datetime
import threading
import queue
import sys
import platform
from database_integration import TrackingDatabase, RealTimeDataProcessor

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Script is in project_root/backend/tracking/, so go up 2 levels to reach project_root
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

class IntegratedRealtimeTracker:
    def __init__(self, model_path="../../models/yolo11m.pt", show_labels=True, ignore_classes=None, 
                 enable_database=True, db_path="../../databases/tracking_data.db", headless=False, force_gui=False):
        """
        Initialize the integrated real-time tracker
        
        Args:
            model_path (str): Path to YOLO model weights (if None, uses default)
            show_labels (bool): Whether to show class labels
            ignore_classes (list): List of class names to ignore
            enable_database (bool): Whether to enable database storage
            db_path (str): Path to SQLite database file
            headless (bool): Whether to run in headless mode (no GUI)
            force_gui (bool): Force GUI mode, skip headless detection
        """
        # Set default paths relative to project root
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, "models", "yolo11m.pt")
        if db_path is None:
            db_path = os.path.join(PROJECT_ROOT, "databases", "tracking_data.db")
        
        self.tracker = sv.ByteTrack()
        self.model = YOLO(model_path)
        self.annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
        self.box_annotator = sv.BoxAnnotator()
        self.show_labels = show_labels
        self.ignore_classes = ignore_classes or []
        self.headless = headless
        
        # Auto-detect headless mode if running in WSL2 or without display
        # Skip detection if GUI is forced
        if not self.headless and not force_gui:
            try:
                detected_headless = self._detect_headless_mode()
                print(f"Headless detection: {detected_headless}")
                self.headless = detected_headless
            except Exception as e:
                print(f"Headless detection failed, defaulting to headless mode: {e}")
                self.headless = True
        elif force_gui:
            print("GUI mode forced - skipping headless detection")
            self.headless = False
            # Test if GUI actually works when forced
            try:
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imshow('test', test_img)
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                print("GUI test successful - GUI mode enabled")
            except Exception as e:
                print(f"WARNING: GUI mode forced but OpenCV doesn't support GUI: {e}")
                print("The script will attempt to run in GUI mode but may fail during video display.")
        
        # Database integration
        self.enable_database = enable_database
        if enable_database:
            # Ensure database directory exists
            db_dir = os.path.dirname(os.path.abspath(db_path))
            os.makedirs(db_dir, exist_ok=True)
            
            self.db = TrackingDatabase(db_path)
            self.data_processor = RealTimeDataProcessor(self.db)
        else:
            self.db = None
            self.data_processor = None
        
        # Tracking state
        self.frame_count = 0
        self.start_time = None
        self.session_id = None
        
        # Headless mode settings
        if self.headless:
            self.output_dir = "output_frames"
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Running in headless mode. Frames will be saved to: {self.output_dir}")
    
    def _detect_headless_mode(self):
        """Detect if running in headless environment (WSL2, no display, etc.)"""
        system = platform.system()
        
        # On macOS and Windows, always test GUI directly since they don't use DISPLAY/WAYLAND
        if system in ['Darwin', 'Windows']:
            print(f"Testing GUI on {system}...")
            try:
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imshow('test', test_img)
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                print(f"GUI test successful on {system}")
                return False  # GUI is available
            except Exception as e:
                print(f"GUI test failed on {system}: {e}")
                return True  # GUI not available
        
        # For Linux/Unix systems, check environment variables first
        # Check for WSL2 with WSLg (Wayland support)
        if os.environ.get('WAYLAND_DISPLAY'):
            # WSLg is available, but still test GUI
            pass
        elif not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
            # No display environment detected
            print(f"No display environment detected on {system}")
            return True
            
        # Try to create a test window to see if GUI is available
        # This is the most reliable way to detect if OpenCV has GUI support
        try:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow('test', test_img)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            print(f"GUI test successful on {system}")
            return False  # GUI is available
        except Exception as e:
            print(f"GUI test failed on {system}: {e}")
            return True  # GUI not available
    
    def _safe_destroy_windows(self):
        """Safely destroy OpenCV windows, handling cases where GUI is not available"""
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Warning: Could not destroy OpenCV windows: {e}")
        
    def filter_detections(self, detections):
        """Filter out ignored classes"""
        if not self.ignore_classes:
            return detections
            
        class_names = self.model.names
        keep_mask = []
        for class_id in detections.class_id:
            class_name = class_names[class_id]
            keep_mask.append(class_name.lower() not in [c.lower() for c in self.ignore_classes])
        
        if any(keep_mask):
            return detections[keep_mask]
        else:
            return detections[[]]  # Empty detections
    
    def create_labels(self, detections):
        """Create labels for detections"""
        labels = []
        for i in range(len(detections)):
            if self.show_labels and detections.class_id[i] is not None:
                class_name = self.model.names[detections.class_id[i]]
                if detections.tracker_id[i] is not None:
                    labels.append(f"{class_name} ID:{detections.tracker_id[i]}")
                else:
                    labels.append(f"{class_name} New")
            else:
                if detections.tracker_id[i] is not None:
                    labels.append(f"ID: {detections.tracker_id[i]}")
                else:
                    labels.append("New")
        return labels
    
    def extract_tracking_data(self, detections, frame_timestamp):
        """Extract tracking data for database storage with velocity and acceleration"""
        frame_data = {
            'frame_number': self.frame_count,
            'timestamp': frame_timestamp,
            'objects': []
        }
        
        for i in range(len(detections)):
            if detections.tracker_id[i] is not None:
                obj_data = {
                    'tracker_id': int(detections.tracker_id[i]),
                    'class_id': int(detections.class_id[i]) if detections.class_id[i] is not None else None,
                    'class_name': self.model.names[detections.class_id[i]] if detections.class_id[i] is not None else "Unknown",
                    'confidence': float(detections.confidence[i]) if detections.confidence[i] is not None else None,
                    'bbox': {
                        'x1': float(detections.xyxy[i][0]),
                        'y1': float(detections.xyxy[i][1]),
                        'x2': float(detections.xyxy[i][2]),
                        'y2': float(detections.xyxy[i][3])
                    },
                    'center': {
                        'x': float((detections.xyxy[i][0] + detections.xyxy[i][2]) / 2),
                        'y': float((detections.xyxy[i][1] + detections.xyxy[i][3]) / 2)
                    }
                }
                
                # Calculate velocity and acceleration if we have track history
                velocity = self._compute_velocity(int(detections.tracker_id[i]), obj_data['center'], frame_timestamp)
                obj_data['velocity'] = velocity
                
                frame_data['objects'].append(obj_data)
        
        return frame_data
    
    def _compute_velocity(self, track_id, center, timestamp):
        """Compute velocity and acceleration for a track"""
        # Initialize track history if not exists
        if not hasattr(self, 'track_history'):
            self.track_history = {}
        
        prev_state = self.track_history.get(track_id)
        velocity = {
            'vx': 0.0,
            'vy': 0.0,
            'speed': 0.0,
            'direction': None,
            'delta_time': None
        }
        
        if prev_state is not None:
            dt = timestamp - prev_state['timestamp']
            velocity['delta_time'] = dt if dt >= 0 else None
            
            if dt and dt > 1e-3:
                prev_cx, prev_cy = prev_state['center']
                dx = center['x'] - prev_cx
                dy = center['y'] - prev_cy
                vx = dx / dt
                vy = dy / dt
                
                # Calculate acceleration
                acceleration = {'ax': 0.0, 'ay': 0.0, 'magnitude': 0.0}
                if 'velocity' in prev_state:
                    prev_vx, prev_vy = prev_state['velocity']['vx'], prev_state['velocity']['vy']
                    ax = (vx - prev_vx) / dt
                    ay = (vy - prev_vy) / dt
                    acceleration = {
                        'ax': ax,
                        'ay': ay,
                        'magnitude': math.sqrt(ax**2 + ay**2)
                    }
                
                velocity['acceleration'] = acceleration
            else:
                velocity['acceleration'] = prev_state.get('acceleration', {'ax': 0.0, 'ay': 0.0, 'magnitude': 0.0})
        else:
            velocity['acceleration'] = {'ax': 0.0, 'ay': 0.0, 'magnitude': 0.0}
        
        speed = math.sqrt(velocity['vx']**2 + velocity['vy']**2)
        heading = None
        if speed > 1e-3:
            heading = (math.degrees(math.atan2(velocity['vy'], velocity['vx'])) + 360.0) % 360.0
        
        velocity.update({
            'vx': velocity['vx'],
            'vy': velocity['vy'],
            'speed': speed,
            'direction': heading,
        })
        
        # Update history
        self.track_history[track_id] = {
            'center': (center['x'], center['y']),
            'timestamp': timestamp,
            'velocity': velocity
        }
        
        return velocity
    
    def process_frame(self, frame, frame_timestamp):
        """Process a single frame for detection and tracking"""
        # Run YOLO detection
        result = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter ignored classes
        detections = self.filter_detections(detections)
        
        # Update tracker
        detections = self.tracker.update_with_detections(detections)
        
        # Extract tracking data for database
        tracking_data = self.extract_tracking_data(detections, frame_timestamp)
        
        # Send to real-time data processor
        if self.data_processor:
            self.data_processor.add_frame_data(tracking_data)
        
        # Create labels
        labels = self.create_labels(detections)
        
        # Annotate frame
        annotated_frame = self.box_annotator.annotate(frame, detections)
        if labels:
            annotated_frame = self.annotator.annotate(annotated_frame, detections, labels=labels)
        
        return annotated_frame, detections, tracking_data
    
    def add_info_overlay(self, frame, fps, total_objects, db_status=""):
        """Add information overlay to frame"""
        # FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Object count
        cv2.putText(frame, f"Objects: {total_objects}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Database status
        if self.enable_database:
            cv2.putText(frame, f"DB: {db_status}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Controls info
        cv2.putText(frame, "Controls: 'q'=quit, 's'=save, 'r'=reset, SPACE=pause, 'i'=db info", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    
    def run(self, video_path, session_id=None, reset_frame_count=True):
        """Main loop for integrated real-time video tracking.

        Args:
            video_path (str): Path to the video file that should be processed.
            session_id (int | None): Existing database session identifier to
                reuse. When omitted, a new session is created automatically.
            reset_frame_count (bool): Whether to reset the internal frame
                counter before processing. Leave as True when regenerating
                detections for an existing session so frame numbering starts at
                zero.
        """
        if reset_frame_count:
            self.frame_count = 0

        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {total_frames} frames at {fps:.2f} FPS")
        print(f"Resolution: {width}x{height}")
        print(f"Duration: {total_frames/fps:.2f} seconds")
        print(f"Show labels: {self.show_labels}")
        if self.ignore_classes:
            print(f"Ignoring classes: {', '.join(self.ignore_classes)}")
        print(f"Database enabled: {self.enable_database}")
        
        # Start or reuse database session
        if self.enable_database:
            if session_id is not None:
                self.session_id = session_id
                self.data_processor.start_processing(video_path, fps, self.session_id)
                print(f"Using existing database session: {self.session_id}")
            else:
                self.session_id = self.db.start_session(video_path, fps)
                self.data_processor.start_processing(video_path, fps, self.session_id)
                print(f"Database session started: {self.session_id}")
        
        if self.headless:
            print("\n=== Integrated Real-time Video Tracking (Headless Mode) ===")
            print("Processing video without GUI display")
            print("Frames will be saved to output directory")
            print("Press Ctrl+C to stop processing")
            print("========================================================\n")
        else:
            print("\n=== Integrated Real-time Video Tracking ===")
            print("Press 'q' to quit")
            print("Press 's' to save current frame")
            print("Press 'r' to reset tracker")
            print("Press SPACE to pause/resume")
            print("Press 'i' to show database info")
            print("==========================================\n")
        
        self.start_time = time.time()
        paused = False
        frame_times = []
        last_db_check = time.time()
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("End of video reached")
                        break
                    
                    # Calculate frame timestamp
                    frame_timestamp = self.frame_count / fps
                    
                    # Process frame
                    start_process = time.time()
                    annotated_frame, detections, tracking_data = self.process_frame(frame, frame_timestamp)
                    process_time = time.time() - start_process
                    
                    # Calculate FPS
                    frame_times.append(process_time)
                    if len(frame_times) > 30:
                        frame_times.pop(0)
                    avg_process_time = sum(frame_times) / len(frame_times)
                    current_fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
                    
                    # Check database status
                    db_status = "Active"
                    if time.time() - last_db_check > 5:  # Check every 5 seconds
                        if self.enable_database:
                            # Could add database health checks here
                            pass
                        last_db_check = time.time()
                    
                    # Add info overlay
                    total_objects = len(tracking_data['objects'])
                    annotated_frame = self.add_info_overlay(annotated_frame, current_fps, total_objects, db_status)
                    
                    # Display or save frame based on mode
                    if self.headless:
                        # Save frame in headless mode
                        frame_filename = os.path.join(self.output_dir, f"frame_{self.frame_count:06d}.jpg")
                        cv2.imwrite(frame_filename, annotated_frame)
                        
                        # Save every 30th frame or frames with objects
                        if self.frame_count % 30 == 0 or total_objects > 0:
                            print(f"Frame {self.frame_count}: {total_objects} objects tracked - saved to {frame_filename}")
                    else:
                        # Display frame in GUI mode
                        cv2.imshow('Integrated Real-time Video Tracking', annotated_frame)
                    
                    self.frame_count += 1
                    
                    # Print tracking info every 30 frames
                    if self.frame_count % 30 == 0 and tracking_data['objects']:
                        print(f"Frame {self.frame_count}: {len(tracking_data['objects'])} objects tracked")
                        for obj in tracking_data['objects']:
                            print(f"  - {obj['class_name']} ID:{obj['tracker_id']} at ({obj['center']['x']:.1f}, {obj['center']['y']:.1f})")
                
                else:
                    # Show paused message (only in GUI mode)
                    if not self.headless:
                        paused_frame = frame.copy() if 'frame' in locals() else np.zeros((height, width, 3), dtype=np.uint8)
                        cv2.putText(paused_frame, "PAUSED - Press SPACE to resume", 
                                  (50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow('Integrated Real-time Video Tracking', paused_frame)
                
                # Handle keyboard input (only in GUI mode)
                if not self.headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save current frame
                        if not paused and 'annotated_frame' in locals():
                            filename = f"captured_frame_{int(time.time())}.jpg"
                            cv2.imwrite(filename, annotated_frame)
                            print(f"Frame saved as {filename}")
                    elif key == ord('r'):
                        # Reset tracker
                        self.tracker = sv.ByteTrack()
                        print("Tracker reset")
                    elif key == ord('i'):
                        # Show database info
                        if self.enable_database and self.session_id:
                            summary = self.db.get_session_summary(self.session_id)
                            print(f"Database Session Info: {summary}")
                    elif key == ord(' '):
                        # Toggle pause
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cap.release()
            
            # Safely destroy windows (handles cases where GUI is not available)
            if not self.headless:
                self._safe_destroy_windows()
            
            # Stop database processing
            if self.enable_database and self.data_processor:
                self.data_processor.stop_processing(self.frame_count)
                print(f"Database session ended: {self.session_id}")
            
            
            if self.headless:
                print(f"Headless processing complete! Check {self.output_dir} for saved frames.")
            else:
                print("Video processing complete!")

def parse_arguments():
    """Parse command line arguments"""
    # Set default paths relative to project root
    default_video = os.path.join(PROJECT_ROOT, "data", "Cropped_Vid_720p.mp4")
    default_model = os.path.join(PROJECT_ROOT, "models", "yolo11m.pt")
    default_db = os.path.join(PROJECT_ROOT, "databases", "tracking_data.db")
    
    parser = argparse.ArgumentParser(description="Integrated real-time video tracking with database storage")
    parser.add_argument("video_path", type=str, help="Path to the video file", 
                       nargs='?', default="../../data/Individual_2.mp4")
    parser.add_argument("--model", type=str, default=default_model,
                       help=f"Path to YOLO model file (default: {default_model})")
    parser.add_argument("--show-labels", action="store_true", 
                       help="Show class labels on bounding boxes")
    parser.add_argument("--ignore-classes", nargs="*", default=[], 
                       help="List of class names to ignore (e.g., --ignore-classes car truck)")
    parser.add_argument("--no-database", action="store_true", 
                       help="Disable database storage")
    parser.add_argument("--headless", action="store_true", 
                       help="Force headless mode (no GUI display)")
    parser.add_argument("--gui", action="store_true", 
                       help="Force GUI mode (override headless detection)")
    parser.add_argument("--force-gui", action="store_true", 
                       help="Force GUI mode and skip headless detection entirely")
    parser.add_argument("--db-path", type=str, default=default_db, 
                       help=f"Path to SQLite database file (default: {default_db})")
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Handle GUI/headless mode flags
    headless_mode = args.headless
    force_gui = args.gui or args.force_gui
    if args.gui:
        headless_mode = False
        print("Forcing GUI mode (--gui flag detected)")
    if args.force_gui:
        headless_mode = False
        print("Forcing GUI mode (--force-gui flag detected)")
    
    tracker = IntegratedRealtimeTracker(
        model_path=args.model,
        show_labels=args.show_labels,
        ignore_classes=args.ignore_classes,
        enable_database=not args.no_database,
        db_path=args.db_path,
        headless=headless_mode,
        force_gui=force_gui
    )
    
    tracker.run(args.video_path)

if __name__ == "__main__":
    main()
