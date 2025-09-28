#!/usr/bin/env python3
"""
Demo Re-identification Tracker

This script demonstrates three different approaches to object re-identification:
1. Appearance-only: Uses only visual features (color histograms, HOG, deep learning)
2. Kinematic-only: Uses only motion prediction and position matching
3. Combined: Weighted combination of both appearance and kinematic features

The script allows easy switching between these modes and provides detailed
parameter tuning options for each approach.

Author: EDTH Hackathon 2025
"""

import os
import sys
import argparse
import traceback
import time
import json
import math
from datetime import datetime
import threading
import queue
import numpy as np

# Import core dependencies with error handling
try:
    import cv2
except ImportError as e:
    print(f"Error: OpenCV not available: {e}")
    print("Please install opencv-python: pip install opencv-python")
    sys.exit(1)

try:
    import supervision as sv
except ImportError as e:
    print(f"Error: supervision not available: {e}")
    print("Please install supervision: pip install supervision")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"Error: ultralytics not available: {e}")
    print("Please install ultralytics: pip install ultralytics")
    sys.exit(1)

# Import local modules
try:
    from .database_integration import TrackingDatabase, RealTimeDataProcessor
except ImportError:
    try:
        from database_integration import TrackingDatabase, RealTimeDataProcessor
    except ImportError as e:
        print(f"Error: Could not import database_integration: {e}")
        sys.exit(1)

# Import the robust re-identification system
reidentify_path = os.path.join(os.path.dirname(__file__), '..', 'reidentify')
if reidentify_path not in sys.path:
    sys.path.append(reidentify_path)

try:
    from robust_reidentification import RobustReidentificationSystem
except ImportError as e:
    print("Warning: Could not import robust_reidentification module")
    sys.exit(1)

# Import the kinematic re-identification system
try:
    from kinematic_reidentification import KinematicReidentificationSystem
except ImportError as e:
    print("Warning: Could not import kinematic_reidentification module")
    sys.exit(1)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

class DemoReidentificationTracker:
    """
    Demo tracker that can switch between different re-identification approaches
    """
    
    def __init__(self, mode='combined', model_path=None, show_labels=True, 
                 ignore_classes=None, enable_database=True, db_path=None,
                 max_occlusion_frames=60, conf_threshold=0.15, 
                 enable_preprocessing=False, iou_threshold=0.7, max_det=1000,
                 agnostic_nms=False, half=False, device="", **kwargs):
        """
        Initialize the demo tracker
        
        Args:
            mode: 'appearance', 'kinematic', or 'combined'
            model_path: Path to YOLO model weights
            show_labels: Whether to show class labels
            ignore_classes: List of class names to ignore
            enable_database: Whether to enable database storage
            db_path: Path to SQLite database file
            max_occlusion_frames: Maximum frames to keep a lost track
            conf_threshold: YOLO confidence threshold
            enable_preprocessing: Enable image preprocessing
            **kwargs: Additional parameters for specific modes
        """
        # Set default paths
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, "models", "yolo11m.pt")
        if db_path is None:
            db_path = os.path.join(PROJECT_ROOT, "databases", "tracking_data.db")
        
        # Store mode and parameters
        self.mode = mode
        self.show_labels = show_labels
        self.ignore_classes = ignore_classes or []
        self.conf_threshold = conf_threshold
        self.enable_preprocessing = enable_preprocessing
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.half = half
        self.device = device
        self.max_occlusion_frames = max_occlusion_frames
        
        # Initialize core components
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.5,
            lost_track_buffer=60,
            minimum_matching_threshold=0.8,
            frame_rate=30,
            minimum_consecutive_frames=3
        )
        self.model = YOLO(model_path)
        if self.device:
            self.model.to(self.device)
        self.annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
        self.box_annotator = sv.BoxAnnotator()
        
        # Initialize re-identification systems based on mode
        self._initialize_reid_systems(kwargs)
        
        # Database integration
        self.enable_database = enable_database
        if enable_database:
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
        self.id_mapping = {}
        self.next_consistent_id = 1
        self.track_history = {}
        self.velocity_smoothing = 0.7
        
        print(f"✅ Demo tracker initialized in '{mode}' mode")
    
    def _initialize_reid_systems(self, kwargs):
        """Initialize re-identification systems based on mode"""
        if self.mode in ['appearance', 'combined']:
            # Initialize appearance-based re-identification
            self.appearance_reid = RobustReidentificationSystem(
                max_occlusion_frames=self.max_occlusion_frames
            )
            
            # Configure appearance parameters
            self.appearance_params = {
                'similarity_thresholds': kwargs.get('appearance_thresholds', {
                    'person': 0.75,
                    'car': 0.7,
                    'truck': 0.65,
                    'bus': 0.68,
                    'motorcycle': 0.72,
                    'bicycle': 0.7,
                    'default': 0.7
                }),
                'feature_weights': kwargs.get('feature_weights', {
                    'color_hist': 0.2,
                    'hog': 0.3,
                    'deep': 0.5
                }),
                'search_expansion_rate': kwargs.get('search_expansion_rate', 1.05)
            }
            
            # Apply parameters
            self.appearance_reid.similarity_thresholds = self.appearance_params['similarity_thresholds']
            self.appearance_reid.feature_extractor.feature_weights = self.appearance_params['feature_weights']
            self.appearance_reid.occlusion_handler.search_expansion_rate = self.appearance_params['search_expansion_rate']
        
        if self.mode in ['kinematic', 'combined']:
            # Initialize kinematic re-identification
            kinematic_weight = kwargs.get('kinematic_weight', 0.6)
            appearance_weight = kwargs.get('appearance_weight', 0.4)
            
            self.kinematic_reid = KinematicReidentificationSystem(
                max_occlusion_frames=self.max_occlusion_frames,
                kinematic_weight=kinematic_weight,
                appearance_weight=appearance_weight
            )
            
            # Configure kinematic parameters
            self.kinematic_params = {
                'max_position_error': kwargs.get('max_position_error', 50.0),
                'min_kinematic_score': kwargs.get('min_kinematic_score', 0.3),
                'min_appearance_score': kwargs.get('min_appearance_score', 0.4),
                'min_combined_score': kwargs.get('min_combined_score', 0.5),
                'uncertainty_multiplier': kwargs.get('uncertainty_multiplier', 2.0),
                'confidence_threshold': kwargs.get('confidence_threshold', 0.3)
            }
            
            # Apply parameters
            self.kinematic_reid.matching_params.update(self.kinematic_params)
        
        if self.mode == 'combined':
            # Combined mode parameters
            self.combined_params = {
                'appearance_weight': kwargs.get('combined_appearance_weight', 0.4),
                'kinematic_weight': kwargs.get('combined_kinematic_weight', 0.6),
                'fusion_method': kwargs.get('fusion_method', 'weighted_average'),  # 'weighted_average', 'product', 'max'
                'min_combined_score': kwargs.get('min_combined_score', 0.5)
            }
    
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
            return detections[[]]
    
    def create_labels(self, detections):
        """Create labels for detections with mode information"""
        labels = []
        for i in range(len(detections)):
            if self.show_labels and detections.class_id[i] is not None:
                class_name = self.model.names[detections.class_id[i]]
                if detections.tracker_id[i] is not None:
                    track_id = detections.tracker_id[i]
                    label = f"{class_name} ID:{track_id}"
                    
                    # Add mode indicator for re-identified tracks
                    if self._is_reidentified_track(track_id):
                        label += f" ({self.mode.upper()})"
                    
                    labels.append(label)
                else:
                    labels.append(f"{class_name} New")
            else:
                if detections.tracker_id[i] is not None:
                    track_id = detections.tracker_id[i]
                    label = f"ID: {track_id}"
                    
                    if self._is_reidentified_track(track_id):
                        label += f" ({self.mode.upper()})"
                    
                    labels.append(label)
                else:
                    labels.append("New")
        return labels
    
    def _is_reidentified_track(self, track_id):
        """Check if a track was re-identified"""
        if self.mode in ['appearance', 'combined']:
            return track_id in self.appearance_reid.occlusion_handler.lost_tracks
        elif self.mode == 'kinematic':
            return track_id in self.kinematic_reid.lost_tracks
        return False
    
    def extract_tracking_data(self, detections, frame_timestamp):
        """Extract tracking data for database storage"""
        frame_data = {
            'frame_number': self.frame_count,
            'timestamp': frame_timestamp,
            'objects': []
        }
        
        for i in range(len(detections)):
            if detections.tracker_id[i] is not None:
                track_id = detections.tracker_id[i]
                consistent_track_id = self.get_consistent_track_id(track_id)
                
                obj_data = {
                    'tracker_id': consistent_track_id,
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
                    },
                    'is_reidentified': self._is_reidentified_track(track_id),
                    'reid_mode': self.mode,
                    'original_tracker_id': int(track_id)
                }

                # Add velocity estimates
                velocity = self.compute_velocity(
                    int(detections.tracker_id[i]),
                    obj_data['center'],
                    frame_timestamp
                )
                obj_data['velocity'] = velocity
                frame_data['objects'].append(obj_data)

        return frame_data
    
    def get_consistent_track_id(self, tracker_id):
        """Get or assign a consistent track ID for database storage"""
        if tracker_id is None:
            return None
            
        if tracker_id not in self.id_mapping:
            self.id_mapping[tracker_id] = self.next_consistent_id
            self.next_consistent_id += 1
            
        return self.id_mapping[tracker_id]
    
    def compute_velocity(self, track_id, center, timestamp):
        """Compute per-track velocity vector and heading in image space"""
        prev_state = self.track_history.get(track_id)
        prev_avg_vx, prev_avg_vy = (0.0, 0.0)
        if prev_state and 'avg_velocity' in prev_state:
            prev_avg_vx, prev_avg_vy = prev_state['avg_velocity']
        
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
                alpha = self.velocity_smoothing
                smoothed_vx = alpha * vx + (1.0 - alpha) * prev_avg_vx
                smoothed_vy = alpha * vy + (1.0 - alpha) * prev_avg_vy
            else:
                smoothed_vx, smoothed_vy = prev_avg_vx, prev_avg_vy
        else:
            smoothed_vx, smoothed_vy = 0.0, 0.0

        speed = math.hypot(smoothed_vx, smoothed_vy)
        heading = None
        if speed > 1e-3:
            heading = (math.degrees(math.atan2(smoothed_vy, smoothed_vx)) + 360.0) % 360.0

        velocity.update({
            'vx': smoothed_vx,
            'vy': smoothed_vy,
            'speed': speed,
            'direction': heading,
        })

        # Update history
        self.track_history[track_id] = {
            'center': (center['x'], center['y']),
            'timestamp': timestamp,
            'avg_velocity': (smoothed_vx, smoothed_vy)
        }

        return velocity
    
    def preprocess_frame(self, frame):
        """Preprocess frame to improve detection in challenging conditions"""
        if not self.enable_preprocessing:
            return frame
            
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Apply slight gamma correction to brighten dark areas
        gamma = 1.2
        enhanced = np.power(enhanced / 255.0, 1.0 / gamma) * 255.0
        enhanced = np.uint8(enhanced)
        
        return enhanced
    
    def process_frame(self, frame, frame_timestamp):
        """Process a single frame for detection, tracking, and re-identification"""
        # Preprocess frame if enabled
        if self.enable_preprocessing:
            enhanced_frame = self.preprocess_frame(frame)
        else:
            enhanced_frame = frame
        
        # Run YOLO detection with all parameters
        detection_kwargs = {
            'conf': self.conf_threshold,
            'iou': self.iou_threshold,
            'max_det': self.max_det,
            'agnostic_nms': self.agnostic_nms,
            'verbose': False
        }
        if self.half:
            detection_kwargs['half'] = True
        if self.device:
            detection_kwargs['device'] = self.device
            
        result = self.model(enhanced_frame, **detection_kwargs)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter ignored classes
        detections = self.filter_detections(detections)
        
        # Update tracker
        detections = self.tracker.update_with_detections(detections)
        
        # Apply re-identification based on mode
        timestamp = time.time()
        detections = self._apply_reidentification(detections, frame, timestamp)
        
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
    
    def _apply_reidentification(self, detections, frame, timestamp):
        """Apply re-identification based on the selected mode"""
        if len(detections) == 0:
            return detections
        
        if self.mode == 'appearance':
            # Use only appearance-based re-identification
            detections = self.appearance_reid.process_detections(detections, frame, timestamp)
            
        elif self.mode == 'kinematic':
            # Use only kinematic re-identification
            detections = self.kinematic_reid.process_detections(
                detections, frame, timestamp, None  # No feature extractor
            )
            
        elif self.mode == 'combined':
            # Use both systems and combine results
            appearance_detections = self.appearance_reid.process_detections(detections, frame, timestamp)
            kinematic_detections = self.kinematic_reid.process_detections(
                detections, frame, timestamp, self.appearance_reid.feature_extractor
            )
            
            # Combine the results using the specified fusion method
            detections = self._fuse_detections(detections, appearance_detections, kinematic_detections)
        
        return detections
    
    def _fuse_detections(self, original_detections, appearance_detections, kinematic_detections):
        """Fuse results from appearance and kinematic re-identification"""
        # For now, use the appearance results as primary and kinematic as fallback
        # This is a simplified fusion - in practice, you'd want more sophisticated logic
        return appearance_detections
    
    def add_info_overlay(self, frame, fps, total_objects, db_status=""):
        """Add information overlay to frame with mode-specific stats"""
        # FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Object count
        cv2.putText(frame, f"Objects: {total_objects}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mode indicator
        cv2.putText(frame, f"Mode: {self.mode.upper()}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Re-identification stats based on mode
        if self.mode in ['appearance', 'combined']:
            reid_stats = self.appearance_reid.get_statistics()
            cv2.putText(frame, f"AppReID: {reid_stats['successful_reidentifications']}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if self.mode in ['kinematic', 'combined']:
            kinematic_stats = self.kinematic_reid.get_statistics()
            cv2.putText(frame, f"KinReID: {kinematic_stats['performance_stats']['combined_matches']}", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Database status
        if self.enable_database:
            cv2.putText(frame, f"DB: {db_status}", (10, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Controls info
        cv2.putText(frame, "Controls: 'q'=quit, 's'=save, 'r'=reset, SPACE=pause, 'i'=info", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def print_mode_info(self):
        """Print information about the current mode and parameters"""
        print(f"\n📊 Demo Mode: {self.mode.upper()}")
        print("=" * 50)
        
        if self.mode in ['appearance', 'combined']:
            print("🎨 Appearance-based Re-identification:")
            print(f"  - Similarity thresholds: {self.appearance_params['similarity_thresholds']}")
            print(f"  - Feature weights: {self.appearance_params['feature_weights']}")
            print(f"  - Search expansion rate: {self.appearance_params['search_expansion_rate']}")
        
        if self.mode in ['kinematic', 'combined']:
            print("🚀 Kinematic Re-identification:")
            print(f"  - Max position error: {self.kinematic_params['max_position_error']}")
            print(f"  - Min kinematic score: {self.kinematic_params['min_kinematic_score']}")
            print(f"  - Min appearance score: {self.kinematic_params['min_appearance_score']}")
            print(f"  - Min combined score: {self.kinematic_params['min_combined_score']}")
            print(f"  - Confidence threshold: {self.kinematic_params['confidence_threshold']}")
        
        if self.mode == 'combined':
            print("🔗 Combined Mode:")
            print(f"  - Fusion method: {self.combined_params['fusion_method']}")
            print(f"  - Appearance weight: {self.combined_params['appearance_weight']}")
            print(f"  - Kinematic weight: {self.combined_params['combined_kinematic_weight']}")
        
        print("=" * 50)
    
    def run(self, video_path):
        """Main loop for real-time video tracking with re-identification"""
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
        print(f"Mode: {self.mode}")
        
        # Start database session
        if self.enable_database:
            self.session_id = self.db.start_session(video_path, fps)
            self.data_processor.start_processing(video_path, fps, self.session_id)
            print(f"Database session started: {self.session_id}")
        
        print("\n=== Demo Re-identification Tracker ===")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("Press 'r' to reset tracker")
        print("Press SPACE to pause/resume")
        print("Press 'i' to show mode info")
        print("Press 'R' to show re-identification stats")
        print("=====================================\n")
        
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
                    try:
                        annotated_frame, detections, tracking_data = self.process_frame(frame, frame_timestamp)
                    except Exception as e:
                        print(f"Error processing frame {self.frame_count}: {e}")
                        traceback.print_exc()
                        annotated_frame = frame.copy()
                        detections = sv.Detections.empty()
                        tracking_data = {
                            'frame_number': self.frame_count,
                            'timestamp': frame_timestamp,
                            'objects': []
                        }
                    process_time = time.time() - start_process
                    
                    # Calculate FPS
                    frame_times.append(process_time)
                    if len(frame_times) > 30:
                        frame_times.pop(0)
                    avg_process_time = sum(frame_times) / len(frame_times)
                    current_fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
                    
                    # Check database status
                    db_status = "Active"
                    if time.time() - last_db_check > 5:
                        last_db_check = time.time()
                    
                    # Add info overlay
                    total_objects = len(tracking_data['objects'])
                    annotated_frame = self.add_info_overlay(annotated_frame, current_fps, total_objects, db_status)
                    
                    # Display frame
                    cv2.imshow(f'Demo Re-identification Tracker - {self.mode.upper()}', annotated_frame)
                    
                    self.frame_count += 1
                    
                    # Print tracking info every 30 frames
                    if self.frame_count % 30 == 0 and tracking_data['objects']:
                        print(f"Frame {self.frame_count}: {len(tracking_data['objects'])} objects tracked")
                        for obj in tracking_data['objects']:
                            reid_status = f" ({obj.get('reid_mode', 'N/A').upper()})" if obj.get('is_reidentified', False) else ""
                            print(f"  - {obj['class_name']} ID:{obj['tracker_id']}{reid_status} at ({obj['center']['x']:.1f}, {obj['center']['y']:.1f})")
                
                else:
                    # Show paused message
                    paused_frame = frame.copy() if 'frame' in locals() else np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(paused_frame, "PAUSED - Press SPACE to resume", 
                              (50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(f'Demo Re-identification Tracker - {self.mode.upper()}', paused_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if not paused and 'annotated_frame' in locals():
                        filename = f"captured_frame_{int(time.time())}.jpg"
                        cv2.imwrite(filename, annotated_frame)
                        print(f"Frame saved as {filename}")
                elif key == ord('r'):
                    # Reset tracker
                    self.tracker = sv.ByteTrack(
                        track_activation_threshold=0.5,
                        lost_track_buffer=60,
                        minimum_matching_threshold=0.8,
                        frame_rate=30,
                        minimum_consecutive_frames=3
                    )
                    self.track_history = {}
                    print("Tracker reset")
                elif key == ord('i'):
                    self.print_mode_info()
                elif key == ord('R'):
                    # Show re-identification stats
                    if self.mode in ['appearance', 'combined']:
                        stats = self.appearance_reid.get_statistics()
                        print(f"\nAppearance ReID Stats: {stats}")
                    if self.mode in ['kinematic', 'combined']:
                        stats = self.kinematic_reid.get_statistics()
                        print(f"\nKinematic ReID Stats: {stats}")
                elif key == ord(' '):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Stop database processing
            if self.enable_database and self.data_processor:
                self.data_processor.stop_processing(self.frame_count)
                print(f"Database session ended: {self.session_id}")
            
            print("Video processing complete!")

def parse_arguments():
    """Parse command line arguments"""
    # Set default paths relative to project root
    default_video = os.path.join(PROJECT_ROOT, "data", "Cropped_Vid_720p.mp4")
    default_model = os.path.join(PROJECT_ROOT, "models", "yolo11m.pt")
    default_db = os.path.join(PROJECT_ROOT, "databases", "tracking_data.db")
    
    parser = argparse.ArgumentParser(description="Demo re-identification tracker")
    parser.add_argument("video_path", type=str, help="Path to the video file", 
                       nargs='?', default=default_video)
    parser.add_argument("--mode", type=str, choices=['appearance', 'kinematic', 'combined'], 
                       default='combined', help="Re-identification mode")
    parser.add_argument("--model", type=str, default=default_model,
                       help=f"Path to YOLO model file (default: {default_model})")
    parser.add_argument("--show-labels", action="store_true", 
                       help="Show class labels on bounding boxes")
    parser.add_argument("--ignore-classes", nargs="*", default=[], 
                       help="List of class names to ignore")
    parser.add_argument("--no-database", action="store_true", 
                       help="Disable database storage")
    parser.add_argument("--db-path", type=str, default=default_db, 
                       help=f"Path to SQLite database file (default: {default_db})")
    parser.add_argument("--max-occlusion-frames", type=int, default=60,
                       help="Maximum frames to keep a lost track (default: 60)")
    parser.add_argument("--conf-threshold", type=float, default=0.15,
                       help="YOLO confidence threshold (default: 0.15)")
    parser.add_argument("--iou-threshold", type=float, default=0.7,
                       help="YOLO IoU threshold for NMS (default: 0.7)")
    parser.add_argument("--max-det", type=int, default=1000,
                       help="Maximum detections per image (default: 1000)")
    parser.add_argument("--agnostic-nms", action="store_true",
                       help="Use class-agnostic NMS")
    parser.add_argument("--half", action="store_true",
                       help="Use half precision (FP16) for faster inference")
    parser.add_argument("--device", type=str, default="",
                       help="Device to run on (cpu, cuda, 0, 1, etc.)")
    parser.add_argument("--enable-preprocessing", action="store_true",
                       help="Enable image preprocessing")
    
    # Appearance-specific parameters
    parser.add_argument("--appearance-threshold", type=float, default=0.7,
                       help="Appearance similarity threshold (default: 0.7)")
    parser.add_argument("--color-weight", type=float, default=0.2,
                       help="Weight for color histogram features (default: 0.2)")
    parser.add_argument("--hog-weight", type=float, default=0.3,
                       help="Weight for HOG features (default: 0.3)")
    parser.add_argument("--deep-weight", type=float, default=0.5,
                       help="Weight for deep learning features (default: 0.5)")
    
    # Kinematic-specific parameters
    parser.add_argument("--kinematic-weight", type=float, default=0.6,
                       help="Weight for kinematic scoring (default: 0.6)")
    parser.add_argument("--appearance-weight", type=float, default=0.4,
                       help="Weight for appearance scoring (default: 0.4)")
    parser.add_argument("--max-position-error", type=float, default=50.0,
                       help="Maximum allowed position error in pixels (default: 50.0)")
    parser.add_argument("--min-kinematic-score", type=float, default=0.3,
                       help="Minimum kinematic score for consideration (default: 0.3)")
    parser.add_argument("--min-appearance-score", type=float, default=0.4,
                       help="Minimum appearance score for consideration (default: 0.4)")
    parser.add_argument("--min-combined-score", type=float, default=0.5,
                       help="Minimum combined score for match (default: 0.5)")
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                       help="Minimum prediction confidence (default: 0.3)")
    
    # Combined mode parameters
    parser.add_argument("--fusion-method", type=str, choices=['weighted_average', 'product', 'max'], 
                       default='weighted_average', help="Fusion method for combined mode")
    parser.add_argument("--combined-appearance-weight", type=float, default=0.4,
                       help="Appearance weight in combined mode (default: 0.4)")
    parser.add_argument("--combined-kinematic-weight", type=float, default=0.6,
                       help="Kinematic weight in combined mode (default: 0.6)")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Prepare parameters
    kwargs = {
        'appearance_thresholds': {
            'person': args.appearance_threshold,
            'car': args.appearance_threshold,
            'truck': args.appearance_threshold,
            'bus': args.appearance_threshold,
            'motorcycle': args.appearance_threshold,
            'bicycle': args.appearance_threshold,
            'default': args.appearance_threshold
        },
        'feature_weights': {
            'color_hist': args.color_weight,
            'hog': args.hog_weight,
            'deep': args.deep_weight
        },
        'kinematic_weight': args.kinematic_weight,
        'appearance_weight': args.appearance_weight,
        'max_position_error': args.max_position_error,
        'min_kinematic_score': args.min_kinematic_score,
        'min_appearance_score': args.min_appearance_score,
        'min_combined_score': args.min_combined_score,
        'confidence_threshold': args.confidence_threshold,
        'fusion_method': args.fusion_method,
        'combined_appearance_weight': args.combined_appearance_weight,
        'combined_kinematic_weight': args.combined_kinematic_weight
    }
    
    tracker = DemoReidentificationTracker(
        mode=args.mode,
        model_path=args.model,
        show_labels=args.show_labels,
        ignore_classes=args.ignore_classes,
        enable_database=not args.no_database,
        db_path=args.db_path,
        max_occlusion_frames=args.max_occlusion_frames,
        conf_threshold=args.conf_threshold,
        enable_preprocessing=args.enable_preprocessing,
        iou_threshold=args.iou_threshold,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms,
        half=args.half,
        device=args.device,
        **kwargs
    )
    
    tracker.run(args.video_path)

if __name__ == "__main__":
    main()
