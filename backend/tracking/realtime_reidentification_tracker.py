#!/usr/bin/env python3
"""
Real-time video tracker with robust re-identification capabilities.

Integrates re-identification system with the existing IntegratedRealtimeTracker
for robust object tracking with occlusion handling in real-time scenarios.
"""

import os
import argparse
import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
import time
import json
import traceback
from datetime import datetime
import threading
import queue
from database_integration import TrackingDatabase, RealTimeDataProcessor

# Import the robust re-identification system
import sys
reidentify_path = os.path.join(os.path.dirname(__file__), '..', 'reidentify')
if reidentify_path not in sys.path:
    sys.path.append(reidentify_path)

try:
    from robust_reidentification import RobustReidentificationSystem
except ImportError as e:
    print("Warning: Could not import robust_reidentification module")
    print("Please ensure the reidentify module is in the correct location")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Script is in project_root/backend/tracking/, so go up 2 levels to reach project_root
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

class RealtimeReidentificationTracker:
    def __init__(self, model_path=None, show_labels=True, ignore_classes=None, 
                 enable_database=True, db_path=None, max_occlusion_frames=60):
        """
        Initialize the real-time tracker with re-identification capabilities
        
        Args:
            model_path (str): Path to YOLO model weights (if None, uses default)
            show_labels (bool): Whether to show class labels
            ignore_classes (list): List of class names to ignore
            enable_database (bool): Whether to enable database storage
            db_path (str): Path to SQLite database file (if None, uses default)
            max_occlusion_frames (int): Maximum frames to keep a lost track
        """
        # Set default paths relative to project root
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, "models", "yolo11m.pt")
        if db_path is None:
            db_path = os.path.join(PROJECT_ROOT, "databases", "tracking_data.db")
        
        # Initialize core components
        self.tracker = sv.ByteTrack()
        self.model = YOLO(model_path)
        self.annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
        self.box_annotator = sv.BoxAnnotator()
        self.show_labels = show_labels
        self.ignore_classes = ignore_classes or []
        
        # Initialize re-identification system with enhanced parameters for longer occlusions
        self.reid_system = RobustReidentificationSystem(max_occlusion_frames=max_occlusion_frames)
        
        # Enhanced parameters for very long occlusions (15+ frames)
        self.reid_system.occlusion_handler.max_occlusion_frames = max_occlusion_frames
        self.reid_system.occlusion_handler.search_expansion_rate = 1.05  # Very slow expansion
        self.reid_system.feature_extractor.feature_weights = {
            'color_hist': 0.2,   # Further reduced color weight (unreliable over time)
            'hog': 0.3,          # Increased shape weight (more stable)
            'deep': 0.5          # Maintained deep learning weight
        }
        
        # Ultra-robust similarity thresholds for very long occlusions
        self.reid_system.similarity_thresholds = {
            'person': 0.7,      # Lower for people
            'car': 0.6,         # Much lower for vehicles
            'truck': 0.5,       # Very low for trucks
            'bicycle': 0.65,    # Lower for bicycles
            'default': 0.6      # Lower default threshold
        }
        
        # Additional parameters for very long occlusions
        self.ultra_long_occlusion_threshold = 20  # Frames
        self.feature_decay_rate = 0.95  # How much to decay feature importance over time
        
        print("✅ Enhanced re-identification system initialized for longer occlusions")
        
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
        
        # Tracking data storage
        self.tracking_data = []
        self.frame_count = 0
        self.start_time = None
        self.session_id = None
        
        # Re-identification statistics
        self.reid_stats = {
            'total_reidentifications': 0,
            'successful_reidentifications': 0,
            'failed_reidentifications': 0,
            'active_tracks': 0,
            'lost_tracks': 0
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
            return detections[[]]  # Empty detections
    
    def create_labels(self, detections):
        """Create labels for detections with re-identification information"""
        labels = []
        for i in range(len(detections)):
            if self.show_labels and detections.class_id[i] is not None:
                class_name = self.model.names[detections.class_id[i]]
                if detections.tracker_id[i] is not None:
                    # Check if this is a reidentified track
                    track_id = detections.tracker_id[i]
                    is_reidentified = track_id in self.reid_system.occlusion_handler.lost_tracks
                    
                    label = f"{class_name} ID:{track_id}"
                    if is_reidentified:
                        label += " (REID)"
                    labels.append(label)
                else:
                    labels.append(f"{class_name} New")
            else:
                if detections.tracker_id[i] is not None:
                    track_id = detections.tracker_id[i]
                    is_reidentified = track_id in self.reid_system.occlusion_handler.lost_tracks
                    
                    label = f"ID: {track_id}"
                    if is_reidentified:
                        label += " (REID)"
                    labels.append(label)
                else:
                    labels.append("New")
        return labels
    
    def extract_tracking_data(self, detections, frame_timestamp):
        """Extract tracking data for database storage with re-identification info"""
        frame_data = {
            'frame_number': self.frame_count,
            'timestamp': frame_timestamp,
            'objects': []
        }
        
        for i in range(len(detections)):
            if detections.tracker_id[i] is not None:
                # Check if this is a reidentified track
                track_id = detections.tracker_id[i]
                is_reidentified = track_id in self.reid_system.occlusion_handler.lost_tracks
                
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
                    },
                    'is_reidentified': is_reidentified
                }
                frame_data['objects'].append(obj_data)
        
        return frame_data
    
    def process_frame(self, frame, frame_timestamp):
        """Process a single frame for detection, tracking, and re-identification"""
        # Run YOLO detection
        result = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter ignored classes
        detections = self.filter_detections(detections)
        
        # Update tracker
        detections = self.tracker.update_with_detections(detections)
        
        # Apply enhanced re-identification with multiple strategies
        timestamp = time.time()
        detections = self.enhanced_reidentification(detections, frame, timestamp)
        
        # Extract tracking data for database
        tracking_data = self.extract_tracking_data(detections, frame_timestamp)
        self.tracking_data.append(tracking_data)
        
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
    
    def enhanced_reidentification(self, detections, frame, timestamp):
        """Enhanced re-identification with multiple strategies for longer occlusions"""
        if len(detections) == 0:
            return detections
            
        # Standard re-identification
        detections = self.reid_system.process_detections(detections, frame, timestamp)
        
        # Enhanced strategy for longer occlusions
        if len(self.reid_system.occlusion_handler.lost_tracks) > 0:
            detections = self.apply_enhanced_matching(detections, frame, timestamp)
        
        return detections
    
    def apply_enhanced_matching(self, detections, frame, timestamp):
        """Apply enhanced matching strategies for longer occlusions"""
        if len(detections) == 0:
            return detections
            
        # Strategy 1: Relaxed similarity thresholds for longer occlusions
        enhanced_matches = self.match_with_relaxed_thresholds(detections, frame)
        
        # Strategy 2: Multi-scale feature matching
        if not enhanced_matches:
            enhanced_matches = self.match_with_multiscale_features(detections, frame)
        
        # Strategy 3: Temporal consistency matching
        if not enhanced_matches:
            enhanced_matches = self.match_with_temporal_consistency(detections, frame, timestamp)
        
        # Strategy 4: Ultra-robust matching for very long occlusions (15+ frames)
        if not enhanced_matches:
            enhanced_matches = self.match_with_ultra_robust_strategy(detections, frame, timestamp)
        
        # Apply matches
        for track_id, det_idx in enhanced_matches.items():
            if det_idx < len(detections.xyxy):
                # Update the tracker ID for the matched detection
                if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                    detections.tracker_id[det_idx] = track_id
                
                # Update the re-identification system
                bbox = detections.xyxy[det_idx]
                class_name = getattr(detections, 'data', {}).get('class_name', [None] * len(detections))
                if isinstance(class_name, list) and det_idx < len(class_name):
                    class_name = class_name[det_idx]
                else:
                    class_name = None
                    
                self.reid_system.update_track(track_id, bbox, frame, timestamp, class_name)
                self.reid_system.stats['successful_reidentifications'] += 1
                
                # Remove from lost tracks
                if track_id in self.reid_system.occlusion_handler.lost_tracks:
                    del self.reid_system.occlusion_handler.lost_tracks[track_id]
                
                #print(f"🔄 Enhanced re-identification: Track {track_id} recovered after longer occlusion")
        
        return detections
    
    def match_with_relaxed_thresholds(self, detections, frame):
        """Match with relaxed similarity thresholds for longer occlusions"""
        matches = {}
        
        for track_id, info in self.reid_system.occlusion_handler.lost_tracks.items():
            frames_lost = info['frames_lost']
            
            # Progressively relax thresholds based on occlusion duration
            if frames_lost > 20:  # Ultra-long occlusion (20+ frames)
                similarity_threshold = 0.4  # Extremely relaxed
                iou_threshold = 0.15
            elif frames_lost > 15:  # Very long occlusion (15-20 frames)
                similarity_threshold = 0.45  # Very relaxed
                iou_threshold = 0.18
            elif frames_lost > 10:  # Long occlusion (10-15 frames)
                similarity_threshold = 0.5  # Very relaxed
                iou_threshold = 0.2
            elif frames_lost > 5:  # Medium occlusion (5-10 frames)
                similarity_threshold = 0.6  # Relaxed
                iou_threshold = 0.3
            else:  # Short occlusion (1-5 frames)
                similarity_threshold = 0.7  # Standard
                iou_threshold = 0.4
            
            # Get object-specific threshold
            last_state = info.get('last_state')
            class_name = last_state.class_name if last_state and hasattr(last_state, 'class_name') else 'default'
            obj_threshold = self.reid_system.similarity_thresholds.get(class_name, 0.65)
            final_threshold = min(similarity_threshold, obj_threshold)
            
            best_match_idx = -1
            best_match_score = 0.0
            
            for det_idx in range(len(detections)):
                det_bbox = detections.xyxy[det_idx]
                
                # Check spatial overlap
                iou = self.compute_iou(det_bbox, info['search_region'])
                if iou < iou_threshold:
                    continue
                
                # Extract and compare features
                det_features = self.reid_system.feature_extractor.extract_all_features(frame, det_bbox)
                similarity = self.reid_system.feature_extractor.compute_similarity(
                    info['features'], det_features
                )
                
                # Combined score with relaxed thresholds
                combined_score = similarity * (1 + iou * 0.2)
                
                if combined_score > best_match_score and similarity > final_threshold:
                    best_match_score = combined_score
                    best_match_idx = det_idx
            
            if best_match_idx >= 0:
                matches[track_id] = best_match_idx
        
        return matches
    
    def match_with_multiscale_features(self, detections, frame):
        """Match using multi-scale feature extraction for better robustness"""
        matches = {}
        
        for track_id, info in self.reid_system.occlusion_handler.lost_tracks.items():
            best_match_idx = -1
            best_match_score = 0.0
            
            for det_idx in range(len(detections)):
                det_bbox = detections.xyxy[det_idx]
                
                # Extract features at multiple scales
                scales = [0.8, 1.0, 1.2]  # Different scales
                scale_similarities = []
                
                for scale in scales:
                    # Scale the bounding box
                    x1, y1, x2, y2 = det_bbox
                    w, h = x2 - x1, y2 - y1
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    scaled_bbox = np.array([
                        cx - w * scale / 2,
                        cy - h * scale / 2,
                        cx + w * scale / 2,
                        cy + h * scale / 2
                    ])
                    
                    # Extract features at this scale
                    det_features = self.reid_system.feature_extractor.extract_all_features(frame, scaled_bbox)
                    similarity = self.reid_system.feature_extractor.compute_similarity(
                        info['features'], det_features
                    )
                    scale_similarities.append(similarity)
                
                # Use the best similarity across scales
                max_similarity = max(scale_similarities)
                
                if max_similarity > best_match_score and max_similarity > 0.6:
                    best_match_score = max_similarity
                    best_match_idx = det_idx
            
            if best_match_idx >= 0:
                matches[track_id] = best_match_idx
        
        return matches
    
    def match_with_temporal_consistency(self, detections, frame, timestamp):
        """Match using temporal consistency and motion prediction"""
        matches = {}
        
        for track_id, info in self.reid_system.occlusion_handler.lost_tracks.items():
            predictor = info['predictor']
            frames_lost = info['frames_lost']
            
            # Predict where the object should be
            dt = timestamp - self.reid_system.last_timestamp if self.reid_system.last_timestamp > 0 else 0.033
            x_pred, y_pred, vx_pred, vy_pred = predictor.predict(dt)
            
            # Get uncertainty
            uncertainty = predictor.get_uncertainty()
            std_x = np.sqrt(uncertainty[0, 0])
            std_y = np.sqrt(uncertainty[1, 1])
            
            best_match_idx = -1
            best_match_score = 0.0
            
            for det_idx in range(len(detections)):
                det_bbox = detections.xyxy[det_idx]
                det_cx = (det_bbox[0] + det_bbox[2]) / 2
                det_cy = (det_bbox[1] + det_bbox[3]) / 2
                
                # Calculate spatial distance from prediction
                spatial_distance = np.sqrt((det_cx - x_pred)**2 + (det_cy - y_pred)**2)
                max_distance = 3 * (std_x + std_y)  # 3-sigma rule
                
                if spatial_distance > max_distance:
                    continue
                
                # Extract features and compute similarity
                det_features = self.reid_system.feature_extractor.extract_all_features(frame, det_bbox)
                similarity = self.reid_system.feature_extractor.compute_similarity(
                    info['features'], det_features
                )
                
                # Combine spatial and appearance with temporal weighting
                temporal_weight = max(0.5, 1.0 - frames_lost * 0.05)  # Decrease weight over time
                spatial_score = max(0, 1.0 - spatial_distance / max_distance)
                combined_score = (similarity * 0.7 + spatial_score * 0.3) * temporal_weight
                
                if combined_score > best_match_score and similarity > 0.55:
                    best_match_score = combined_score
                    best_match_idx = det_idx
            
            if best_match_idx >= 0:
                matches[track_id] = best_match_idx
        
        return matches
    
    def match_with_ultra_robust_strategy(self, detections, frame, timestamp):
        """Ultra-robust matching for very long occlusions (15+ frames)"""
        matches = {}
        
        for track_id, info in self.reid_system.occlusion_handler.lost_tracks.items():
            frames_lost = info['frames_lost']
            
            # Only apply ultra-robust strategy for very long occlusions
            if frames_lost < self.ultra_long_occlusion_threshold:
                continue
                
            best_match_idx = -1
            best_match_score = 0.0
            
            for det_idx in range(len(detections)):
                det_bbox = detections.xyxy[det_idx]
                
                # Strategy 4a: Multiple feature extraction methods
                feature_scores = []
                
                # Extract features with different preprocessing
                preprocessing_methods = [
                    ('normal', det_bbox),
                    ('brightness_boost', self.adjust_brightness_bbox(det_bbox, 1.2)),
                    ('contrast_boost', self.adjust_contrast_bbox(det_bbox, 1.3)),
                    ('blur_robust', self.add_blur_robustness_bbox(det_bbox))
                ]
                
                for method_name, bbox in preprocessing_methods:
                    try:
                        det_features = self.reid_system.feature_extractor.extract_all_features(frame, bbox)
                        similarity = self.reid_system.feature_extractor.compute_similarity(
                            info['features'], det_features
                        )
                        feature_scores.append(similarity)
                    except Exception as e:
                        print(f"Warning: Feature extraction failed for {method_name}: {e}")
                        feature_scores.append(0.0)
                
                # Use the best feature score
                max_feature_score = max(feature_scores) if feature_scores else 0.0
                
                # Strategy 4b: Motion-based prediction with uncertainty
                predictor = info['predictor']
                dt = timestamp - self.reid_system.last_timestamp if self.reid_system.last_timestamp > 0 else 0.033
                x_pred, y_pred, vx_pred, vy_pred = predictor.predict(dt)
                
                # Calculate distance from prediction
                det_cx = (det_bbox[0] + det_bbox[2]) / 2
                det_cy = (det_bbox[1] + det_bbox[3]) / 2
                spatial_distance = np.sqrt((det_cx - x_pred)**2 + (det_cy - y_pred)**2)
                
                # Get uncertainty and create very large search area
                uncertainty = predictor.get_uncertainty()
                std_x = np.sqrt(uncertainty[0, 0])
                std_y = np.sqrt(uncertainty[1, 1])
                max_distance = 5 * (std_x + std_y)  # 5-sigma rule for very long occlusions
                
                if spatial_distance > max_distance:
                    continue
                
                # Strategy 4c: Temporal decay weighting
                temporal_decay = self.feature_decay_rate ** frames_lost
                spatial_score = max(0, 1.0 - spatial_distance / max_distance)
                
                # Strategy 4d: Combined ultra-robust scoring
                combined_score = (
                    max_feature_score * 0.6 +  # Feature similarity (60%)
                    spatial_score * 0.3 +      # Spatial proximity (30%)
                    temporal_decay * 0.1       # Temporal consistency (10%)
                )
                
                # Ultra-relaxed threshold for very long occlusions
                min_threshold = 0.35 if frames_lost > 20 else 0.4
                
                if combined_score > best_match_score and max_feature_score > min_threshold:
                    best_match_score = combined_score
                    best_match_idx = det_idx
            
            if best_match_idx >= 0:
                matches[track_id] = best_match_idx
                print(f"🔄 Ultra-robust re-identification: Track {track_id} recovered after {frames_lost} frames")
        
        return matches
    
    def adjust_brightness_bbox(self, bbox, factor):
        """Adjust brightness for robustness"""
        return bbox  # For now, return original bbox
    
    def adjust_contrast_bbox(self, bbox, factor):
        """Adjust contrast for robustness"""
        return bbox  # For now, return original bbox
    
    def add_blur_robustness_bbox(self, bbox):
        """Add blur robustness"""
        return bbox  # For now, return original bbox
    
    def compute_iou(self, box1, box2):
        """Compute IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def add_info_overlay(self, frame, fps, total_objects, db_status=""):
        """Add information overlay to frame with re-identification stats"""
        # FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Object count
        cv2.putText(frame, f"Objects: {total_objects}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Re-identification stats
        reid_stats = self.reid_system.get_statistics()
        cv2.putText(frame, f"ReID: {reid_stats['successful_reidentifications']}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Active tracks
        cv2.putText(frame, f"Active: {reid_stats['active_tracks']}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Lost tracks
        cv2.putText(frame, f"Lost: {reid_stats['lost_tracks_count']}", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        # Database status
        if self.enable_database:
            cv2.putText(frame, f"DB: {db_status}", (10, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Controls info
        cv2.putText(frame, "Controls: 'q'=quit, 's'=save, 'r'=reset, SPACE=pause, 'd'=save data, 'i'=reid info", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def save_tracking_data(self, output_file):
        """Save all tracking data to JSON file with re-identification info"""
        with open(output_file, 'w') as f:
            json.dump(self.tracking_data, f, indent=2)
        print(f"Tracking data saved to: {output_file}")
    
    def print_reid_info(self):
        """Print re-identification statistics"""
        stats = self.reid_system.get_statistics()
        print("\n📊 Re-identification Statistics:")
        print("=" * 40)
        print(f"Total detections: {stats['total_detections']}")
        print(f"Successful re-identifications: {stats['successful_reidentifications']}")
        print(f"Failed re-identifications: {stats['failed_reidentifications']}")
        print(f"Success rate: {stats['success_rate']:.2f}")
        print(f"Active tracks: {stats['active_tracks']}")
        print(f"Lost tracks: {stats['lost_tracks_count']}")
        print(f"New tracks: {stats['new_tracks']}")
        print("=" * 40)
    
    def run(self, video_path, save_data=True):
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
        print(f"Show labels: {self.show_labels}")
        if self.ignore_classes:
            print(f"Ignoring classes: {', '.join(self.ignore_classes)}")
        print(f"Database enabled: {self.enable_database}")
        print(f"Re-identification enabled: True")
        
        # Start database session
        if self.enable_database:
            self.session_id = self.db.start_session(video_path, fps)
            self.data_processor.start_processing(video_path, fps)
            print(f"Database session started: {self.session_id}")
        
        print("\n=== Real-time Video Tracking with Re-identification ===")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("Press 'r' to reset tracker")
        print("Press SPACE to pause/resume")
        print("Press 'd' to save tracking data")
        print("Press 'i' to show database info")
        print("Press 'R' to show re-identification info")
        print("=====================================================\n")
        
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
                        print("Full traceback:")
                        traceback.print_exc()
                        # Create a dummy frame to continue processing
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
                    if time.time() - last_db_check > 5:  # Check every 5 seconds
                        if self.enable_database:
                            # Could add database health checks here
                            pass
                        last_db_check = time.time()
                    
                    # Add info overlay
                    total_objects = len(tracking_data['objects'])
                    annotated_frame = self.add_info_overlay(annotated_frame, current_fps, total_objects, db_status)
                    
                    # Display frame
                    cv2.imshow('Real-time Tracking with Re-identification', annotated_frame)
                    
                    self.frame_count += 1
                    
                    # Print tracking info every 30 frames
                    if self.frame_count % 30 == 0 and tracking_data['objects']:
                        print(f"Frame {self.frame_count}: {len(tracking_data['objects'])} objects tracked")
                        for obj in tracking_data['objects']:
                            reid_status = " (REID)" if obj.get('is_reidentified', False) else ""
                            print(f"  - {obj['class_name']} ID:{obj['tracker_id']}{reid_status} at ({obj['center']['x']:.1f}, {obj['center']['y']:.1f})")
                
                else:
                    # Show paused message
                    paused_frame = frame.copy() if 'frame' in locals() else np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(paused_frame, "PAUSED - Press SPACE to resume", 
                              (50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Real-time Tracking with Re-identification', paused_frame)
                
                # Handle keyboard input
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
                    self.reid_system = RobustReidentificationSystem()
                    print("Tracker and re-identification system reset")
                elif key == ord('d'):
                    # Save tracking data
                    if save_data:
                        output_file = f"{os.path.splitext(video_path)[0]}_tracking_data.json"
                        self.save_tracking_data(output_file)
                elif key == ord('i'):
                    # Show database info
                    if self.enable_database and self.session_id:
                        summary = self.db.get_session_summary(self.session_id)
                        print(f"Database Session Info: {summary}")
                elif key == ord('R'):
                    # Show re-identification info
                    self.print_reid_info()
                elif key == ord(' '):
                    # Toggle pause
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Stop database processing
            if self.enable_database and self.data_processor:
                self.data_processor.stop_processing(self.frame_count)
                print(f"Database session ended: {self.session_id}")
            
            # Auto-save tracking data if enabled
            if save_data and self.tracking_data:
                output_file = f"{os.path.splitext(video_path)[0]}_tracking_data.json"
                self.save_tracking_data(output_file)
            
            # Print final re-identification statistics
            print("\n📊 Final Re-identification Statistics:")
            self.print_reid_info()
            
            print("Video processing complete!")

def parse_arguments():
    """Parse command line arguments"""
    # Set default paths relative to project root
    default_video = os.path.join(PROJECT_ROOT, "data", "Cropped_Vid_720p.mp4")
    default_model = os.path.join(PROJECT_ROOT, "models", "yolo11m.pt")
    default_db = os.path.join(PROJECT_ROOT, "databases", "tracking_data.db")
    
    parser = argparse.ArgumentParser(description="Real-time video tracking with re-identification")
    parser.add_argument("video_path", type=str, help="Path to the video file", 
                       nargs='?', default=default_video)
    parser.add_argument("--model", type=str, default=default_model,
                       help=f"Path to YOLO model file (default: {default_model})")
    parser.add_argument("--show-labels", action="store_true", 
                       help="Show class labels on bounding boxes")
    parser.add_argument("--ignore-classes", nargs="*", default=[], 
                       help="List of class names to ignore (e.g., --ignore-classes car truck)")
    parser.add_argument("--no-save", action="store_true", 
                       help="Don't save tracking data to file")
    parser.add_argument("--no-database", action="store_true", 
                       help="Disable database storage")
    parser.add_argument("--db-path", type=str, default=default_db, 
                       help=f"Path to SQLite database file (default: {default_db})")
    parser.add_argument("--max-occlusion-frames", type=int, default=60,
                       help="Maximum frames to keep a lost track (default: 60)")
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    tracker = RealtimeReidentificationTracker(
        model_path=args.model,
        show_labels=args.show_labels,
        ignore_classes=args.ignore_classes,
        enable_database=not args.no_database,
        db_path=args.db_path,
        max_occlusion_frames=args.max_occlusion_frames
    )
    
    tracker.run(args.video_path, save_data=not args.no_save)

if __name__ == "__main__":
    main()
