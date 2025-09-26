# %%
# %%
"""
Vehicle Reidentification Experiment for UAV Videos
=================================================

This experimental script implements a comprehensive vehicle reidentification system
that works with the existing tracking infrastructure. It uses # %% cell markers
for Jupyter-style interactive development.

Key Features:
- Leverages existing YOLO + ByteTrack system
- Implements deep learning feature extraction
- Handles occlusion scenarios
- Provides comprehensive evaluation metrics
- Modular design for easy experimentation

Author: EDTH Hackathon 2025
"""

# %%
# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import supervision as sv
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import pandas as pd
from collections import defaultdict, deque
import os
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# %%
# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

class Config:
    """Configuration class for the reidentification system"""
    
    # Model paths
    YOLO_MODEL_PATH = "yolo11m.pt"
    
    # Feature extraction
    FEATURE_DIM = 2048  # ResNet-50 feature dimension
    MAX_FEATURES_PER_ID = 50  # Maximum features to store per object ID
    
    # Object-specific configurations
    OBJECT_CONFIGS = {
        'person': {
            'similarity_threshold': 0.6,  # People are more distinctive
            'temporal_window': 150,        # Longer memory for people
            'occlusion_threshold': 0.4,
            'trace_length': 0
        },
        'car': {
            'similarity_threshold': 0.6,  # Cars have moderate distinctiveness
            'temporal_window': 150,
            'occlusion_threshold': 0.3,
            'trace_length': 0
        },
        'truck': {
            'similarity_threshold': 0.6,  # Trucks are less distinctive
            'temporal_window': 150,
            'occlusion_threshold': 0.3,
            'trace_length': 0
        },
        'bicycle': {
            'similarity_threshold': 0.6, # Bicycles are fairly distinctive
            'temporal_window': 150,
            'occlusion_threshold': 0.35,
            'trace_length': 0
        },
        'default': {
            'similarity_threshold': 0.6,
            'temporal_window': 150,
            'occlusion_threshold': 0.3,
            'trace_length': 0
        }
    }
    
    # Default values (can be overridden by object-specific configs)
    SIMILARITY_THRESHOLD = 0.7
    TEMPORAL_WINDOW = 30
    OCCLUSION_THRESHOLD = 0.3
    
    # Visualization
    TRACE_LENGTH = 0
    BOX_THICKNESS = 2
    TEXT_SCALE = 0.6
    
    # Video processing
    TARGET_FPS = 30
    RESIZE_WIDTH = 1280
    RESIZE_HEIGHT = 720
    
    @classmethod
    def get_config_for_object(cls, object_class: str) -> dict:
        """Get configuration for specific object class"""
        return cls.OBJECT_CONFIGS.get(object_class, cls.OBJECT_CONFIGS['default'])

config = Config()
print("✅ Configuration loaded!")

# %%
# =============================================================================
# FEATURE EXTRACTION MODULE
# =============================================================================

class VehicleFeatureExtractor:
    """
    Deep learning-based feature extractor for vehicles using ResNet-50
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize the feature extractor
        
        Args:
            device: Device to run the model on ('cpu', 'cuda', or 'auto')
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Load pre-trained ResNet-50
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Feature extractor initialized on {self.device}")
    
    def extract_features(self, image: np.ndarray, bbox: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Extract features from a vehicle crop
        
        Args:
            image: Full frame image
            bbox: Bounding box [x1, y1, x2, y2]
            debug: Whether to print debug information
            
        Returns:
            Feature vector of shape (FEATURE_DIM,)
        """
        # Crop vehicle from image
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if debug:
            print(f"      Bbox: [{x1}, {y1}, {x2}, {y2}], Image shape: {image.shape}")
        
        if x2 <= x1 or y2 <= y1:
            if debug:
                print(f"      ❌ Invalid bbox dimensions")
            return np.zeros(config.FEATURE_DIM)
            
        vehicle_crop = image[y1:y2, x1:x2]
        
        if debug:
            print(f"      Crop shape: {vehicle_crop.shape}")
        
        if vehicle_crop.size == 0:
            if debug:
                print(f"      ❌ Empty crop")
            return np.zeros(config.FEATURE_DIM)
        
        # Preprocess and extract features
        try:
            input_tensor = self.transform(vehicle_crop).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(input_tensor)
                features = features.squeeze().cpu().numpy()
                
            # L2 normalize features
            features = features / (np.linalg.norm(features) + 1e-8)
            
            if debug:
                print(f"      ✅ Feature extracted: shape {features.shape}, norm {np.linalg.norm(features):.4f}")
            
            return features
            
        except Exception as e:
            if debug:
                print(f"      ❌ Feature extraction failed: {e}")
            return np.zeros(config.FEATURE_DIM)

# Initialize feature extractor
feature_extractor = VehicleFeatureExtractor()

# %%
# =============================================================================
# REIDENTIFICATION MODULE
# =============================================================================

class VehicleReidentifier:
    """
    Main reidentification system that works with ByteTrack
    """
    
    def __init__(self):
        """Initialize the reidentifier"""
        # Feature storage: {track_id: deque of features}
        self.feature_bank: Dict[int, deque] = defaultdict(lambda: deque(maxlen=config.MAX_FEATURES_PER_ID))
        
        # Tracking history: {track_id: {'last_seen': frame, 'features': [], 'bbox_history': []}}
        self.tracking_history: Dict[int, Dict] = defaultdict(lambda: {
            'last_seen': 0,
            'features': deque(maxlen=config.MAX_FEATURES_PER_ID),
            'bbox_history': deque(maxlen=10),
            'confidence_history': deque(maxlen=10),
            'class_name': 'unknown'
        })
        
        # Lost tracks: tracks that disappeared and might reappear
        self.lost_tracks: Dict[int, Dict] = {}
        
        # Reidentification statistics
        self.stats = {
            'total_detections': 0,
            'successful_reidentifications': 0,
            'failed_reidentifications': 0,
            'new_tracks': 0
        }
        
        # Track which IDs were reidentified in the current frame
        self.reidentified_this_frame = set()
        
        print("✅ Vehicle reidentifier initialized!")
    
    def update_tracking(self, detections: sv.Detections, frame_idx: int, frame: np.ndarray = None) -> sv.Detections:
        """
        Update tracking with reidentification capabilities
        
        Args:
            detections: Supervision detections with tracking IDs
            frame_idx: Current frame index
            frame: Current frame for feature extraction (required)
            
        Returns:
            Updated detections with reidentified tracks
        """
        if len(detections) == 0:
            return detections
            
        if frame is None:
            print(f"⚠️  Frame {frame_idx}: No frame provided for feature extraction")
            return detections
        
        # Clear reidentified tracks from previous frame
        self.reidentified_this_frame.clear()
            
        # Debug: Print detection info
        if frame_idx % 30 == 0:  # Every second at 30 FPS
            print(f"🔍 Frame {frame_idx}: {len(detections)} detections")
            if len(detections) > 0:
                print(f"   Track IDs: {detections.tracker_id}")
                print(f"   Class IDs: {detections.class_id if hasattr(detections, 'class_id') else 'None'}")
            
        # Extract features for all detections and create mapping
        detection_features = {}  # Maps detection index to feature
        for i, bbox in enumerate(detections.xyxy):
            debug_mode = (frame_idx % 30 == 0 and i == 0)  # Debug first detection every second
            feature = feature_extractor.extract_features(frame, bbox, debug=debug_mode)
            
            # Validate feature
            if np.linalg.norm(feature) < 1e-6:
                if debug_mode:
                    print(f"   ❌ Invalid feature (zero vector) for detection {i}")
                # Skip this detection - don't store feature
                continue
            
            # Store valid feature with its detection index
            detection_features[i] = feature
            
            if debug_mode:
                print(f"   ✅ Feature extracted: shape {feature.shape}, norm {np.linalg.norm(feature):.4f}")
        
        # Debug: Show feature extraction summary
        if frame_idx % 30 == 0:
            print(f"   📊 Feature extraction: {len(detection_features)}/{len(detections)} successful")
        
        # Update tracking history with valid features only
        for i, track_id in enumerate(detections.tracker_id):
            if track_id is not None and i in detection_features:
                # Get object class from detection
                class_id = detections.class_id[i] if hasattr(detections, 'class_id') and detections.class_id is not None else None
                class_name = self._get_class_name(class_id)
                
                self.tracking_history[track_id]['last_seen'] = frame_idx
                self.tracking_history[track_id]['features'].append(detection_features[i])
                self.tracking_history[track_id]['bbox_history'].append(detections.xyxy[i])
                self.tracking_history[track_id]['confidence_history'].append(detections.confidence[i])
                self.tracking_history[track_id]['class_name'] = class_name
                
                # Debug: Print tracking history info
                if frame_idx % 30 == 0 and i == 0:  # Debug first track every second
                    num_features = len(self.tracking_history[track_id]['features'])
                    print(f"   Track {track_id} ({class_name}): {num_features} features stored")
        
        # Check for lost tracks and attempt reidentification
        self._handle_lost_tracks(detections, frame_idx, frame)
        
        return detections
    
    def _handle_lost_tracks(self, detections: sv.Detections, frame_idx: int, frame: np.ndarray):
        """
        Handle tracks that have been lost and attempt reidentification
        """
        # Find tracks that haven't been seen recently
        current_track_ids = set(detections.tracker_id[detections.tracker_id != None])
        
        for track_id, history in self.tracking_history.items():
            if (track_id not in current_track_ids and 
                frame_idx - history['last_seen'] > 5 and
                track_id not in self.lost_tracks):
                
                # Only move to lost tracks if we have valid features
                if len(history['features']) > 0:
                    self.lost_tracks[track_id] = {
                        'last_seen': history['last_seen'],
                        'features': list(history['features']),
                        'bbox_history': list(history['bbox_history']),
                        'confidence_history': list(history['confidence_history']),
                        'class_name': history.get('class_name', 'unknown')
                    }
                    
                    if frame_idx % 30 == 0:
                        print(f"   📤 Track {track_id} moved to lost tracks ({len(history['features'])} features)")
        
        # Attempt reidentification for lost tracks
        if len(self.lost_tracks) > 0 and len(detections) > 0:
            self._attempt_reidentification(detections, frame_idx, frame)
    
    def _attempt_reidentification(self, detections: sv.Detections, frame_idx: int, frame: np.ndarray):
        """
        Attempt to reidentify lost tracks with current detections
        """
        if len(detections) == 0:
            return
            
        # Extract features for current detections
        current_features = []
        valid_detection_indices = []
        
        for i, bbox in enumerate(detections.xyxy):
            feature = feature_extractor.extract_features(frame, bbox, debug=False)
            
            # Validate feature
            if np.linalg.norm(feature) > 1e-6:
                current_features.append(feature)
                valid_detection_indices.append(i)
        
        if len(current_features) == 0:
            return
            
        current_features = np.array(current_features)
        
        for lost_id, lost_data in list(self.lost_tracks.items()):
            # Get object class for lost track
            lost_class = lost_data.get('class_name', 'unknown')
            obj_config = config.get_config_for_object(lost_class)
            
            # Check if enough time has passed for reidentification
            if frame_idx - lost_data['last_seen'] > obj_config['temporal_window']:
                # Remove from lost tracks (timeout)
                del self.lost_tracks[lost_id]
                self.stats['failed_reidentifications'] += 1
                if frame_idx % 30 == 0:
                    print(f"   ⏰ Track {lost_id} timed out (no reidentification)")
                continue
            
            # Get lost track features
            lost_features = np.array(list(lost_data['features']))
            if len(lost_features) == 0:
                continue
                
            # Calculate similarity between lost track and current detections
            # Use the most recent feature from lost track
            lost_feature = lost_features[-1].reshape(1, -1)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(lost_feature, current_features)[0]
            
            # Find best match above threshold
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            
            if best_similarity > obj_config['similarity_threshold']:
                # Reidentification successful!
                original_detection_idx = valid_detection_indices[best_match_idx]
                
                # Update the detection's track ID to the lost track ID
                detections.tracker_id[original_detection_idx] = lost_id
                
                # Move back to active tracking
                self.tracking_history[lost_id]['last_seen'] = frame_idx
                self.tracking_history[lost_id]['features'].append(current_features[best_match_idx])
                self.tracking_history[lost_id]['bbox_history'].append(detections.xyxy[original_detection_idx])
                self.tracking_history[lost_id]['confidence_history'].append(detections.confidence[original_detection_idx])
                
                # Remove from lost tracks
                del self.lost_tracks[lost_id]
                self.stats['successful_reidentifications'] += 1
                
                # Mark this track as reidentified this frame
                self.reidentified_this_frame.add(lost_id)
                
                if frame_idx % 30 == 0:
                    print(f"   ✅ Track {lost_id} reidentified! Similarity: {best_similarity:.3f}")
                    print(f"   🏷️  Track {lost_id} will show (REID) label this frame")
            else:
                if frame_idx % 30 == 0:
                    print(f"   🔍 Track {lost_id} no match found (best similarity: {best_similarity:.3f}, threshold: {obj_config['similarity_threshold']})")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reidentification statistics"""
        total_attempts = (self.stats['successful_reidentifications'] + 
                         self.stats['failed_reidentifications'])
        
        success_rate = (self.stats['successful_reidentifications'] / total_attempts 
                       if total_attempts > 0 else 0)
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'active_tracks': len(self.tracking_history),
            'lost_tracks': len(self.lost_tracks)
        }
    
    def debug_tracking_history(self):
        """Debug function to check tracking history state"""
        print("🔍 TRACKING HISTORY DEBUG:")
        print(f"   Total tracks: {len(self.tracking_history)}")
        
        for track_id, history in self.tracking_history.items():
            num_features = len(history['features'])
            class_name = history.get('class_name', 'unknown')
            last_seen = history['last_seen']
            print(f"   Track {track_id} ({class_name}): {num_features} features, last seen: {last_seen}")
            
            if num_features > 0:
                latest_feature = list(history['features'])[-1]
                if isinstance(latest_feature, np.ndarray):
                    print(f"      Latest feature: shape {latest_feature.shape}, norm {np.linalg.norm(latest_feature):.4f}")
                else:
                    print(f"      Latest feature: invalid type {type(latest_feature)}")
        
        print(f"   Lost tracks: {len(self.lost_tracks)}")
        for lost_id, lost_data in self.lost_tracks.items():
            num_features = len(lost_data['features'])
            class_name = lost_data.get('class_name', 'unknown')
            print(f"   Lost track {lost_id} ({class_name}): {num_features} features")
        
        print(f"   Reidentified this frame: {len(self.reidentified_this_frame)} tracks")
        if self.reidentified_this_frame:
            print(f"   Reidentified IDs: {list(self.reidentified_this_frame)}")

    def _get_class_name(self, class_id: int) -> str:
        """
        Convert class ID to class name
        """
        if class_id is None:
            return 'unknown'
            
        # YOLO COCO class names
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        if 0 <= class_id < len(coco_classes):
            return coco_classes[class_id]
        else:
            return 'unknown'

# Initialize reidentifier
reidentifier = VehicleReidentifier()

# Convenience function to debug the enhanced tracker's reidentifier
def debug_enhanced_tracker():
    """Debug the enhanced tracker's reidentifier"""
    print("🔍 DEBUGGING ENHANCED TRACKER:")
    enhanced_tracker.reidentifier.debug_tracking_history()

# %%
# =============================================================================
# ENHANCED TRACKING SYSTEM
# =============================================================================

class EnhancedTracker:
    """
    Enhanced tracking system that integrates reidentification with ByteTrack
    """
    
    def __init__(self, model_path: str = config.YOLO_MODEL_PATH):
        """Initialize the enhanced tracker"""
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.reidentifier = VehicleReidentifier()
        
        # Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=config.BOX_THICKNESS)
        self.label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.CENTER,
            text_scale=config.TEXT_SCALE
        )
        self.trace_annotator = sv.TraceAnnotator(trace_length=config.TRACE_LENGTH)
        
        # Statistics
        self.frame_count = 0
        self.processing_times = []
        
        print("✅ Enhanced tracker initialized!")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, sv.Detections, Dict]:
        """
        Process a single frame with detection, tracking, and reidentification
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (annotated_frame, detections, statistics)
        """
        start_time = cv2.getTickCount()
        
        # Run detection
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Update tracker
        detections = self.tracker.update_with_detections(detections)
        
        # Update reidentifier with frame for feature extraction
        detections = self.reidentifier.update_tracking(detections, self.frame_count, frame)
        
        # Create labels with enhanced information
        labels = self._create_labels(detections)
        
        # Annotate frame
        annotated_frame = self.box_annotator.annotate(frame, detections)
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels=labels)
        
        # Calculate processing time
        processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        self.processing_times.append(processing_time)
        
        # Get statistics
        stats = {
            'frame': self.frame_count,
            'processing_time': processing_time,
            'detections': len(detections),
            'tracks': len(set(detections.tracker_id[detections.tracker_id != None])),
            'reid_stats': self.reidentifier.get_statistics()
        }
        
        self.frame_count += 1
        
        return annotated_frame, detections, stats
    
    def _create_labels(self, detections: sv.Detections) -> List[str]:
        """Create enhanced labels for detections"""
        labels = []
        
        for i in range(len(detections)):
            if detections.tracker_id[i] is not None:
                # Check if this is a reidentified track
                track_id = detections.tracker_id[i]
                is_reidentified = track_id in self.reidentifier.reidentified_this_frame
                
                label = f"ID: {track_id}"
                if is_reidentified:
                    label += " (REID)"
                
                # Add confidence if available
                if hasattr(detections, 'confidence') and detections.confidence[i] is not None:
                    label += f" ({detections.confidence[i]:.2f})"
                    
                labels.append(label)
            else:
                labels.append("New")
        
        return labels

# Initialize enhanced tracker
enhanced_tracker = EnhancedTracker()

# %%
# =============================================================================
# VISUALIZATION AND ANALYSIS TOOLS
# =============================================================================

class ReidentificationAnalyzer:
    """
    Tools for analyzing and visualizing reidentification performance
    """
    
    def __init__(self):
        """Initialize the analyzer"""
        self.results_history = []
        self.feature_similarities = []
        
    def add_frame_results(self, stats: Dict):
        """Add frame results for analysis"""
        self.results_history.append(stats)
    
    def plot_processing_performance(self):
        """Plot processing performance over time"""
        if not self.results_history:
            print("No data to plot")
            return
            
        frames = [r['frame'] for r in self.results_history]
        processing_times = [r['processing_time'] for r in self.results_history]
        detections = [r['detections'] for r in self.results_history]
        tracks = [r['tracks'] for r in self.results_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Processing time
        axes[0, 0].plot(frames, processing_times)
        axes[0, 0].set_title('Processing Time per Frame')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].grid(True)
        
        # Number of detections
        axes[0, 1].plot(frames, detections)
        axes[0, 1].set_title('Number of Detections per Frame')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(True)
        
        # Number of tracks
        axes[1, 0].plot(frames, tracks)
        axes[1, 0].set_title('Number of Active Tracks per Frame')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True)
        
        # Reidentification statistics
        reid_stats = [r['reid_stats'] for r in self.results_history]
        success_rates = [s['success_rate'] for s in reid_stats]
        axes[1, 1].plot(frames, success_rates)
        axes[1, 1].set_title('Reidentification Success Rate')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_similarity_matrix(self, features: np.ndarray, labels: List[str]):
        """Plot feature similarity matrix"""
        if len(features) < 2:
            print("Need at least 2 features to plot similarity matrix")
            return
            
        similarity_matrix = cosine_similarity(features)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                   xticklabels=labels, 
                   yticklabels=labels,
                   annot=True, 
                   cmap='viridis',
                   fmt='.2f')
        plt.title('Feature Similarity Matrix')
        plt.tight_layout()
        plt.show()
    
    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        if not self.results_history:
            return {"error": "No data available"}
        
        # Calculate statistics
        total_frames = len(self.results_history)
        avg_processing_time = np.mean([r['processing_time'] for r in self.results_history])
        avg_detections = np.mean([r['detections'] for r in self.results_history])
        avg_tracks = np.mean([r['tracks'] for r in self.results_history])
        
        # Reidentification statistics
        latest_reid_stats = self.results_history[-1]['reid_stats']
        
        report = {
            'total_frames_processed': total_frames,
            'average_processing_time': avg_processing_time,
            'average_detections_per_frame': avg_detections,
            'average_tracks_per_frame': avg_tracks,
            'reidentification_statistics': latest_reid_stats,
            'estimated_fps': 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        }
        
        return report

# Initialize analyzer
analyzer = ReidentificationAnalyzer()

# %%
# =============================================================================
# DISPLAY DETECTION UTILITY
# =============================================================================

def _check_display_availability() -> bool:
    """
    Check if a display is available for cv2.imshow()
    
    Returns:
        True if display is available, False otherwise
    """
    try:
        # Try to create a small test window
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow('test', test_img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        return True
    except cv2.error:
        return False
    except Exception:
        return False

def view_saved_preview_frames(preview_dir: str = "preview_frames", max_frames: int = 20):
    """
    View saved preview frames (useful for remote machines)
    
    Args:
        preview_dir: Directory containing saved preview frames
        max_frames: Maximum number of frames to display
    """
    if not os.path.exists(preview_dir):
        print(f"❌ Preview directory not found: {preview_dir}")
        return
    
    # Get all preview frame files
    frame_files = [f for f in os.listdir(preview_dir) if f.endswith('.jpg')]
    frame_files.sort()
    
    if not frame_files:
        print(f"❌ No preview frames found in: {preview_dir}")
        return
    
    print(f"📸 Found {len(frame_files)} preview frames")
    print(f"🖼️  Displaying first {min(max_frames, len(frame_files))} frames...")
    
    # Display frames
    for i, frame_file in enumerate(frame_files[:max_frames]):
        frame_path = os.path.join(preview_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is not None:
            # Resize for display
            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow(f'Preview Frame {i+1}/{len(frame_files)}', display_frame)
            
            print(f"   Showing: {frame_file}")
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                print("⏹️  Stopped by user")
                break
            elif key == ord('s'):
                print("⏭️  Skipping remaining frames")
                break
    
    cv2.destroyAllWindows()
    print("✅ Preview viewing complete!")

# %%
# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_video_with_reidentification(video_path: str, output_path: str = None, 
                                      max_frames: int = None, show_preview: bool = True,
                                      preview_save_interval: int = 30):
    """
    Process a video with enhanced tracking and reidentification
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video (optional)
        max_frames: Maximum number of frames to process (for testing)
        show_preview: Whether to show real-time preview (or save preview frames if no display)
        preview_save_interval: Save preview frame every N frames when no display available
    """
    print(f"🎬 Processing video: {video_path}")
    
    # Check if display is available
    display_available = _check_display_availability()
    if show_preview and not display_available:
        print("🖥️  No display detected - will save preview frames instead")
        preview_dir = "preview_frames"
        os.makedirs(preview_dir, exist_ok=True)
        print(f"📁 Preview frames will be saved to: {preview_dir}/")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video info: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
    
    # Set up output video if specified
    out = None
    if output_path:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"📁 Created output directory: {output_dir}")
        
        # Use absolute path to avoid issues with working directory
        abs_output_path = os.path.abspath(output_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(abs_output_path, fourcc, fps, (width, height))
                
        if out:
            print(f"💾 Output will be saved to: {abs_output_path}")
        else:
            print(f"❌ Failed to initialize video writer for: {abs_output_path}")
            out = None
    
    # Process frames
    frame_count = 0
    target_frames = min(max_frames, total_frames) if max_frames else total_frames
    
    try:
        while frame_count < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, detections, stats = enhanced_tracker.process_frame(frame)
            
            # Add to analyzer
            analyzer.add_frame_results(stats)
            
            # Save frame if output specified
            if out:
                try:
                    out.write(annotated_frame)
                except Exception as e:
                    print(f"❌ Error writing frame {frame_count}: {e}")
                    # Continue processing but note the error
            
            # Show preview or save preview frames
            if show_preview:
                if display_available:
                    # Resize for display
                    display_frame = cv2.resize(annotated_frame, (1280, 720))
                    cv2.imshow('Vehicle Reidentification', display_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("⏹️  Processing stopped by user")
                        break
                else:
                    # Save preview frame every N frames
                    if frame_count % preview_save_interval == 0:
                        preview_filename = f"preview_frames/frame_{frame_count:06d}.jpg"
                        cv2.imwrite(preview_filename, annotated_frame)
                        if frame_count % (preview_save_interval * 10) == 0:  # Print every 10 saved frames
                            print(f"📸 Saved preview frame: {preview_filename}")
            
            # Progress update
            if frame_count % 30 == 0:  # Every second at 30 FPS
                progress = (frame_count / target_frames) * 100
                print(f"📈 Progress: {progress:.1f}% ({frame_count}/{target_frames})")
            
            frame_count += 1
    
    finally:
        cap.release()
        if out:
            out.release()
            # Check if output file was created successfully
            if os.path.exists(abs_output_path if output_path else ""):
                file_size = os.path.getsize(abs_output_path if output_path else "")
                print(f"✅ Video saved successfully: {abs_output_path if output_path else ''}")
                print(f"   File size: {file_size / (1024*1024):.2f} MB")
            else:
                print(f"❌ Output video file not found: {abs_output_path if output_path else ''}")
        if show_preview and display_available:
            cv2.destroyAllWindows()
        elif show_preview and not display_available:
            print(f"📁 Preview frames saved in: preview_frames/")
            print(f"   Total preview frames: {frame_count // preview_save_interval}")
    
    print(f"✅ Processing complete! Processed {frame_count} frames")
    
    # Generate and display report
    report = analyzer.generate_report()
    print("\n📋 PROCESSING REPORT:")
    print("=" * 50)
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")

# %%
# =============================================================================
# UNIVERSAL OBJECT REIDENTIFICATION EXAMPLES
# =============================================================================

# Object types supported by the system
supported_objects = list(config.OBJECT_CONFIGS.keys())
print(f"✅ Pre-configured object types: {supported_objects}")


# %%
# =============================================================================
# SIMILARITY ANALYSIS
# =============================================================================

def analyze_feature_similarities(reidentifier_instance=None):
    """Analyze feature similarities using real tracking data"""
    print("🔍 Analyzing feature similarities...")
    
    # Use the provided reidentifier instance or the global one
    if reidentifier_instance is None:
        reidentifier_instance = reidentifier
    
    if not hasattr(reidentifier_instance, 'tracking_history'):
        print("❌ No tracking history available")
        return
    
    # Use real tracking data
    features = []
    labels = []
    
    print(f"🔍 Checking tracking history: {len(reidentifier_instance.tracking_history)} tracks")
    
    for track_id, history in reidentifier_instance.tracking_history.items():
        num_features = len(history['features'])
        print(f"   Track {track_id}: {num_features} features")
        
        if len(history['features']) > 0:
            # Use the most recent feature for each track
            latest_feature = list(history['features'])[-1]
            
            if isinstance(latest_feature, np.ndarray) and latest_feature.size > 0:
                features.append(latest_feature)
                labels.append(f"ID_{track_id}")
                print(f"   ✅ Added feature for track {track_id}")
            else:
                print(f"   ❌ Invalid feature for track {track_id}")
        else:
            print(f"   ❌ No features for track {track_id}")
    
    print(f"📊 Total valid features found: {len(features)}")
    
    if len(features) < 2:
        print("❌ Need at least 2 features to analyze similarities")
        return
    
    features = np.array(features)
    
    # Plot similarity matrix
    analyzer.plot_feature_similarity_matrix(features, labels)
    
    # Calculate average similarity
    similarity_matrix = cosine_similarity(features)
    avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
    
    print(f"✅ Similarity analysis complete!")
    print(f"   Number of tracks: {len(features)}")
    print(f"   Average similarity: {avg_similarity:.4f}")
    print(f"   Max similarity: {similarity_matrix.max():.4f}")
    print(f"   Min similarity: {similarity_matrix.min():.4f}")
    
    # Show which tracks are most/least similar
    if len(features) > 2:
        # Find most similar pair
        upper_tri = np.triu_indices_from(similarity_matrix, k=1)
        max_sim_idx = np.argmax(similarity_matrix[upper_tri])
        i, j = upper_tri[0][max_sim_idx], upper_tri[1][max_sim_idx]
        print(f"   Most similar: {labels[i]} ↔ {labels[j]} ({similarity_matrix[i,j]:.4f})")
        
        # Find least similar pair
        min_sim_idx = np.argmin(similarity_matrix[upper_tri])
        i, j = upper_tri[0][min_sim_idx], upper_tri[1][min_sim_idx]
        print(f"   Least similar: {labels[i]} ↔ {labels[j]} ({similarity_matrix[i,j]:.4f})")

# %%
# =============================================================================
# EXPERIMENTAL TESTING CELLS
# =============================================================================

# Test with a sample video
if __name__ == "__main__":
    # You can run this cell to test the system
    video_path = "/home/alvaro/edth_hackathon_2025/data/Individual_1.mp4"  # Adjust path as needed
    
    if os.path.exists(video_path):
        print("🚀 Starting universal reidentification experiment...")
        process_video_with_reidentification(
            video_path=video_path,
            output_path="/home/alvaro/edth_hackathon_2025/data/reidentification_output.mp4",
            max_frames=None,  # Process first 10 seconds for testing
            show_preview=True,  # Will automatically detect display and save frames if needed
            preview_save_interval=30  # Save preview frame every 30 frames (1 second at 30 FPS)
        )
        
        # If you're on a remote machine and preview frames were saved, you can view them:
        # view_saved_preview_frames("preview_frames", max_frames=10)
        
        # Plot results
        analyzer.plot_processing_performance()
        
        # Analyze feature similarities with real vehicle IDs
        print("\n🔍 Analyzing feature similarities with real vehicle IDs...")
        # Use the reidentifier instance from the enhanced tracker
        analyze_feature_similarities(enhanced_tracker.reidentifier)
    else:
        print(f"❌ Video not found: {video_path}")
        # print("Available videos:")
        # for file in os.listdir("data"):
        #     if file.endswith(('.mp4', '.avi', '.mov')):
        #         print(f"  - {file}")



# %%
# =============================================================================
# CONFIGURATION AND PARAMETER TUNING
# =============================================================================

# Configuration parameters can be modified in the Config class
# Object-specific parameters are available in config.OBJECT_CONFIGS

print("\n🎉 Vehicle Reidentification System Ready!")
print("=" * 50)
print("🌍 SUPPORTED OBJECT TYPES:")
print("   ✅ People (person)")
print("   ✅ Vehicles (car, truck, bus, motorcycle, bicycle)")
print("   ✅ Animals (bird, cat, dog, horse, sheep, cow, etc.)")
print("   ✅ Objects (backpack, umbrella, handbag, etc.)")
print("   ✅ Any YOLO COCO class (80+ object types)")

print("\n🔧 KEY FEATURES:")
print("   • Object-specific parameter tuning")
print("   • Automatic class-based configuration")
print("   • Universal feature extraction (ResNet-50)")
print("   • Flexible similarity thresholds")
print("   • Adaptive temporal windows")
print("   • Real-time reidentification")

print("\n📋 USAGE:")
print("1. Use process_video_with_reidentification() to process videos")
print("2. Adjust object-specific parameters in Config.OBJECT_CONFIGS")
print("3. Call analyze_feature_similarities(enhanced_tracker.reidentifier) after processing")
print("4. Features are automatically extracted, stored, and compared")
print("5. Lost tracks are automatically reidentified when possible")

# %%
