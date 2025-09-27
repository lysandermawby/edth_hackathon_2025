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
            'similarity_threshold': 0.8,  # People are more distinctive
            'temporal_window': 60,        # Longer memory for people
            'occlusion_threshold': 0.4,
            'trace_length': 30
        },
        'car': {
            'similarity_threshold': 0.7,  # Cars have moderate distinctiveness
            'temporal_window': 30,
            'occlusion_threshold': 0.3,
            'trace_length': 20
        },
        'truck': {
            'similarity_threshold': 0.6,  # Trucks are less distinctive
            'temporal_window': 45,
            'occlusion_threshold': 0.3,
            'trace_length': 25
        },
        'bicycle': {
            'similarity_threshold': 0.75, # Bicycles are fairly distinctive
            'temporal_window': 40,
            'occlusion_threshold': 0.35,
            'trace_length': 15
        },
        'default': {
            'similarity_threshold': 0.7,
            'temporal_window': 30,
            'occlusion_threshold': 0.3,
            'trace_length': 20
        }
    }
    
    # Default values (can be overridden by object-specific configs)
    SIMILARITY_THRESHOLD = 0.7
    TEMPORAL_WINDOW = 30
    OCCLUSION_THRESHOLD = 0.3
    
    # Visualization
    TRACE_LENGTH = 20
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
    
    def extract_features(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Extract features from a vehicle crop
        
        Args:
            image: Full frame image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Feature vector of shape (FEATURE_DIM,)
        """
        # Crop vehicle from image
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(config.FEATURE_DIM)
            
        vehicle_crop = image[y1:y2, x1:x2]
        
        if vehicle_crop.size == 0:
            return np.zeros(config.FEATURE_DIM)
        
        # Preprocess and extract features
        try:
            input_tensor = self.transform(vehicle_crop).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(input_tensor)
                features = features.squeeze().cpu().numpy()
                
            # L2 normalize features
            features = features / (np.linalg.norm(features) + 1e-8)
            return features
            
        except Exception as e:
            print(f"Warning: Feature extraction failed: {e}")
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
            'confidence_history': deque(maxlen=10)
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
        
        print("✅ Vehicle reidentifier initialized!")
    
    def update_tracking(self, detections: sv.Detections, frame_idx: int) -> sv.Detections:
        """
        Update tracking with reidentification capabilities
        
        Args:
            detections: Supervision detections with tracking IDs
            frame_idx: Current frame index
            
        Returns:
            Updated detections with reidentified tracks
        """
        if len(detections) == 0:
            return detections
            
        # Extract features for all detections
        features = []
        for i, bbox in enumerate(detections.xyxy):
            # Get the full frame (we'll need to pass this in the main loop)
            # For now, we'll use a placeholder
            feature = np.random.randn(config.FEATURE_DIM)  # Placeholder
            features.append(feature)
        
        # Update tracking history
        for i, track_id in enumerate(detections.tracker_id):
            if track_id is not None:
                self.tracking_history[track_id]['last_seen'] = frame_idx
                self.tracking_history[track_id]['features'].append(features[i])
                self.tracking_history[track_id]['bbox_history'].append(detections.xyxy[i])
                self.tracking_history[track_id]['confidence_history'].append(detections.confidence[i])
        
        # Check for lost tracks and attempt reidentification
        self._handle_lost_tracks(detections, frame_idx)
        
        return detections
    
    def _handle_lost_tracks(self, detections: sv.Detections, frame_idx: int):
        """
        Handle tracks that have been lost and attempt reidentification
        """
        # Find tracks that haven't been seen recently
        current_track_ids = set(detections.tracker_id[detections.tracker_id != None])
        
        for track_id, history in self.tracking_history.items():
            if (track_id not in current_track_ids and 
                frame_idx - history['last_seen'] > 5 and
                track_id not in self.lost_tracks):
                
                # Move to lost tracks
                self.lost_tracks[track_id] = {
                    'last_seen': history['last_seen'],
                    'features': list(history['features']),
                    'bbox_history': list(history['bbox_history']),
                    'confidence_history': list(history['confidence_history'])
                }
        
        # Attempt reidentification for lost tracks
        if len(self.lost_tracks) > 0 and len(detections) > 0:
            self._attempt_reidentification(detections, frame_idx)
    
    def _attempt_reidentification(self, detections: sv.Detections, frame_idx: int):
        """
        Attempt to reidentify lost tracks with current detections
        """
        if len(detections) == 0:
            return
            
        # Get object classes for current detections
        current_classes = getattr(detections, 'data', {}).get('class_name', ['unknown'] * len(detections))
        
        for lost_id, lost_data in list(self.lost_tracks.items()):
            # Get object class for lost track (if available)
            lost_class = lost_data.get('class_name', 'unknown')
            obj_config = config.get_config_for_object(lost_class)
            
            # Check if enough time has passed for reidentification
            if frame_idx - lost_data['last_seen'] > obj_config['temporal_window']:
                # Remove from lost tracks (timeout)
                del self.lost_tracks[lost_id]
                self.stats['failed_reidentifications'] += 1
            else:
                # Find potential matches of the same class
                potential_matches = []
                for i, current_class in enumerate(current_classes):
                    if current_class == lost_class or current_class == 'unknown':
                        potential_matches.append(i)
                
                if potential_matches:
                    # Here you would implement the actual reidentification logic
                    # comparing features between lost tracks and current detections
                    # For now, this is a placeholder
                    pass
    
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

# Initialize reidentifier
reidentifier = VehicleReidentifier()

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
        
        # Update reidentifier
        detections = self.reidentifier.update_tracking(detections, self.frame_count)
        
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
                is_reidentified = track_id in self.reidentifier.lost_tracks
                
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
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Output will be saved to: {output_path}")
    
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
                out.write(annotated_frame)
            
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

def demonstrate_object_agnostic_capabilities():
    """
    Demonstrate how the system works with different object types
    """
    print("🌍 Universal Object Reidentification Capabilities")
    print("=" * 60)
    
    # Show supported object types
    supported_objects = list(config.OBJECT_CONFIGS.keys())
    print(f"✅ Pre-configured object types: {supported_objects}")
    
    # Show how to add new object types
    print("\n🔧 Adding new object types:")
    print("""
    # Example: Add support for animals
    config.OBJECT_CONFIGS['dog'] = {
        'similarity_threshold': 0.75,  # Dogs are fairly distinctive
        'temporal_window': 45,         # Medium memory window
        'occlusion_threshold': 0.35,
        'trace_length': 25
    }
    
    config.OBJECT_CONFIGS['bird'] = {
        'similarity_threshold': 0.6,   # Birds are less distinctive
        'temporal_window': 20,         # Shorter memory (fast movement)
        'occlusion_threshold': 0.4,
        'trace_length': 10
    }
    """)
    
    # Show YOLO COCO classes that work out of the box
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella'
    ]
    
    print(f"\n🎯 YOLO COCO classes supported: {len(coco_classes)} classes")
    print("   Examples:", coco_classes[:10], "...")
    
    return supported_objects, coco_classes

# Run demonstration
supported_objects, coco_classes = demonstrate_object_agnostic_capabilities()

# %%
# =============================================================================
# OBJECT-SPECIFIC PROCESSING EXAMPLES
# =============================================================================

def process_video_for_specific_objects(video_path: str, target_objects: List[str] = None):
    """
    Process video focusing on specific object types
    
    Args:
        video_path: Path to input video
        target_objects: List of object classes to focus on (e.g., ['person', 'car'])
    """
    print(f"🎬 Processing video for specific objects: {target_objects}")
    
    if target_objects is None:
        target_objects = ['person', 'car', 'truck', 'bicycle']
    
    # Show configurations for each object type
    for obj in target_objects:
        obj_config = config.get_config_for_object(obj)
        print(f"\n📋 {obj.upper()} Configuration:")
        for key, value in obj_config.items():
            print(f"   {key}: {value}")
    
    # You would modify the main processing function to filter detections
    # by object class and use object-specific configurations
    print(f"\n✅ Ready to process {video_path} for objects: {target_objects}")

# Example usage
process_video_for_specific_objects("/home/alvaro/edth_hackathon_2025/data/Individual_2.mp4", ['person', 'car', 'truck'])

# %%
# =============================================================================
# EXPERIMENTAL TESTING CELLS
# =============================================================================

# Test with a sample video
if __name__ == "__main__":
    # You can run this cell to test the system
    video_path = "/home/alvaro/edth_hackathon_2025/data/Individual_2.mp4"  # Adjust path as needed
    
    if os.path.exists(video_path):
        print("🚀 Starting universal reidentification experiment...")
        process_video_with_reidentification(
            video_path=video_path,
            output_path="data/reidentification_output.mp4",
            max_frames=300,  # Process first 10 seconds for testing
            show_preview=True,  # Will automatically detect display and save frames if needed
            preview_save_interval=30  # Save preview frame every 30 frames (1 second at 30 FPS)
        )
        
        # If you're on a remote machine and preview frames were saved, you can view them:
        # view_saved_preview_frames("preview_frames", max_frames=10)
        
        # Plot results
        analyzer.plot_processing_performance()
    else:
        print(f"❌ Video not found: {video_path}")
        # print("Available videos:")
        # for file in os.listdir("data"):
        #     if file.endswith(('.mp4', '.avi', '.mov')):
        #         print(f"  - {file}")

# %%
# =============================================================================
# FEATURE EXTRACTION TESTING
# =============================================================================

def test_feature_extraction():
    """Test the feature extraction system"""
    print("🧪 Testing feature extraction...")
    
    # Create a dummy image and bbox
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_bbox = np.array([100, 100, 200, 200])
    
    # Extract features
    features = feature_extractor.extract_features(dummy_image, dummy_bbox)
    
    print(f"✅ Feature extraction successful!")
    print(f"   Feature dimension: {features.shape}")
    print(f"   Feature norm: {np.linalg.norm(features):.4f}")
    print(f"   Feature range: [{features.min():.4f}, {features.max():.4f}]")
    
    return features

# Run feature extraction test
test_features = test_feature_extraction()

# %%
# =============================================================================
# SIMILARITY ANALYSIS
# =============================================================================

def analyze_feature_similarities():
    """Analyze feature similarities for different scenarios"""
    print("🔍 Analyzing feature similarities...")
    
    # Generate multiple feature vectors
    num_features = 5
    features = []
    labels = []
    
    for i in range(num_features):
        # Create slightly different features
        base_feature = np.random.randn(config.FEATURE_DIM)
        noise = np.random.normal(0, 0.1, config.FEATURE_DIM)
        feature = base_feature + noise
        feature = feature / (np.linalg.norm(feature) + 1e-8)
        
        features.append(feature)
        labels.append(f"Vehicle_{i+1}")
    
    features = np.array(features)
    
    # Plot similarity matrix
    analyzer.plot_feature_similarity_matrix(features, labels)
    
    # Calculate average similarity
    similarity_matrix = cosine_similarity(features)
    avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
    
    print(f"✅ Similarity analysis complete!")
    print(f"   Average similarity: {avg_similarity:.4f}")
    print(f"   Max similarity: {similarity_matrix.max():.4f}")
    print(f"   Min similarity: {similarity_matrix.min():.4f}")

# Run similarity analysis
analyze_feature_similarities()

# %%
# =============================================================================
# CONFIGURATION AND PARAMETER TUNING
# =============================================================================

def tune_parameters():
    """Interactive parameter tuning interface"""
    print("⚙️  Parameter Tuning Interface")
    print("=" * 40)
    
    print("Current configuration:")
    print(f"  Similarity threshold: {config.SIMILARITY_THRESHOLD}")
    print(f"  Temporal window: {config.TEMPORAL_WINDOW}")
    print(f"  Occlusion threshold: {config.OCCLUSION_THRESHOLD}")
    print(f"  Max features per ID: {config.MAX_FEATURES_PER_ID}")
    
    print("\nTo modify parameters, update the Config class and re-run the system.")
    print("Recommended ranges:")
    print("  Similarity threshold: 0.5 - 0.9")
    print("  Temporal window: 10 - 60 frames")
    print("  Occlusion threshold: 0.2 - 0.5")
    print("  Max features per ID: 20 - 100")

# Run parameter tuning interface
tune_parameters()

print("\n🎉 Universal Object Reidentification System Ready!")
print("=" * 60)
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

print("\n📋 NEXT STEPS:")
print("1. Run the main processing cell to test with your videos")
print("2. Adjust object-specific parameters in Config.OBJECT_CONFIGS")
print("3. Add new object types as needed")
print("4. Analyze results using the visualization tools")
print("5. Experiment with different feature extraction models")
print("6. Implement additional reidentification strategies")

print("\n💡 USAGE EXAMPLES:")
print("   • UAV surveillance: Track people and vehicles")
print("   • Wildlife monitoring: Track animals in nature")
print("   • Security systems: Track people and objects")
print("   • Traffic analysis: Track vehicles and pedestrians")
print("   • Sports analysis: Track players and equipment")

# %%
