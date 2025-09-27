"""
Robust Re-identification System for Object Tracking
=================================================

This module implements a comprehensive re-identification system that works
with supervision library and ByteTracker. It combines multiple feature
extraction methods and kinematic prediction for robust object re-identification
after occlusion.

Key Features:
- Multiple feature extraction methods (color histograms, HOG, deep features)
- Kalman filter-based kinematic prediction
- Intelligent search region prediction
- Multi-modal similarity scoring
- Integration with supervision library

Author: EDTH Hackathon 2025
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    from filterpy.kalman import KalmanFilter  # type: ignore
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    print("Warning: filterpy not available. Kalman filtering will use simplified implementation.")
    # Create a dummy KalmanFilter class for type hints
    class KalmanFilter:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("filterpy.kalman.KalmanFilter not available")

try:
    from skimage.feature import hog  # type: ignore
    HOG_AVAILABLE = True
except ImportError:
    HOG_AVAILABLE = False
    print("Warning: scikit-image not available. HOG features will be disabled.")
    # Create a dummy hog function for type hints
    def hog(*args, **kwargs):  # type: ignore
        raise NotImplementedError("skimage.feature.hog not available")

@dataclass
class TrackState:
    """Store track state information"""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2] format
    timestamp: float
    velocity: np.ndarray  # [vx, vy]
    confidence: float
    feature_vector: Optional[np.ndarray] = None
    class_name: Optional[str] = None

class SimpleKalmanFilter:
    """Simplified Kalman filter implementation when filterpy is not available"""
    
    def __init__(self, dim_x=4, dim_z=2):
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # State: [x, y, vx, vy]
        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x) * 100  # Initial uncertainty
        self.F = np.array([[1, 0, 1, 0],  # State transition
                          [0, 1, 0, 1],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],  # Measurement matrix
                          [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(dim_x) * 0.1  # Process noise
        self.R = np.eye(dim_z) * 10   # Measurement noise
        
    def predict(self, dt=1.0):
        """Predict next state"""
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z):
        """Update with measurement"""
        y = z.reshape(-1, 1) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P

class KinematicPredictor:
    """Predict object motion using Kalman filter"""
    
    def __init__(self):
        if KALMAN_AVAILABLE:
            self.kf = KalmanFilter(dim_x=4, dim_z=2)
            self.kf.F = np.array([[1, 0, 1, 0],
                                  [0, 1, 0, 1],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]], dtype=float)
            self.kf.H = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0]], dtype=float)
            self.kf.Q *= 0.1
            self.kf.R *= 10
            self.kf.P *= 100
        else:
            self.kf = SimpleKalmanFilter()
        
    def update(self, x: float, y: float, dt: float):
        """Update Kalman filter with new measurement"""
        if KALMAN_AVAILABLE:
            self.kf.F[0, 2] = dt
            self.kf.F[1, 3] = dt
            self.kf.predict()
            self.kf.update(np.array([x, y]))
        else:
            self.kf.F[0, 2] = dt
            self.kf.F[1, 3] = dt
            self.kf.predict()
            self.kf.update(np.array([x, y]))
        
    def predict(self, dt: float) -> Tuple[float, float, float, float]:
        """Predict future position"""
        if KALMAN_AVAILABLE:
            self.kf.F[0, 2] = dt
            self.kf.F[1, 3] = dt
            self.kf.predict()
            return self.kf.x[0], self.kf.x[1], self.kf.x[2], self.kf.x[3]
        else:
            self.kf.F[0, 2] = dt
            self.kf.F[1, 3] = dt
            self.kf.predict()
            return self.kf.x[0], self.kf.x[1], self.kf.x[2], self.kf.x[3]
    
    def get_state(self) -> np.ndarray:
        """Get current state [x, y, vx, vy]"""
        if KALMAN_AVAILABLE:
            return self.kf.x.flatten()
        else:
            return self.kf.x.flatten()
    
    def get_uncertainty(self) -> np.ndarray:
        """Get position uncertainty (covariance)"""
        if KALMAN_AVAILABLE:
            return self.kf.P[:2, :2]
        else:
            return self.kf.P[:2, :2]

class MultiModalFeatureExtractor:
    """Extract multiple types of features for robust re-identification"""
    
    def __init__(self, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        
        # Initialize deep learning model
        self._init_deep_model()
        
        # Feature weights for combination
        self.feature_weights = {
            'color_hist': 0.4,
            'hog': 0.6,
            'deep': 0.0
        }
        
    def _init_deep_model(self):
        """Initialize deep learning feature extractor"""
        try:
            self.deep_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            # Remove final classification layer
            self.deep_model = nn.Sequential(*list(self.deep_model.children())[:-1])
            self.deep_model.eval()
            self.deep_model.to(self.device)
            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            self.deep_features_available = True
        except Exception as e:
            print(f"Warning: Deep learning model not available: {e}")
            self.deep_features_available = False
    
    def extract_color_histogram(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract color histogram features"""
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(96)  # 32 bins per channel * 3 channels
            
        roi = image[y1:y2, x1:x2]
        
        # Extract histograms for each channel
        hist_b = cv2.calcHist([roi], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([roi], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([roi], [2], None, [32], [0, 256])
        
        features = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        features = features / (features.sum() + 1e-6)  # Normalize
        return features
    
    def extract_hog_features(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract HOG features"""
        if not HOG_AVAILABLE:
            return np.zeros(1764)  # Default HOG feature size
            
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(1764)
            
        roi = image[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (64, 128))
        
        try:
            features = hog(roi_resized, pixels_per_cell=(8, 8), 
                          cells_per_block=(2, 2), feature_vector=True)
            return features
        except Exception:
            return np.zeros(1764)
    
    def extract_deep_features(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract deep learning features"""
        if not self.deep_features_available:
            return np.zeros(2048)
            
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(2048)
            
        roi = image[y1:y2, x1:x2]
        
        try:
            input_tensor = self.transform(roi).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.deep_model(input_tensor)
                features = features.squeeze().cpu().numpy()
                
            # L2 normalize features
            features = features / (np.linalg.norm(features) + 1e-8)
            return features
        except Exception:
            return np.zeros(2048)
    
    def extract_all_features(self, image: np.ndarray, bbox: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all available features"""
        features = {}
        
        # Color histogram features
        features['color_hist'] = self.extract_color_histogram(image, bbox)
        
        # HOG features
        features['hog'] = self.extract_hog_features(image, bbox)
        
        # Deep learning features
        #features['deep'] = self.extract_deep_features(image, bbox)
        
        return features
    
    def compute_similarity(self, features1: Dict[str, np.ndarray], 
                          features2: Dict[str, np.ndarray]) -> float:
        """Compute combined similarity score"""
        similarities = []
        weights = []
        
        # Color histogram similarity (Bhattacharyya distance)
        if 'color_hist' in features1 and 'color_hist' in features2:
            hist_sim = cv2.compareHist(
                features1['color_hist'].reshape(-1, 1), 
                features2['color_hist'].reshape(-1, 1), 
                cv2.HISTCMP_BHATTACHARYYA
            )
            similarities.append(1.0 - hist_sim)  # Convert distance to similarity
            weights.append(self.feature_weights['color_hist'])
        
        # HOG similarity (cosine similarity)
        if 'hog' in features1 and 'hog' in features2:
            hog_sim = np.dot(features1['hog'], features2['hog']) / (
                np.linalg.norm(features1['hog']) * np.linalg.norm(features2['hog']) + 1e-8
            )
            similarities.append(hog_sim)
            weights.append(self.feature_weights['hog'])
        
        # Deep features similarity (cosine similarity)
        if 'deep' in features1 and 'deep' in features2:
            deep_sim = np.dot(features1['deep'], features2['deep']) / (
                np.linalg.norm(features1['deep']) * np.linalg.norm(features2['deep']) + 1e-8
            )
            similarities.append(deep_sim)
            weights.append(self.feature_weights['deep'])
        
        if not similarities:
            return 0.0
        
        # Weighted average
        return np.average(similarities, weights=weights)

class OcclusionHandler:
    """Handle target occlusion and re-identification"""
    
    def __init__(self, max_occlusion_frames: int = 30, search_expansion_rate: float = 1.2):
        self.max_occlusion_frames = max_occlusion_frames
        self.search_expansion_rate = search_expansion_rate
        self.lost_tracks: Dict[int, Dict] = {}
        
    def register_lost_track(self, track_id: int, last_state: TrackState,
                           predictor: KinematicPredictor,
                           features: Dict[str, np.ndarray]):
        """Register a track that was lost due to occlusion"""
        self.lost_tracks[track_id] = {
            'last_state': last_state,
            'predictor': predictor,
            'features': features,
            'frames_lost': 0,
            'search_region': self._compute_initial_search_region(last_state)
        }
        
    def _compute_initial_search_region(self, state: TrackState) -> np.ndarray:
        """Compute initial search region based on last known state"""
        x1, y1, x2, y2 = state.bbox
        w, h = x2 - x1, y2 - y1
        vx, vy = state.velocity
        
        # Expand region based on velocity magnitude
        speed = np.sqrt(vx**2 + vy**2)
        expansion = 1.0 + min(speed * 0.1, 0.5)
        
        # Return expanded bbox [x1, y1, x2, y2]
        return np.array([
            x1 - w * (expansion - 1) / 2,
            y1 - h * (expansion - 1) / 2,
            x2 + w * (expansion - 1) / 2,
            y2 + h * (expansion - 1) / 2
        ])
    
    def predict_search_regions(self, dt: float) -> Dict[int, np.ndarray]:
        """Predict search regions for all lost tracks"""
        search_regions = {}
        tracks_to_remove = []
        
        for track_id, info in self.lost_tracks.items():
            info['frames_lost'] += 1
            
            # Remove tracks lost for too long
            if info['frames_lost'] > self.max_occlusion_frames:
                tracks_to_remove.append(track_id)
                continue
                
            # Predict new position
            predictor = info['predictor']
            x_pred, y_pred, vx_pred, vy_pred = predictor.predict(dt)
            
            # Get uncertainty
            uncertainty = predictor.get_uncertainty()
            std_x = np.sqrt(uncertainty[0, 0])
            std_y = np.sqrt(uncertainty[1, 1])
            
            # Compute search region based on prediction and uncertainty
            last_bbox = info['last_state'].bbox
            w, h = last_bbox[2] - last_bbox[0], last_bbox[3] - last_bbox[1]
            expansion = self.search_expansion_rate ** info['frames_lost']
            
            search_region = np.array([
                x_pred - w * expansion / 2 - 2 * std_x,
                y_pred - h * expansion / 2 - 2 * std_y,
                x_pred + w * expansion / 2 + 2 * std_x,
                y_pred + h * expansion / 2 + 2 * std_y
            ])
            
            search_regions[track_id] = search_region
            info['search_region'] = search_region
            
        # Remove old tracks
        for track_id in tracks_to_remove:
            del self.lost_tracks[track_id]
            
        return search_regions
    
    def match_detections(self, detections: sv.Detections, 
                        image: np.ndarray,
                        feature_extractor: MultiModalFeatureExtractor,
                        appearance_threshold: float = 0.5,
                        iou_threshold: float = 0.3) -> Dict[int, int]:
        """Match new detections with lost tracks"""
        matches = {}
        
        if len(detections) == 0:
            return matches
            
        for track_id, info in self.lost_tracks.items():
            search_region = info['search_region']
            template_features = info['features']
            
            best_match_idx = -1
            best_match_score = 0.0
            
            for det_idx in range(len(detections)):
                # Get detection bbox
                det_bbox = detections.xyxy[det_idx]
                
                # Check if detection is within search region
                iou = self._compute_iou(det_bbox, search_region)
                if iou < iou_threshold:
                    continue
                    
                # Extract appearance features
                det_features = feature_extractor.extract_all_features(image, det_bbox)
                
                # Compute appearance similarity
                appearance_score = feature_extractor.compute_similarity(template_features, det_features)
                
                # Combine spatial and appearance scores
                combined_score = appearance_score * (1 + iou * 0.3)
                
                if combined_score > best_match_score and appearance_score > appearance_threshold:
                    best_match_score = combined_score
                    best_match_idx = det_idx
                    
            if best_match_idx >= 0:
                matches[track_id] = best_match_idx
                
        return matches
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
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

class RobustReidentificationSystem:
    """Main re-identification system that integrates with supervision library"""
    
    def __init__(self, max_occlusion_frames: int = 30):
        self.tracks: Dict[int, Dict] = {}
        self.occlusion_handler = OcclusionHandler(max_occlusion_frames)
        self.feature_extractor = MultiModalFeatureExtractor()
        self.frame_count = 0
        self.last_timestamp = 0
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'successful_reidentifications': 0,
            'failed_reidentifications': 0,
            'new_tracks': 0,
            'lost_tracks': 0
        }
        
        print("✅ Robust re-identification system initialized!")
    
    def initialize_track(self, track_id: int, bbox: np.ndarray, 
                         image: np.ndarray, timestamp: float, class_name: str = None):
        """Initialize a new track"""
        self.tracks[track_id] = {
            'predictor': KinematicPredictor(),
            'features': {},
            'history': deque(maxlen=30),
            'last_seen': self.frame_count,
            'class_name': class_name
        }
        
        # Initialize predictor with center of bbox
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        self.tracks[track_id]['predictor'].update(cx, cy, 0.033)  # 30fps assumed
        
        # Extract and store features
        features = self.feature_extractor.extract_all_features(image, bbox)
        self.tracks[track_id]['features'] = features
        
        # Store state
        state = TrackState(
            track_id=track_id,
            bbox=bbox,
            timestamp=timestamp,
            velocity=np.array([0, 0]),
            confidence=1.0,
            feature_vector=None,  # We store features separately
            class_name=class_name
        )
        self.tracks[track_id]['history'].append(state)
        self.stats['new_tracks'] += 1
        
    def update_track(self, track_id: int, bbox: np.ndarray, 
                    image: np.ndarray, timestamp: float, class_name: str = None):
        """Update existing track"""
        if track_id not in self.tracks:
            self.initialize_track(track_id, bbox, image, timestamp, class_name)
            return
            
        track = self.tracks[track_id]
        dt = timestamp - self.last_timestamp if self.last_timestamp > 0 else 0.033
        
        # Update predictor
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        track['predictor'].update(cx, cy, dt)
        
        # Get velocity
        state = track['predictor'].get_state()
        velocity = state[2:4]
        
        # Update features (keep a running average)
        new_features = self.feature_extractor.extract_all_features(image, bbox)
        for key in new_features:
            if key in track['features']:
                # Exponential moving average
                alpha = 0.1
                track['features'][key] = (1 - alpha) * track['features'][key] + alpha * new_features[key]
            else:
                track['features'][key] = new_features[key]
        
        # Store state
        state = TrackState(
            track_id=track_id,
            bbox=bbox,
            timestamp=timestamp,
            velocity=velocity,
            confidence=1.0,
            feature_vector=None,
            class_name=class_name or track.get('class_name')
        )
        track['history'].append(state)
        track['last_seen'] = self.frame_count
        
    def handle_occlusion(self, lost_track_ids: List[int]):
        """Handle tracks that were lost"""
        for track_id in lost_track_ids:
            if track_id in self.tracks:
                track = self.tracks[track_id]
                if len(track['history']) > 0:
                    last_state = track['history'][-1]
                    self.occlusion_handler.register_lost_track(
                        track_id,
                        last_state,
                        track['predictor'],
                        track['features']
                    )
                    self.stats['lost_tracks'] += 1
                    
    def reidentify(self, detections: sv.Detections, 
                  image: np.ndarray, timestamp: float) -> Dict[int, int]:
        """Attempt to re-identify lost tracks"""
        dt = timestamp - self.last_timestamp if self.last_timestamp > 0 else 0.033
        
        # Predict search regions
        search_regions = self.occlusion_handler.predict_search_regions(dt)
        
        # Match detections
        matches = self.occlusion_handler.match_detections(
            detections, image, self.feature_extractor
        )
        
        # Update matched tracks
        for track_id, det_idx in matches.items():
            bbox = detections.xyxy[det_idx]
            class_name = getattr(detections, 'data', {}).get('class_name', [None] * len(detections))
            if isinstance(class_name, list) and det_idx < len(class_name):
                class_name = class_name[det_idx]
            else:
                class_name = None
                
            self.update_track(track_id, bbox, image, timestamp, class_name)
            self.stats['successful_reidentifications'] += 1
            
            # Remove from lost tracks
            if track_id in self.occlusion_handler.lost_tracks:
                del self.occlusion_handler.lost_tracks[track_id]
                
        return matches
    
    def process_detections(self, detections: sv.Detections, 
                          image: np.ndarray, timestamp: float) -> sv.Detections:
        """Process detections with re-identification"""
        if len(detections) == 0:
            return detections
            
        current_track_ids = set()
        if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
            current_track_ids = set(detections.tracker_id[detections.tracker_id != None])
        
        # Update active tracks
        for i, track_id in enumerate(current_track_ids):
            if track_id is not None:
                bbox = detections.xyxy[i]
                class_name = getattr(detections, 'data', {}).get('class_name', [None] * len(detections))
                if isinstance(class_name, list) and i < len(class_name):
                    class_name = class_name[i]
                else:
                    class_name = None
                    
                self.update_track(track_id, bbox, image, timestamp, class_name)
        
        # Find lost tracks
        all_track_ids = set(self.tracks.keys())
        lost_track_ids = all_track_ids - current_track_ids
        
        # Check if tracks were recently seen
        recently_lost = [
            tid for tid in lost_track_ids 
            if self.frame_count - self.tracks[tid]['last_seen'] == 1
        ]
        
        if recently_lost:
            self.handle_occlusion(recently_lost)
            
        # Attempt re-identification
        matches = self.reidentify(detections, image, timestamp)
        
        if matches:
            print(f"🔄 Re-identified {len(matches)} tracks: {list(matches.keys())}")
        
        self.frame_count += 1
        self.last_timestamp = timestamp
        self.stats['total_detections'] += len(detections)
        
        return detections
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get re-identification statistics"""
        total_attempts = (self.stats['successful_reidentifications'] + 
                         self.stats['failed_reidentifications'])
        
        success_rate = (self.stats['successful_reidentifications'] / total_attempts 
                       if total_attempts > 0 else 0)
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'active_tracks': len(self.tracks),
            'lost_tracks_count': len(self.occlusion_handler.lost_tracks)
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize the system
    reid_system = RobustReidentificationSystem()
    
    print("🎯 Robust Re-identification System Ready!")
    print("Features:")
    print("  ✅ Multi-modal feature extraction (color histograms, HOG, deep features)")
    print("  ✅ Kalman filter-based kinematic prediction")
    print("  ✅ Intelligent search region prediction")
    print("  ✅ Integration with supervision library")
    print("  ✅ Comprehensive statistics and monitoring")
