#!/usr/bin/env python3
"""
Kinematic-Enhanced Re-identification System

This module extends the existing re-identification system with advanced kinematic
prediction capabilities. It combines appearance-based features with position,
velocity, and acceleration predictions to improve object re-identification
after occlusion.

Key Features:
- Continuous position projection for lost tracks
- Kinematic-aware matching with uncertainty estimation
- Integration with existing feature-based re-identification
- Multi-modal scoring combining appearance and motion
- Adaptive thresholds based on prediction confidence

Author: EDTH Hackathon 2025
"""

import numpy as np
import math
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import supervision as sv

# Import the kinematic predictor
from kinematic_predictor import KinematicPredictor, PredictionResult
from position_3d_calculator import Position3DCalculator, CameraParameters, DronePose, Object3DPosition

@dataclass
class KinematicMatch:
    """Represents a kinematic-based match between a lost track and detection"""
    track_id: int
    detection_idx: int
    kinematic_score: float
    appearance_score: float
    combined_score: float
    predicted_position: Tuple[float, float]
    actual_position: Tuple[float, float]
    position_error: float
    confidence: float

class KinematicReidentificationSystem:
    """Enhanced re-identification system with kinematic prediction"""
    
    def __init__(self, max_occlusion_frames: int = 60, 
                 kinematic_weight: float = 0.6,
                 appearance_weight: float = 0.4,
                 camera_params: Optional[CameraParameters] = None):
        """
        Initialize the kinematic re-identification system
        
        Args:
            max_occlusion_frames: Maximum frames to keep a lost track
            kinematic_weight: Weight for kinematic scoring (0-1)
            appearance_weight: Weight for appearance scoring (0-1)
            camera_params: Camera parameters for 3D positioning
        """
        self.max_occlusion_frames = max_occlusion_frames
        self.kinematic_weight = kinematic_weight
        self.appearance_weight = appearance_weight
        
        # Initialize kinematic predictor
        self.kinematic_predictor = KinematicPredictor()
        
        # Initialize 3D position calculator
        if camera_params is None:
            # Default camera parameters
            camera_params = CameraParameters(width=1280, height=720, hfov=90.0, vfov=60.0)
        self.position_calculator = Position3DCalculator(camera_params)
        
        # Lost tracks with kinematic information
        self.lost_tracks: Dict[int, Dict[str, Any]] = {}
        
        # Track statistics for adaptive thresholds
        self.track_statistics: Dict[int, Dict[str, Any]] = {}
        
        # Matching parameters
        self.matching_params = {
            'max_position_error': 50.0,  # Maximum allowed position error (pixels)
            'min_kinematic_score': 0.3,  # Minimum kinematic score for consideration
            'min_appearance_score': 0.4,  # Minimum appearance score for consideration
            'min_combined_score': 0.5,   # Minimum combined score for match
            'uncertainty_multiplier': 2.0,  # Multiplier for uncertainty-based thresholds
            'confidence_threshold': 0.3    # Minimum prediction confidence
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_matches_attempted': 0,
            'kinematic_matches': 0,
            'appearance_matches': 0,
            'combined_matches': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        print("✅ Kinematic re-identification system initialized")
    
    def update_track(self, track_id: int, position: Tuple[float, float], 
                    bbox: np.ndarray, timestamp: float, 
                    appearance_features: Optional[np.ndarray] = None,
                    class_name: Optional[str] = None,
                    drone_pose: Optional[DronePose] = None) -> None:
        """
        Update a track with new observations
        
        Args:
            track_id: Track identifier
            position: Current position (x, y)
            bbox: Bounding box
            timestamp: Current timestamp
            appearance_features: Optional appearance features
            class_name: Object class name
            drone_pose: Current drone pose for 3D positioning
        """
        # Calculate 3D position if drone pose is available
        position_3d = None
        if drone_pose is not None and class_name is not None:
            try:
                position_3d = self.position_calculator.pixel_to_3d(
                    tuple(bbox), class_name, drone_pose
                )
            except Exception as e:
                print(f"Warning: 3D positioning failed for track {track_id}: {e}")
        
        # Update kinematic predictor with 3D position if available
        if position_3d is not None:
            # Use 3D world coordinates for kinematic prediction
            world_position = (position_3d.latitude, position_3d.longitude)
            kinematic_state = self.kinematic_predictor.update_track(
                track_id, world_position, timestamp, bbox
            )
        else:
            # Fall back to 2D pixel coordinates
            kinematic_state = self.kinematic_predictor.update_track(
                track_id, position, timestamp, bbox
            )
        
        # Update track statistics
        if track_id not in self.track_statistics:
            self.track_statistics[track_id] = {
                'class_name': class_name or 'unknown',
                'first_seen': timestamp,
                'last_seen': timestamp,
                'total_observations': 0,
                'reidentification_count': 0,
                'motion_pattern': 'unknown'
            }
        
        stats = self.track_statistics[track_id]
        stats['last_seen'] = timestamp
        stats['total_observations'] += 1
        
        # Update motion pattern classification
        self._update_motion_pattern(track_id, kinematic_state)
        
        # Remove from lost tracks if it was lost
        if track_id in self.lost_tracks:
            del self.lost_tracks[track_id]
            stats['reidentification_count'] += 1
    
    def lose_track(self, track_id: int, last_position: Tuple[float, float],
                  last_bbox: np.ndarray, last_timestamp: float,
                  appearance_features: Optional[np.ndarray] = None,
                  class_name: Optional[str] = None,
                  drone_pose: Optional[DronePose] = None) -> None:
        """
        Mark a track as lost and store its last known state
        
        Args:
            track_id: Track identifier
            last_position: Last known position
            last_bbox: Last known bounding box
            last_timestamp: Last observation timestamp
            appearance_features: Last appearance features
            class_name: Object class name
        """
        # Store lost track information
        self.lost_tracks[track_id] = {
            'last_position': last_position,
            'last_bbox': last_bbox,
            'last_timestamp': last_timestamp,
            'appearance_features': appearance_features,
            'class_name': class_name or 'unknown',
            'frames_lost': 0,
            'search_region': self._calculate_search_region(last_position, last_bbox),
            'prediction_history': deque(maxlen=10)
        }
        
        # Update track statistics
        if track_id in self.track_statistics:
            self.track_statistics[track_id]['last_lost'] = last_timestamp
    
    def process_detections(self, detections: sv.Detections, frame: np.ndarray, 
                          timestamp: float, feature_extractor=None) -> sv.Detections:
        """
        Process new detections and attempt re-identification with lost tracks
        
        Args:
            detections: New detections from tracker
            frame: Current frame
            timestamp: Current timestamp
            feature_extractor: Feature extractor for appearance matching
            
        Returns:
            Updated detections with re-identified tracks
        """
        if len(detections) == 0:
            return detections
        
        # Update lost track counters
        self._update_lost_track_counters(timestamp)
        
        # Get kinematic predictions for lost tracks
        lost_track_ids = list(self.lost_tracks.keys())
        predictions = self.kinematic_predictor.get_lost_track_predictions(
            lost_track_ids, timestamp
        )
        
        # Find kinematic matches
        kinematic_matches = self._find_kinematic_matches(
            detections, frame, predictions, feature_extractor
        )
        
        # Apply matches to detections
        updated_detections = self._apply_matches(detections, kinematic_matches)
        
        # Update performance statistics
        self._update_performance_stats(kinematic_matches)
        
        return updated_detections
    
    def _find_kinematic_matches(self, detections: sv.Detections, frame: np.ndarray,
                               predictions: Dict[int, PredictionResult],
                               feature_extractor=None) -> List[KinematicMatch]:
        """Find matches between lost tracks and new detections using kinematic prediction"""
        matches = []
        
        if len(detections) == 0 or not predictions:
            return matches
        
        for track_id, prediction in predictions.items():
            if track_id not in self.lost_tracks:
                continue
            
            lost_track_info = self.lost_tracks[track_id]
            
            # Skip if track has been lost too long
            if lost_track_info['frames_lost'] > self.max_occlusion_frames:
                continue
            
            # Skip if prediction confidence is too low
            if prediction.confidence < self.matching_params['confidence_threshold']:
                continue
            
            # Find best matching detection
            best_match = self._find_best_detection_match(
                track_id, detections, frame, prediction, 
                lost_track_info, feature_extractor
            )
            
            if best_match:
                matches.append(best_match)
        
        # Sort matches by combined score and remove duplicates
        matches.sort(key=lambda m: m.combined_score, reverse=True)
        return self._remove_duplicate_matches(matches)
    
    def _find_best_detection_match(self, track_id: int, detections: sv.Detections,
                                 frame: np.ndarray, prediction: PredictionResult,
                                 lost_track_info: Dict[str, Any],
                                 feature_extractor=None) -> Optional[KinematicMatch]:
        """Find the best matching detection for a specific lost track"""
        best_match = None
        best_score = 0.0
        
        predicted_pos = prediction.predicted_position
        uncertainty = prediction.uncertainty
        
        # Calculate adaptive thresholds based on uncertainty and confidence
        max_error = self.matching_params['max_position_error'] * (
            1.0 + uncertainty / 20.0  # Increase threshold with uncertainty
        )
        
        for det_idx in range(len(detections)):
            det_bbox = detections.xyxy[det_idx]
            det_center = (
                (det_bbox[0] + det_bbox[2]) / 2,
                (det_bbox[1] + det_bbox[3]) / 2
            )
            
            # Calculate kinematic score
            kinematic_score = self._calculate_kinematic_score(
                predicted_pos, det_center, uncertainty, max_error
            )
            
            if kinematic_score < self.matching_params['min_kinematic_score']:
                continue
            
            # Calculate appearance score if feature extractor is available
            appearance_score = 0.0
            if (feature_extractor is not None and 
                lost_track_info['appearance_features'] is not None):
                try:
                    det_features = feature_extractor.extract_all_features(frame, det_bbox)
                    appearance_score = feature_extractor.compute_similarity(
                        lost_track_info['appearance_features'], det_features
                    )
                except Exception as e:
                    print(f"Warning: Feature extraction failed: {e}")
                    appearance_score = 0.0
            
            # Skip if appearance score is too low
            if appearance_score < self.matching_params['min_appearance_score']:
                continue
            
            # Calculate combined score
            combined_score = (
                self.kinematic_weight * kinematic_score +
                self.appearance_weight * appearance_score
            )
            
            if combined_score < self.matching_params['min_combined_score']:
                continue
            
            # Calculate position error
            position_error = math.sqrt(
                (predicted_pos[0] - det_center[0])**2 +
                (predicted_pos[1] - det_center[1])**2
            )
            
            # Create match object
            match = KinematicMatch(
                track_id=track_id,
                detection_idx=det_idx,
                kinematic_score=kinematic_score,
                appearance_score=appearance_score,
                combined_score=combined_score,
                predicted_position=predicted_pos,
                actual_position=det_center,
                position_error=position_error,
                confidence=prediction.confidence
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = match
        
        return best_match
    
    def _calculate_kinematic_score(self, predicted_pos: Tuple[float, float],
                                 actual_pos: Tuple[float, float],
                                 uncertainty: float, max_error: float) -> float:
        """Calculate kinematic-based matching score"""
        # Calculate position error
        position_error = math.sqrt(
            (predicted_pos[0] - actual_pos[0])**2 +
            (predicted_pos[1] - actual_pos[1])**2
        )
        
        # Normalize error by uncertainty and max allowed error
        normalized_error = position_error / (uncertainty + 1e-6)
        max_normalized_error = max_error / (uncertainty + 1e-6)
        
        # Calculate score (higher is better)
        if normalized_error > max_normalized_error:
            return 0.0
        
        # Use exponential decay for score
        score = math.exp(-normalized_error / 2.0)
        
        # Apply uncertainty penalty
        uncertainty_penalty = 1.0 / (1.0 + uncertainty / 10.0)
        
        return score * uncertainty_penalty
    
    def _apply_matches(self, detections: sv.Detections, 
                      matches: List[KinematicMatch]) -> sv.Detections:
        """Apply kinematic matches to detections"""
        if not matches:
            return detections
        
        # Create new tracker IDs array
        new_tracker_ids = detections.tracker_id.copy() if detections.tracker_id is not None else np.full(len(detections), None)
        
        for match in matches:
            if match.detection_idx < len(new_tracker_ids):
                new_tracker_ids[match.detection_idx] = match.track_id
                
                # Update lost track info
                if match.track_id in self.lost_tracks:
                    self.lost_tracks[match.track_id]['frames_lost'] = 0
                    self.lost_tracks[match.track_id]['prediction_history'].append({
                        'timestamp': time.time(),
                        'predicted_pos': match.predicted_position,
                        'actual_pos': match.actual_position,
                        'error': match.position_error,
                        'score': match.combined_score
                    })
        
        # Create new detections with updated tracker IDs
        updated_detections = sv.Detections(
            xyxy=detections.xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id,
            tracker_id=new_tracker_ids
        )
        
        return updated_detections
    
    def _remove_duplicate_matches(self, matches: List[KinematicMatch]) -> List[KinematicMatch]:
        """Remove duplicate matches (same detection matched to multiple tracks)"""
        if not matches:
            return matches
        
        # Sort by combined score (highest first)
        matches.sort(key=lambda m: m.combined_score, reverse=True)
        
        used_detections = set()
        unique_matches = []
        
        for match in matches:
            if match.detection_idx not in used_detections:
                unique_matches.append(match)
                used_detections.add(match.detection_idx)
        
        return unique_matches
    
    def _calculate_search_region(self, position: Tuple[float, float], 
                               bbox: np.ndarray) -> np.ndarray:
        """Calculate search region around last known position"""
        # Use bounding box size to determine search region
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        
        # Expand search region based on expected motion
        expansion_factor = 2.0
        search_width = bbox_width * expansion_factor
        search_height = bbox_height * expansion_factor
        
        # Create search region centered on last position
        x, y = position
        search_region = np.array([
            x - search_width / 2,
            y - search_height / 2,
            x + search_width / 2,
            y + search_height / 2
        ])
        
        return search_region
    
    def _update_lost_track_counters(self, current_timestamp: float) -> None:
        """Update frame counters for lost tracks"""
        tracks_to_remove = []
        
        for track_id, info in self.lost_tracks.items():
            info['frames_lost'] += 1
            
            # Remove tracks that have been lost too long
            if info['frames_lost'] > self.max_occlusion_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.lost_tracks[track_id]
    
    def _update_motion_pattern(self, track_id: int, kinematic_state) -> None:
        """Update motion pattern classification for a track"""
        if track_id not in self.track_statistics:
            return
        
        stats = self.track_statistics[track_id]
        speed = math.sqrt(kinematic_state.velocity[0]**2 + kinematic_state.velocity[1]**2)
        accel = math.sqrt(kinematic_state.acceleration[0]**2 + kinematic_state.acceleration[1]**2)
        
        # Classify motion pattern
        if speed < 1.0:
            pattern = 'stationary'
        elif speed < 5.0:
            pattern = 'slow'
        elif speed < 15.0:
            pattern = 'medium'
        else:
            pattern = 'fast'
        
        # Add acceleration classification
        if accel > 2.0:
            pattern += '_accelerating'
        elif accel < 0.5:
            pattern += '_steady'
        
        stats['motion_pattern'] = pattern
    
    def _update_performance_stats(self, matches: List[KinematicMatch]) -> None:
        """Update performance statistics"""
        self.performance_stats['total_matches_attempted'] += len(matches)
        
        for match in matches:
            if match.kinematic_score > match.appearance_score:
                self.performance_stats['kinematic_matches'] += 1
            else:
                self.performance_stats['appearance_matches'] += 1
            
            self.performance_stats['combined_matches'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'lost_tracks_count': len(self.lost_tracks),
            'active_tracks_count': len(self.track_statistics),
            'performance_stats': self.performance_stats.copy(),
            'matching_params': self.matching_params.copy()
        }
    
    def get_lost_track_info(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific lost track"""
        if track_id not in self.lost_tracks:
            return None
        
        info = self.lost_tracks[track_id].copy()
        
        # Add prediction if available
        predictions = self.kinematic_predictor.get_lost_track_predictions(
            [track_id], time.time()
        )
        
        if track_id in predictions:
            info['current_prediction'] = {
                'position': predictions[track_id].predicted_position,
                'uncertainty': predictions[track_id].uncertainty,
                'confidence': predictions[track_id].confidence
            }
        
        return info
    
    def cleanup_old_tracks(self, active_track_ids: List[int]) -> None:
        """Clean up old track data"""
        # Clean up kinematic predictor
        self.kinematic_predictor.cleanup_old_tracks(active_track_ids)
        
        # Clean up lost tracks
        tracks_to_remove = []
        for track_id in self.lost_tracks:
            if track_id not in active_track_ids:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.lost_tracks[track_id]
        
        # Clean up track statistics
        stats_to_remove = []
        for track_id in self.track_statistics:
            if track_id not in active_track_ids:
                stats_to_remove.append(track_id)
        
        for track_id in stats_to_remove:
            del self.track_statistics[track_id]


if __name__ == "__main__":
    # Test the kinematic re-identification system
    system = KinematicReidentificationSystem()
    
    # Simulate a track
    track_id = 1
    positions = [(100, 100), (105, 102), (110, 104), (115, 106)]
    
    for i, pos in enumerate(positions):
        bbox = np.array([pos[0]-10, pos[1]-10, pos[0]+10, pos[1]+10])
        timestamp = i * 0.033
        system.update_track(track_id, pos, bbox, timestamp)
    
    # Simulate losing the track
    last_pos = positions[-1]
    last_bbox = np.array([last_pos[0]-10, last_pos[1]-10, last_pos[0]+10, last_pos[1]+10])
    system.lose_track(track_id, last_pos, last_bbox, 0.2)
    
    # Simulate new detection
    detections = sv.Detections(
        xyxy=np.array([[120, 108, 140, 128]]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        tracker_id=np.array([None])
    )
    
    # Process detections
    updated_detections = system.process_detections(detections, None, 0.3)
    
    print(f"Original tracker IDs: {detections.tracker_id}")
    print(f"Updated tracker IDs: {updated_detections.tracker_id}")
    print(f"System statistics: {system.get_statistics()}")
