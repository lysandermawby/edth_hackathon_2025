#!/usr/bin/env python3
"""
Enhanced Kinematic Prediction System for Object Tracking

This module provides advanced kinematic prediction capabilities for object tracking,
including position, velocity, and acceleration prediction with uncertainty estimation.
It integrates with the re-identification system to improve object matching after occlusion.

Key Features:
- Continuous position projection for lost tracks
- Velocity and acceleration estimation with smoothing
- Uncertainty estimation for prediction confidence
- Multiple prediction models (constant velocity, constant acceleration, adaptive)
- Integration with re-identification matching

Author: EDTH Hackathon 2025
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import time
import cv2
import os

@dataclass
class KinematicState:
    """Represents the kinematic state of a tracked object"""
    position: Tuple[float, float]  # (x, y) in image coordinates
    velocity: Tuple[float, float]  # (vx, vy) in pixels/frame
    acceleration: Tuple[float, float]  # (ax, ay) in pixels/frame²
    timestamp: float
    uncertainty: float  # Position uncertainty (standard deviation)
    confidence: float  # Overall prediction confidence (0-1)

@dataclass
class PredictionResult:
    """Result of kinematic prediction"""
    predicted_position: Tuple[float, float]
    predicted_velocity: Tuple[float, float]
    predicted_acceleration: Tuple[float, float]
    uncertainty: float
    confidence: float
    prediction_time: float
    frames_ahead: int

class KinematicPredictor:
    """Advanced kinematic predictor with multiple prediction models"""
    
    def __init__(self, max_history: int = 20, smoothing_factor: float = 0.7, 
                 video_path: Optional[str] = None, fps: Optional[float] = None):
        """
        Initialize the kinematic predictor
        
        Args:
            max_history: Maximum number of historical states to keep
            smoothing_factor: Smoothing factor for velocity/acceleration estimation
            video_path: Optional path to video file for FPS extraction
            fps: Optional explicit FPS value (overrides video_path if provided)
        """
        self.max_history = max_history
        self.smoothing_factor = smoothing_factor
        
        # Set FPS from video or explicit value
        if fps is not None:
            self.fps = fps
        elif video_path and os.path.exists(video_path):
            self.fps = self._extract_fps_from_video(video_path)
        else:
            self.fps = 30.0  # Default fallback
            print(f"⚠️  Using default FPS: {self.fps} (no video path provided or file not found)")
        
        # Track history for each object
        self.track_histories: Dict[int, deque] = {}
        
        # Prediction models
        self.prediction_models = {
            'constant_velocity': self._predict_constant_velocity,
            'constant_acceleration': self._predict_constant_acceleration,
            'adaptive': self._predict_adaptive
        }
        
        # Model selection parameters
        self.model_weights = {
            'constant_velocity': 0.4,
            'constant_acceleration': 0.4,
            'adaptive': 0.2
        }
        
        # Acceleration estimation parameters
        self.acceleration_smoothing = 0.6
        self.min_velocity_for_acceleration = 2.0  # pixels/frame
        
        print(f"✅ Enhanced kinematic predictor initialized with FPS: {self.fps}")
    
    def _extract_fps_from_video(self, video_path: str) -> float:
        """
        Extract FPS from video file using OpenCV
        
        Args:
            video_path: Path to video file
            
        Returns:
            FPS value as float
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"⚠️  Could not open video file: {video_path}")
                return 30.0
            
            # Get FPS from video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if fps > 0:
                print(f"📹 Extracted FPS from video: {fps:.2f}")
                return float(fps)
            else:
                print(f"⚠️  Could not extract FPS from video: {video_path}, using default 30.0")
                return 30.0
                
        except Exception as e:
            print(f"⚠️  Error extracting FPS from video {video_path}: {e}, using default 30.0")
            return 30.0
    
    def set_fps(self, fps: float):
        """
        Update the FPS value after initialization
        
        Args:
            fps: New FPS value
        """
        if fps > 0:
            self.fps = fps
            print(f"📹 FPS updated to: {self.fps}")
        else:
            print(f"⚠️  Invalid FPS value: {fps}, keeping current FPS: {self.fps}")
    
    def update_track(self, track_id: int, position: Tuple[float, float], 
                    timestamp: float, bbox: Optional[np.ndarray] = None) -> KinematicState:
        """
        Update the kinematic state of a track
        
        Args:
            track_id: Unique identifier for the track
            position: Current position (x, y)
            timestamp: Current timestamp
            bbox: Optional bounding box for size-based uncertainty
            
        Returns:
            Updated kinematic state
        """
        if track_id not in self.track_histories:
            self.track_histories[track_id] = deque(maxlen=self.max_history)
        
        # Calculate velocity and acceleration
        velocity = self._calculate_velocity(track_id, position, timestamp)
        acceleration = self._calculate_acceleration(track_id, velocity, timestamp)
        
        # Calculate uncertainty based on recent motion stability
        uncertainty = self._calculate_uncertainty(track_id, position, velocity)
        
        # Calculate confidence based on prediction accuracy
        confidence = self._calculate_confidence(track_id, position, velocity)
        
        # Create new kinematic state
        state = KinematicState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            timestamp=timestamp,
            uncertainty=uncertainty,
            confidence=confidence
        )
        
        # Store in history
        self.track_histories[track_id].append(state)
        
        return state
    
    def predict_position(self, track_id: int, frames_ahead: int, 
                        current_timestamp: float) -> PredictionResult:
        """
        Predict the position of a track after a given number of frames
        
        Args:
            track_id: Track identifier
            frames_ahead: Number of frames to predict ahead
            current_timestamp: Current timestamp
            
        Returns:
            Prediction result with position, velocity, and uncertainty
        """
        if track_id not in self.track_histories or len(self.track_histories[track_id]) < 2:
            # Not enough history for prediction
            return PredictionResult(
                predicted_position=(0, 0),
                predicted_velocity=(0, 0),
                predicted_acceleration=(0, 0),
                uncertainty=float('inf'),
                confidence=0.0,
                prediction_time=current_timestamp,
                frames_ahead=frames_ahead
            )
        
        history = self.track_histories[track_id]
        last_state = history[-1]
        
        # Calculate time step using actual FPS
        dt = 1.0 / self.fps
        
        # Get predictions from all models
        predictions = {}
        for model_name, predictor_func in self.prediction_models.items():
            try:
                pred = predictor_func(history, frames_ahead, dt)
                predictions[model_name] = pred
            except Exception as e:
                print(f"Warning: Prediction model {model_name} failed: {e}")
                continue
        
        # Combine predictions using weighted average
        combined_pred = self._combine_predictions(predictions)
        
        # Calculate uncertainty growth over time
        uncertainty_growth = self._calculate_uncertainty_growth(
            last_state.uncertainty, frames_ahead, last_state.velocity
        )
        
        # Calculate confidence based on prediction stability
        confidence = self._calculate_prediction_confidence(
            history, combined_pred, frames_ahead
        )
        
        return PredictionResult(
            predicted_position=combined_pred['position'],
            predicted_velocity=combined_pred['velocity'],
            predicted_acceleration=combined_pred['acceleration'],
            uncertainty=uncertainty_growth,
            confidence=confidence,
            prediction_time=current_timestamp + frames_ahead * dt,
            frames_ahead=frames_ahead
        )
    
    def get_lost_track_predictions(self, lost_track_ids: List[int], 
                                 current_timestamp: float) -> Dict[int, PredictionResult]:
        """
        Get position predictions for all lost tracks
        
        Args:
            lost_track_ids: List of track IDs that are currently lost
            current_timestamp: Current timestamp
            
        Returns:
            Dictionary mapping track IDs to their prediction results
        """
        predictions = {}
        
        for track_id in lost_track_ids:
            if track_id in self.track_histories and len(self.track_histories[track_id]) > 0:
                # Calculate frames lost (approximate)
                last_state = self.track_histories[track_id][-1]
                frames_lost = int((current_timestamp - last_state.timestamp) * self.fps)
                
                if frames_lost > 0:
                    pred = self.predict_position(track_id, frames_lost, current_timestamp)
                    predictions[track_id] = pred
        
        return predictions
    
    def _calculate_velocity(self, track_id: int, position: Tuple[float, float], 
                          timestamp: float) -> Tuple[float, float]:
        """Calculate smoothed velocity for a track"""
        history = self.track_histories[track_id]
        
        if len(history) == 0:
            return (0.0, 0.0)
        
        # Get previous state
        prev_state = history[-1]
        dt = timestamp - prev_state.timestamp
        
        if dt <= 0:
            return prev_state.velocity
        
        # Calculate instantaneous velocity
        dx = position[0] - prev_state.position[0]
        dy = position[1] - prev_state.position[1]
        vx_inst = dx / dt
        vy_inst = dy / dt
        
        # Apply smoothing
        vx_smooth = self.smoothing_factor * vx_inst + (1 - self.smoothing_factor) * prev_state.velocity[0]
        vy_smooth = self.smoothing_factor * vy_inst + (1 - self.smoothing_factor) * prev_state.velocity[1]
        
        return (vx_smooth, vy_smooth)
    
    def _calculate_acceleration(self, track_id: int, velocity: Tuple[float, float], 
                              timestamp: float) -> Tuple[float, float]:
        """Calculate smoothed acceleration for a track"""
        history = self.track_histories[track_id]
        
        if len(history) < 2:
            return (0.0, 0.0)
        
        # Get previous velocity
        prev_state = history[-1]
        dt = timestamp - prev_state.timestamp
        
        if dt <= 0:
            return prev_state.acceleration
        
        # Calculate instantaneous acceleration
        dvx = velocity[0] - prev_state.velocity[0]
        dvy = velocity[1] - prev_state.velocity[1]
        ax_inst = dvx / dt
        ay_inst = dvy / dt
        
        # Apply smoothing
        ax_smooth = self.acceleration_smoothing * ax_inst + (1 - self.acceleration_smoothing) * prev_state.acceleration[0]
        ay_smooth = self.acceleration_smoothing * ay_inst + (1 - self.acceleration_smoothing) * prev_state.acceleration[1]
        
        return (ax_smooth, ay_smooth)
    
    def _calculate_uncertainty(self, track_id: int, position: Tuple[float, float], 
                             velocity: Tuple[float, float]) -> float:
        """Calculate position uncertainty based on motion stability"""
        history = self.track_histories[track_id]
        
        if len(history) < 3:
            return 10.0  # Default uncertainty
        
        # Calculate velocity variance over recent history
        recent_velocities = [state.velocity for state in list(history)[-5:]]
        vx_values = [v[0] for v in recent_velocities]
        vy_values = [v[1] for v in recent_velocities]
        
        vx_var = np.var(vx_values) if len(vx_values) > 1 else 0
        vy_var = np.var(vy_values) if len(vy_values) > 1 else 0
        
        # Base uncertainty on velocity stability
        velocity_instability = math.sqrt(vx_var + vy_var)
        base_uncertainty = 5.0 + velocity_instability * 2.0
        
        # Increase uncertainty for high-speed objects
        speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
        speed_factor = 1.0 + min(speed / 50.0, 2.0)  # Cap at 3x
        
        return base_uncertainty * speed_factor
    
    def _calculate_confidence(self, track_id: int, position: Tuple[float, float], 
                            velocity: Tuple[float, float]) -> float:
        """Calculate prediction confidence based on recent accuracy"""
        history = self.track_histories[track_id]
        
        if len(history) < 3:
            return 0.5  # Default confidence
        
        # Calculate prediction accuracy over recent history
        accuracy_scores = []
        for i in range(2, len(history)):
            # Predict position from i-1 to i
            pred = self._predict_constant_velocity(list(history)[:i], 1, 1.0/self.fps)
            actual = history[i].position
            
            # Calculate prediction error
            error = math.sqrt((pred['position'][0] - actual[0])**2 + 
                            (pred['position'][1] - actual[1])**2)
            
            # Convert error to accuracy score (0-1)
            max_expected_error = 20.0  # pixels
            accuracy = max(0, 1.0 - error / max_expected_error)
            accuracy_scores.append(accuracy)
        
        # Return average accuracy
        return np.mean(accuracy_scores) if accuracy_scores else 0.5
    
    def _predict_constant_velocity(self, history: List[KinematicState], 
                                 frames_ahead: int, dt: float) -> Dict[str, Any]:
        """Predict using constant velocity model"""
        if len(history) < 1:
            return {'position': (0, 0), 'velocity': (0, 0), 'acceleration': (0, 0)}
        
        last_state = history[-1]
        vx, vy = last_state.velocity
        
        # Predict position
        pred_x = last_state.position[0] + vx * frames_ahead * dt
        pred_y = last_state.position[1] + vy * frames_ahead * dt
        
        return {
            'position': (pred_x, pred_y),
            'velocity': (vx, vy),
            'acceleration': (0, 0)
        }
    
    def _predict_constant_acceleration(self, history: List[KinematicState], 
                                     frames_ahead: int, dt: float) -> Dict[str, Any]:
        """Predict using constant acceleration model"""
        if len(history) < 1:
            return {'position': (0, 0), 'velocity': (0, 0), 'acceleration': (0, 0)}
        
        last_state = history[-1]
        vx, vy = last_state.velocity
        ax, ay = last_state.acceleration
        
        # Predict position using kinematic equations
        pred_x = (last_state.position[0] + 
                 vx * frames_ahead * dt + 
                 0.5 * ax * (frames_ahead * dt)**2)
        pred_y = (last_state.position[1] + 
                 vy * frames_ahead * dt + 
                 0.5 * ay * (frames_ahead * dt)**2)
        
        # Predict velocity
        pred_vx = vx + ax * frames_ahead * dt
        pred_vy = vy + ay * frames_ahead * dt
        
        return {
            'position': (pred_x, pred_y),
            'velocity': (pred_vx, pred_vy),
            'acceleration': (ax, ay)
        }
    
    def _predict_adaptive(self, history: List[KinematicState], 
                        frames_ahead: int, dt: float) -> Dict[str, Any]:
        """Adaptive prediction that switches between models based on motion characteristics"""
        if len(history) < 3:
            return self._predict_constant_velocity(history, frames_ahead, dt)
        
        # Analyze recent motion to choose best model
        recent_states = history[-3:]
        
        # Calculate velocity and acceleration changes
        vel_changes = []
        acc_changes = []
        
        for i in range(1, len(recent_states)):
            prev_state = recent_states[i-1]
            curr_state = recent_states[i]
            
            # Velocity change
            vel_change = math.sqrt(
                (curr_state.velocity[0] - prev_state.velocity[0])**2 +
                (curr_state.velocity[1] - prev_state.velocity[1])**2
            )
            vel_changes.append(vel_change)
            
            # Acceleration change
            acc_change = math.sqrt(
                (curr_state.acceleration[0] - prev_state.acceleration[0])**2 +
                (curr_state.acceleration[1] - prev_state.acceleration[1])**2
            )
            acc_changes.append(acc_change)
        
        avg_vel_change = np.mean(vel_changes) if vel_changes else 0
        avg_acc_change = np.mean(acc_changes) if acc_changes else 0
        
        # Choose model based on motion characteristics
        if avg_acc_change < 0.5:  # Low acceleration change
            return self._predict_constant_velocity(history, frames_ahead, dt)
        elif avg_vel_change < 2.0:  # Low velocity change
            return self._predict_constant_acceleration(history, frames_ahead, dt)
        else:  # High variability - use weighted combination
            cv_pred = self._predict_constant_velocity(history, frames_ahead, dt)
            ca_pred = self._predict_constant_acceleration(history, frames_ahead, dt)
            
            # Weight based on recent accuracy
            weight = 0.6  # Favor constant velocity for high variability
            
            return {
                'position': (
                    weight * cv_pred['position'][0] + (1-weight) * ca_pred['position'][0],
                    weight * cv_pred['position'][1] + (1-weight) * ca_pred['position'][1]
                ),
                'velocity': (
                    weight * cv_pred['velocity'][0] + (1-weight) * ca_pred['velocity'][0],
                    weight * cv_pred['velocity'][1] + (1-weight) * ca_pred['velocity'][1]
                ),
                'acceleration': (
                    weight * cv_pred['acceleration'][0] + (1-weight) * ca_pred['acceleration'][0],
                    weight * cv_pred['acceleration'][1] + (1-weight) * ca_pred['acceleration'][1]
                )
            }
    
    def _combine_predictions(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine predictions from multiple models using weighted average"""
        if not predictions:
            return {'position': (0, 0), 'velocity': (0, 0), 'acceleration': (0, 0)}
        
        # Calculate weights
        total_weight = sum(self.model_weights.get(model, 0.1) for model in predictions.keys())
        
        if total_weight == 0:
            # Fallback to equal weights
            weight_per_model = 1.0 / len(predictions)
            weights = {model: weight_per_model for model in predictions.keys()}
        else:
            weights = {model: self.model_weights.get(model, 0.1) / total_weight 
                      for model in predictions.keys()}
        
        # Combine predictions
        combined_pos = (0.0, 0.0)
        combined_vel = (0.0, 0.0)
        combined_acc = (0.0, 0.0)
        
        for model, pred in predictions.items():
            weight = weights[model]
            combined_pos = (
                combined_pos[0] + weight * pred['position'][0],
                combined_pos[1] + weight * pred['position'][1]
            )
            combined_vel = (
                combined_vel[0] + weight * pred['velocity'][0],
                combined_vel[1] + weight * pred['velocity'][1]
            )
            combined_acc = (
                combined_acc[0] + weight * pred['acceleration'][0],
                combined_acc[1] + weight * pred['acceleration'][1]
            )
        
        return {
            'position': combined_pos,
            'velocity': combined_vel,
            'acceleration': combined_acc
        }
    
    def _calculate_uncertainty_growth(self, base_uncertainty: float, 
                                    frames_ahead: int, velocity: Tuple[float, float]) -> float:
        """Calculate how uncertainty grows over time"""
        # Base uncertainty growth
        time_factor = 1.0 + frames_ahead * 0.1
        
        # Velocity-dependent growth
        speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
        velocity_factor = 1.0 + speed * 0.05
        
        return base_uncertainty * time_factor * velocity_factor
    
    def _calculate_prediction_confidence(self, history: List[KinematicState], 
                                       prediction: Dict[str, Any], 
                                       frames_ahead: int) -> float:
        """Calculate confidence in the prediction"""
        if len(history) < 2:
            return 0.5
        
        # Base confidence decreases with prediction distance
        base_confidence = max(0.1, 1.0 - frames_ahead * 0.05)
        
        # Adjust based on recent prediction accuracy
        recent_accuracy = self._calculate_confidence(
            list(history)[-1].timestamp, 
            list(history)[-1].position, 
            list(history)[-1].velocity
        )
        
        # Combine base confidence with accuracy
        final_confidence = 0.7 * base_confidence + 0.3 * recent_accuracy
        
        return min(1.0, max(0.0, final_confidence))
    
    def get_track_statistics(self, track_id: int) -> Dict[str, Any]:
        """Get statistics for a specific track"""
        if track_id not in self.track_histories:
            return {}
        
        history = self.track_histories[track_id]
        if len(history) < 2:
            return {'track_id': track_id, 'history_length': len(history)}
        
        # Calculate statistics
        positions = [state.position for state in history]
        velocities = [state.velocity for state in history]
        accelerations = [state.acceleration for state in history]
        
        # Position statistics
        x_positions = [pos[0] for pos in positions]
        y_positions = [pos[1] for pos in positions]
        
        # Velocity statistics
        vx_values = [vel[0] for vel in velocities]
        vy_values = [vel[1] for vel in velocities]
        speeds = [math.sqrt(vx**2 + vy**2) for vx, vy in velocities]
        
        # Acceleration statistics
        ax_values = [acc[0] for acc in accelerations]
        ay_values = [acc[1] for acc in accelerations]
        acc_magnitudes = [math.sqrt(ax**2 + ay**2) for ax, ay in accelerations]
        
        return {
            'track_id': track_id,
            'history_length': len(history),
            'position_range': {
                'x_min': min(x_positions), 'x_max': max(x_positions),
                'y_min': min(y_positions), 'y_max': max(y_positions)
            },
            'velocity_stats': {
                'avg_speed': np.mean(speeds),
                'max_speed': max(speeds),
                'speed_std': np.std(speeds)
            },
            'acceleration_stats': {
                'avg_magnitude': np.mean(acc_magnitudes),
                'max_magnitude': max(acc_magnitudes),
                'magnitude_std': np.std(acc_magnitudes)
            },
            'last_state': {
                'position': history[-1].position,
                'velocity': history[-1].velocity,
                'acceleration': history[-1].acceleration,
                'uncertainty': history[-1].uncertainty,
                'confidence': history[-1].confidence
            }
        }
    
    def cleanup_old_tracks(self, active_track_ids: List[int], max_age_seconds: float = 60.0):
        """Remove old track histories that are no longer active"""
        current_time = time.time()
        tracks_to_remove = []
        
        for track_id, history in self.track_histories.items():
            if track_id not in active_track_ids and len(history) > 0:
                last_timestamp = history[-1].timestamp
                age = current_time - last_timestamp
                
                if age > max_age_seconds:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.track_histories[track_id]
        
        if tracks_to_remove:
            print(f"Cleaned up {len(tracks_to_remove)} old track histories")


if __name__ == "__main__":
    # Test the kinematic predictor
    predictor = KinematicPredictor()
    
    # Simulate a track
    track_id = 1
    positions = [(100, 100), (105, 102), (110, 104), (115, 106), (120, 108)]
    
    for i, pos in enumerate(positions):
        timestamp = i / predictor.fps  # Use actual FPS
        state = predictor.update_track(track_id, pos, timestamp)
        print(f"Frame {i}: pos={pos}, vel={state.velocity}, acc={state.acceleration}")
    
    # Test prediction
    pred = predictor.predict_position(track_id, 5, 0.2)
    print(f"Prediction 5 frames ahead: {pred.predicted_position}")
    print(f"Uncertainty: {pred.uncertainty}, Confidence: {pred.confidence}")
