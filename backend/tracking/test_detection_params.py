#!/usr/bin/env python3
"""
Test script to compare detection parameters and see actual differences
"""

import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_detection_params(video_path, frame_limit=10):
    """Test different detection parameters on the same video"""
    
    # Load model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolo11m.pt")
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Test configurations
    configs = [
        {
            'name': 'Default',
            'conf': 0.15,
            'iou': 0.7,
            'max_det': 1000,
            'agnostic_nms': False
        },
        {
            'name': 'Aggressive',
            'conf': 0.05,
            'iou': 0.5,
            'max_det': 2000,
            'agnostic_nms': False
        },
        {
            'name': 'Ultra-Aggressive',
            'conf': 0.02,
            'iou': 0.3,
            'max_det': 5000,
            'agnostic_nms': True
        }
    ]
    
    results = {config['name']: [] for config in configs}
    
    frame_count = 0
    while frame_count < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break
            
        print(f"\n--- Frame {frame_count} ---")
        
        for config in configs:
            # Run detection with this config
            result = model(
                frame, 
                conf=config['conf'],
                iou=config['iou'],
                max_det=config['max_det'],
                agnostic_nms=config['agnostic_nms'],
                verbose=False
            )[0]
            
            # Count detections
            num_detections = len(result.boxes) if result.boxes is not None else 0
            results[config['name']].append(num_detections)
            
            print(f"{config['name']}: {num_detections} detections")
        
        frame_count += 1
    
    cap.release()
    
    # Print summary
    print("\n" + "="*50)
    print("DETECTION COMPARISON SUMMARY")
    print("="*50)
    
    for config_name, detections in results.items():
        avg_detections = np.mean(detections)
        max_detections = np.max(detections)
        min_detections = np.min(detections)
        
        print(f"\n{config_name}:")
        print(f"  Average detections per frame: {avg_detections:.1f}")
        print(f"  Max detections in a frame: {max_detections}")
        print(f"  Min detections in a frame: {min_detections}")
        print(f"  Total detections: {sum(detections)}")

if __name__ == "__main__":
    # Test with a more challenging video
    video_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "Cropped_Vid_720p.mp4")
    test_detection_params(video_path, frame_limit=5)
