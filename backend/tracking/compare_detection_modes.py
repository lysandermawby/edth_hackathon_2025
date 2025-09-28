#!/usr/bin/env python3
"""
Visual comparison of different detection modes
Shows the actual difference between default, aggressive, and ultra-aggressive detection
"""

import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def compare_detection_modes(video_path, output_dir="detection_comparison"):
    """Compare different detection modes and save comparison images"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolo11m.pt")
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Detection configurations
    configs = [
        {
            'name': 'Default',
            'conf': 0.15,
            'iou': 0.7,
            'max_det': 1000,
            'agnostic_nms': False,
            'color': (0, 255, 0)  # Green
        },
        {
            'name': 'Aggressive',
            'conf': 0.05,
            'iou': 0.5,
            'max_det': 2000,
            'agnostic_nms': False,
            'color': (0, 255, 255)  # Yellow
        },
        {
            'name': 'Ultra-Aggressive',
            'conf': 0.02,
            'iou': 0.3,
            'max_det': 5000,
            'agnostic_nms': True,
            'color': (0, 0, 255)  # Red
        }
    ]
    
    # Process first 3 frames
    frame_count = 0
    while frame_count < 3:
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"\nProcessing frame {frame_count}...")
        
        # Create comparison image
        comparison_height = frame.shape[0] // 3
        comparison_width = frame.shape[1]
        comparison_image = np.zeros((frame.shape[0] + 100, frame.shape[1], 3), dtype=np.uint8)
        
        y_offset = 0
        detection_counts = {}
        
        for i, config in enumerate(configs):
            # Run detection
            result = model(
                frame, 
                conf=config['conf'],
                iou=config['iou'],
                max_det=config['max_det'],
                agnostic_nms=config['agnostic_nms'],
                verbose=False
            )[0]
            
            # Convert to supervision format
            detections = sv.Detections.from_ultralytics(result)
            
            # Count detections
            num_detections = len(detections.xyxy) if detections.xyxy is not None else 0
            detection_counts[config['name']] = num_detections
            
            # Resize frame for comparison
            resized_frame = cv2.resize(frame, (comparison_width, comparison_height))
            
            # Draw bounding boxes
            annotated_frame = resized_frame.copy()
            if len(detections.xyxy) > 0:
                for bbox in detections.xyxy:
                    x1, y1, x2, y2 = bbox.astype(int)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), config['color'], 2)
            
            # Add label
            label = f"{config['name']}: {num_detections} detections"
            cv2.putText(annotated_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, config['color'], 2)
            
            # Add to comparison image
            comparison_image[y_offset:y_offset+comparison_height, :] = annotated_frame
            y_offset += comparison_height
        
        # Add summary
        summary_y = frame.shape[0] + 20
        cv2.putText(comparison_image, f"Frame {frame_count} Detection Comparison", 
                   (10, summary_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        summary_text = f"Default: {detection_counts['Default']} | Aggressive: {detection_counts['Aggressive']} | Ultra-Aggressive: {detection_counts['Ultra-Aggressive']}"
        cv2.putText(comparison_image, summary_text, 
                   (10, summary_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save comparison image
        output_path = os.path.join(output_dir, f"frame_{frame_count}_comparison.jpg")
        cv2.imwrite(output_path, comparison_image)
        print(f"Saved comparison image: {output_path}")
        
        # Print detection counts
        print(f"  Default: {detection_counts['Default']} detections")
        print(f"  Aggressive: {detection_counts['Aggressive']} detections")
        print(f"  Ultra-Aggressive: {detection_counts['Ultra-Aggressive']} detections")
        
        frame_count += 1
    
    cap.release()
    print(f"\nComparison images saved in: {output_dir}")
    print("Green boxes = Default, Yellow boxes = Aggressive, Red boxes = Ultra-Aggressive")

if __name__ == "__main__":
    # Test with the challenging video
    video_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "Cropped_Vid_720p.mp4")
    compare_detection_modes(video_path)
