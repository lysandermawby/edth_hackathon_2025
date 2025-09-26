#!/usr/bin/env python3
"""
Real-time video playback with live object tracking and annotation.

Plays a pre-recorded video file with live bounding boxes, labels, and tracking IDs
displayed in real-time. Also provides tracking data output for database storage.
"""

import os
import argparse
import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
import time
import json
from datetime import datetime
import threading
import queue

class RealtimeVideoTracker:
    def __init__(self, model_path="yolo11m.pt", show_labels=True, ignore_classes=None):
        """
        Initialize the real-time video tracker
        
        Args:
            model_path (str): Path to YOLO model weights
            show_labels (bool): Whether to show class labels
            ignore_classes (list): List of class names to ignore
        """
        self.tracker = sv.ByteTrack()
        self.model = YOLO(model_path)
        self.annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
        self.box_annotator = sv.BoxAnnotator()
        self.show_labels = show_labels
        self.ignore_classes = ignore_classes or []
        
        # Tracking data storage
        self.tracking_data = []
        self.frame_count = 0
        self.start_time = None
        
        # Data output queue for real-time processing
        self.data_queue = queue.Queue()
        
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
        """Extract tracking data for database storage"""
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
                frame_data['objects'].append(obj_data)
        
        return frame_data
    
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
        self.tracking_data.append(tracking_data)
        
        # Add to real-time data queue
        self.data_queue.put(tracking_data)
        
        # Create labels
        labels = self.create_labels(detections)
        
        # Annotate frame
        annotated_frame = self.box_annotator.annotate(frame, detections)
        if labels:
            annotated_frame = self.annotator.annotate(annotated_frame, detections, labels=labels)
        
        return annotated_frame, detections, tracking_data
    
    def add_info_overlay(self, frame, fps, total_objects):
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
        
        # Controls info
        cv2.putText(frame, "Controls: 'q'=quit, 's'=save, 'r'=reset, SPACE=pause", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def save_tracking_data(self, output_file):
        """Save all tracking data to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.tracking_data, f, indent=2)
        print(f"Tracking data saved to: {output_file}")
    
    def run(self, video_path, save_data=True):
        """Main loop for real-time video tracking"""
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
        
        print("\n=== Real-time Video Tracking ===")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("Press 'r' to reset tracker")
        print("Press SPACE to pause/resume")
        print("Press 'd' to save tracking data")
        print("================================\n")
        
        self.start_time = time.time()
        paused = False
        frame_times = []
        
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
                    
                    # Add info overlay
                    total_objects = len(tracking_data['objects'])
                    annotated_frame = self.add_info_overlay(annotated_frame, current_fps, total_objects)
                    
                    # Display frame
                    cv2.imshow('Real-time Video Tracking', annotated_frame)
                    
                    self.frame_count += 1
                    
                    # Print tracking info every 30 frames
                    if self.frame_count % 30 == 0 and tracking_data['objects']:
                        print(f"Frame {self.frame_count}: {len(tracking_data['objects'])} objects tracked")
                        for obj in tracking_data['objects']:
                            print(f"  - {obj['class_name']} ID:{obj['tracker_id']} at ({obj['center']['x']:.1f}, {obj['center']['y']:.1f})")
                
                else:
                    # Show paused message
                    paused_frame = frame.copy() if 'frame' in locals() else np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(paused_frame, "PAUSED - Press SPACE to resume", 
                              (50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Real-time Video Tracking', paused_frame)
                
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
                    print("Tracker reset")
                elif key == ord('d'):
                    # Save tracking data
                    if save_data:
                        output_file = f"{os.path.splitext(video_path)[0]}_tracking_data.json"
                        self.save_tracking_data(output_file)
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
            cv2.destroyAllWindows()
            
            # Auto-save tracking data if enabled
            if save_data and self.tracking_data:
                output_file = f"{os.path.splitext(video_path)[0]}_tracking_data.json"
                self.save_tracking_data(output_file)
            
            print("Video processing complete!")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Real-time video tracking with live display")
    parser.add_argument("video_path", type=str, help="Path to the video file", 
                       nargs='?', default="../../data/Cropped_Vid_720p.mp4")
    parser.add_argument("--show-labels", action="store_true", 
                       help="Show class labels on bounding boxes")
    parser.add_argument("--ignore-classes", nargs="*", default=[], 
                       help="List of class names to ignore (e.g., --ignore-classes car truck)")
    parser.add_argument("--no-save", action="store_true", 
                       help="Don't save tracking data to file")
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    tracker = RealtimeVideoTracker(
        model_path="yolo11m.pt",
        show_labels=args.show_labels,
        ignore_classes=args.ignore_classes
    )
    
    tracker.run(args.video_path, save_data=not args.no_save)

if __name__ == "__main__":
    main()
