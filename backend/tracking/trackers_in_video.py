#!/usr/bin/env python3
"""
Object tracking in a video file using YOLO11 model (saved as yolo11m.pt).

Accesses a video file and tracks objects in it.
Creates another output video file containing tracking bounding boxes and labels.
"""

import os
import argparse
import cv2
import supervision as sv

import supervision as sv
from ultralytics import YOLO
import argparse
import os
import cv2
from tqdm import tqdm
import sys
from contextlib import redirect_stdout, redirect_stderr

tracker = sv.ByteTrack()
model = YOLO("yolo11m.pt")
annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
box_annotator = sv.BoxAnnotator()

def parse_arguments():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to the video file", nargs='?', default="../../data/Cropped_Vid_720p.mp4")
    parser.add_argument("--show-labels", action="store_true", help="Show labels on bounding boxes")
    parser.add_argument("--ignore-classes", nargs="*", default=[], help="List of class names to ignore (e.g., --ignore-classes car truck)")
    return parser.parse_args()

def callback(frame, frame_idx, progress_bar, show_labels, ignore_classes):
    # Suppress YOLO output to avoid interfering with progress bar
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(open(os.devnull, 'w')):
            with redirect_stderr(open(os.devnull, 'w')):
                result = model(frame, verbose=False)[0]
    
    detections = sv.Detections.from_ultralytics(result)
    
    # Filter out ignored classes
    if ignore_classes:
        # Get class names from YOLO result
        class_names = model.names
        keep_mask = []
        for class_id in detections.class_id:
            class_name = class_names[class_id]
            keep_mask.append(class_name.lower() not in [c.lower() for c in ignore_classes])
        
        if any(keep_mask):
            detections = detections[keep_mask]
        else:
            # If all detections are filtered out, return original frame
            progress_bar.update(1)
            return frame
    
    detections = tracker.update_with_detections(detections)
    
    # Create labels based on show_labels flag
    labels = []
    if show_labels:
        for i in range(len(detections)):
            # Get class name and tracker ID
            class_name = model.names[detections.class_id[i]] if detections.class_id[i] is not None else "Unknown"
            if detections.tracker_id[i] is not None:
                labels.append(f"{class_name} ID:{detections.tracker_id[i]}")
            else:
                labels.append(f"{class_name} New")
    else:
        # Just show tracker IDs without class names
        for i in range(len(detections)):
            if detections.tracker_id[i] is not None:
                labels.append(f"ID: {detections.tracker_id[i]}")
            else:
                labels.append("New")
    
    # Annotate frame with boxes and labels
    annotated_frame = box_annotator.annotate(frame, detections)
    if show_labels or any(detections.tracker_id):
        annotated_frame = annotator.annotate(annotated_frame, detections, labels=labels)
    
    # Update progress bar
    progress_bar.update(1)
    
    return annotated_frame

def main():
    args = parse_arguments()
    video_path = args.video_path
    target_path = f"{os.path.splitext(video_path)[0]}_output.mp4"
    
    # Get total frame count for progress bar
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"Video info: {total_frames} frames at {fps:.2f} FPS")
    print(f"Estimated duration: {total_frames/fps:.2f} seconds")
    print(f"Show labels: {args.show_labels}")
    if args.ignore_classes:
        print(f"Ignoring classes: {', '.join(args.ignore_classes)}")
    
    # Create progress bar with position to avoid conflicts
    progress_bar = tqdm(total=total_frames, desc="Processing video", unit="frames", 
                       position=0, leave=True, file=sys.stdout)
    
    try:
        sv.process_video(
            source_path=video_path,
            target_path=target_path,
            callback=lambda frame, frame_idx: callback(frame, frame_idx, progress_bar, args.show_labels, args.ignore_classes),
        )
    finally:
        progress_bar.close()
    
    print(f"\nProcessing complete! Output saved to: {target_path}")

if __name__ == "__main__":
    main()