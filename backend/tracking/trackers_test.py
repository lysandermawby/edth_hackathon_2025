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
trace_annotator = sv.TraceAnnotator(trace_length=30)

def parse_arguments():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to the video file", nargs='?', default="../../data/Cropped_Vid_720p.mp4")
    return parser.parse_args()

def callback(frame, frame_idx, progress_bar):
    # Suppress YOLO output to avoid interfering with progress bar
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(open(os.devnull, 'w')):
            with redirect_stderr(open(os.devnull, 'w')):
                result = model(frame, verbose=False)[0]
    
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update_with_detections(detections)
    
    # Create labels with tracker IDs
    labels = []
    for i in range(len(detections)):
        if detections.tracker_id[i] is not None:
            labels.append(f"ID: {detections.tracker_id[i]}")
        else:
            labels.append("New")
    
    # Annotate frame with boxes and labels
    annotated_frame = box_annotator.annotate(frame, detections)
    annotated_frame = annotator.annotate(annotated_frame, detections, labels=labels)
    
    # Add trajectory traces (history lines)
    annotated_frame = trace_annotator.annotate(annotated_frame, detections)
    
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
    
    # Create progress bar with position to avoid conflicts
    progress_bar = tqdm(total=total_frames, desc="Processing video", unit="frames", 
                       position=0, leave=True, file=sys.stdout)
    
    try:
        sv.process_video(
            source_path=video_path,
            target_path=target_path,
            callback=lambda frame, frame_idx: callback(frame, frame_idx, progress_bar),
        )
    finally:
        progress_bar.close()
    
    print(f"\nProcessing complete! Output saved to: {target_path}")

if __name__ == "__main__":
    main()