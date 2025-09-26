import supervision as sv
from ultralytics import YOLO
import argparse
import os
import cv2
from tqdm import tqdm
import sys
from contextlib import redirect_stdout, redirect_stderr

# Default codec used for saving MP4 outputs. Declared as constant so it is easy to tweak
MP4_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

tracker = sv.ByteTrack()
model = YOLO("yolo11m.pt")
annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
box_annotator = sv.BoxAnnotator()

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
    
    # Ensure the data directory exists and remove stale outputs so that cv2 can overwrite
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path):
        backup_path = f"{os.path.splitext(target_path)[0]}_old.mp4"
        os.replace(target_path, backup_path)
        print(f"Existing output moved to: {backup_path}")

    # Create progress bar with position to avoid interfering with stdout captures
    progress_bar = tqdm(total=total_frames, desc="Processing video", unit="frames",
                       position=0, leave=True, file=sys.stdout)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        progress_bar.close()
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(target_path, MP4_FOURCC, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        progress_bar.close()
        raise RuntimeError(
            "Failed to initialise video writer. Ensure that the 'mp4v' codec is "
            "available or install `ffmpeg`/`opencv-python` with codec support."
        )

    try:
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = callback(frame, frame_index, progress_bar)
            writer.write(annotated_frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        progress_bar.close()

    print(f"\nProcessing complete! Output saved to: {target_path}")

if __name__ == "__main__":
    main()
