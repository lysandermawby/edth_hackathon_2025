import supervision as sv
from ultralytics import YOLO
import argparse
import os

tracker = sv.ByteTrack()
model = YOLO("yolo11m.pt")
annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
box_annotator = sv.BoxAnnotator()

def parse_arguments():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to the video file", nargs='?', default="../../data/Cropped_Vid_720p.mp4")
    return parser.parse_args()

def callback(frame, _):
    result = model(frame)[0]
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
    
    return annotated_frame

def main():
    args = parse_arguments()
    video_path = args.video_path
    target_path = f"{os.path.splitext(video_path)[0]}_output.mp4"
    sv.process_video(
        source_path=video_path,
        target_path=target_path,
        callback=callback,
    )

if __name__ == "__main__":
    main()