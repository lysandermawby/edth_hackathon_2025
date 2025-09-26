import supervision as sv
from trackers import SORTTracker
from ultralytics import YOLO

tracker = SORTTracker()
model = YOLO("yolo11m.pt")
annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

def callback(frame, _):
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update(detections)
    return annotator.annotate(frame, detections, labels=detections.tracker_id)

sv.process_video(
    source_path="/data/tracking_shortshort.mp4",
    target_path="/data/tracking_shortshort_output.mp4",
    callback=callback,
)