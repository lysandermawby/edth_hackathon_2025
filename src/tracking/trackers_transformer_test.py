import torch
import supervision as sv
from trackers import SORTTracker
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

tracker = SORTTracker()
image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")
annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

def callback(frame, _):
    inputs = image_processor(images=frame, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    h, w, _ = frame.shape
    results = image_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([(h, w)]),
        threshold=0.5
    )[0]

    detections = sv.Detections.from_transformers(
        transformers_results=results,
        id2label=model.config.id2label
    )

    detections = tracker.update(detections)
    return annotator.annotate(frame, detections, labels=detections.tracker_id)


sv.process_video(
    source_path="/data/tracking_shortshort.mp4",
    target_path="/data/tracking_shortshort_output_transformerp.mp4",
    callback=callback,
)
