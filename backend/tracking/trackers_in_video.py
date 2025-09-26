#!/usr/bin/env python3
"""Video object tracking with YOLO11 and ByteTrack."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import Iterable, Set

import cv2
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Script is in project_root/backend/tracking/, so go up 2 levels to reach project_root
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Default codec for MP4 output files.
MP4_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

tracker = sv.ByteTrack()
model = YOLO(os.path.join(PROJECT_ROOT, "models", "yolo11m.pt"))
annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
box_annotator = sv.BoxAnnotator()


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments."""
    default_video = os.path.join(PROJECT_ROOT, "data", "Cropped_Vid_720p.mp4")

    parser = argparse.ArgumentParser(description="Process a video and save an annotated copy")
    parser.add_argument(
        "video_path",
        type=str,
        nargs="?",
        default=default_video,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Overlay class names alongside tracker IDs",
    )
    parser.add_argument(
        "--ignore-classes",
        nargs="*",
        default=[],
        help="Class names to ignore (e.g. --ignore-classes car truck)",
    )
    return parser.parse_args()


def to_ignore_mask(detections: sv.Detections, ignore_classes: Set[str]) -> Iterable[bool]:
    """Return a mask of detections that should be kept."""
    names = model.names
    for class_id in detections.class_id:
        class_name = names[class_id] if class_id is not None else "unknown"
        yield class_name.lower() not in ignore_classes


def format_labels(detections: sv.Detections, show_labels: bool) -> list[str]:
    """Build label strings for the current detections."""
    labels: list[str] = []
    for idx in range(len(detections)):
        tracker_id = detections.tracker_id[idx]
        class_id = detections.class_id[idx]
        class_name = model.names[class_id] if class_id is not None else "Unknown"

        if show_labels:
            if tracker_id is not None:
                labels.append(f"{class_name} ID:{tracker_id}")
            else:
                labels.append(f"{class_name} New")
        else:
            if tracker_id is not None:
                labels.append(f"ID:{tracker_id}")
            else:
                labels.append("New")
    return labels


def annotate_frame(
    frame,
    frame_idx: int,
    progress_bar: tqdm,
    show_labels: bool,
    ignore_classes: Set[str],
):
    """Annotate a single frame with detections and tracking metadata."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(open(os.devnull, "w")):
            with redirect_stderr(open(os.devnull, "w")):
                result = model(frame, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(result)

    if ignore_classes:
        mask = list(to_ignore_mask(detections, ignore_classes))
        if any(mask):
            detections = detections[mask]
        else:
            progress_bar.update(1)
            return frame

    detections = tracker.update_with_detections(detections)

    annotated = box_annotator.annotate(frame, detections)
    labels = format_labels(detections, show_labels)
    if labels:
        annotated = annotator.annotate(annotated, detections, labels=labels)

    progress_bar.update(1)
    return annotated


def main() -> None:
    args = parse_arguments()
    ignore_classes = {name.lower() for name in args.ignore_classes}

    video_path = args.video_path
    target_path = f"{os.path.splitext(video_path)[0]}_output.mp4"

    cap_probe = cv2.VideoCapture(video_path)
    if not cap_probe.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    cap_probe.release()

    print(f"Video info: {total_frames} frames at {fps:.2f} FPS")
    print(f"Estimated duration: {total_frames / fps:.2f} seconds")
    print(f"Show labels: {args.show_labels}")
    if ignore_classes:
        print(f"Ignoring classes: {', '.join(sorted(ignore_classes))}")

    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
    if os.path.exists(target_path):
        backup_path = f"{os.path.splitext(target_path)[0]}_old.mp4"
        os.replace(target_path, backup_path)
        print(f"Existing output moved to: {backup_path}")

    progress_bar = tqdm(
        total=total_frames,
        desc="Processing video",
        unit="frames",
        position=0,
        leave=True,
        file=sys.stdout,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        progress_bar.close()
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(target_path, MP4_FOURCC, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        progress_bar.close()
        raise RuntimeError(
            "Failed to initialise video writer. Ensure that the 'mp4v' codec is available"
            " or install `ffmpeg`/`opencv-python` with codec support."
        )

    try:
        frame_index = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            annotated = annotate_frame(
                frame,
                frame_index,
                progress_bar,
                args.show_labels,
                ignore_classes,
            )
            writer.write(annotated)
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        progress_bar.close()

    print(f"\nProcessing complete! Output saved to: {target_path}")


if __name__ == "__main__":
    main()
