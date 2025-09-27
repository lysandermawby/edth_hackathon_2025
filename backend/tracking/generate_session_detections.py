#!/usr/bin/env python3
"""Re-generate detections for an existing tracking session.

This helper reuses the RealtimeReidentificationTracker (improved algorithm) so the 
resulting detections include enhanced re-identification capabilities and are
improved compared to the basic IntegratedRealtimeTracker. It targets a specific
session already present in the SQLite database. Existing detections can be
cleared before reprocessing to guarantee the overlays match the video segment.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

# Ensure we can import the tracker modules
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

# Import the improved re-identification tracker instead of basic tracker
from realtime_reidentification_tracker import RealtimeReidentificationTracker  # noqa: E402


def fetch_session_details(db_path: Path, session_id: int) -> Optional[tuple[str, Optional[float]]]:
    """Fetch the video path and FPS for a given session."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT video_path, fps FROM tracking_sessions WHERE session_id = ?",
                (session_id,),
            )
            result = cursor.fetchone()
            return result if result else None
    except sqlite3.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        return None


def clear_session_data(db_path: Path, session_id: int) -> None:
    """Clear all tracking data for the specified session."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM tracked_objects WHERE session_id = ?", (session_id,))
            conn.commit()
            print(f"Cleared existing data for session {session_id}")
    except sqlite3.Error as e:
        print(f"Database error during cleanup: {e}", file=sys.stderr)


def resolve_video_path(video_path: str) -> str:
    """Resolve video path to absolute path."""
    if os.path.isabs(video_path):
        return video_path
    
    # Try relative to project root
    project_video_path = PROJECT_ROOT / video_path
    if project_video_path.exists():
        return str(project_video_path)
    
    # Try relative to current directory
    cwd_video_path = Path.cwd() / video_path
    if cwd_video_path.exists():
        return str(cwd_video_path)
    
    # Return as-is if not found (will be handled by caller)
    return video_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-generate detections for an existing session using improved re-identification algorithm"
    )
    parser.add_argument("session_id", type=int, help="Session ID to regenerate")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=PROJECT_ROOT / "databases" / "tracking_data.db",
        help="Path to tracking database",
    )
    parser.add_argument(
        "--video-path", type=str, help="Override video path (optional)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "yolo11m.pt"),
        help="Path to YOLO model",
    )
    parser.add_argument(
        "--ignore-classes", nargs="*", help="Class names to ignore during detection"
    )
    parser.add_argument(
        "--show-labels", action="store_true", help="Show class labels on detections"
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep existing detections (append new ones)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    db_path = Path(args.db_path).resolve()
    if not db_path.exists():
        print(f"Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    session_details = fetch_session_details(db_path, args.session_id)
    if session_details is None:
        print(f"Session {args.session_id} not found in database", file=sys.stderr)
        sys.exit(1)

    stored_video_path, stored_fps = session_details
    video_path = resolve_video_path(args.video_path or stored_video_path)

    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if not args.keep_existing:
        clear_session_data(db_path, args.session_id)

    # Update FPS information in the session if needed
    with sqlite3.connect(db_path) as conn:
        if stored_fps is None:
            import cv2  # Lazy import: only needed when fps missing

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            conn.execute(
                "UPDATE tracking_sessions SET fps = ? WHERE session_id = ?",
                (fps, args.session_id),
            )
            conn.commit()

    # Use the improved re-identification tracker instead of basic tracker
    print(f"Using improved re-identification tracker for session {args.session_id}")
    tracker = RealtimeReidentificationTracker(
        model_path=args.model_path,
        show_labels=args.show_labels,
        ignore_classes=args.ignore_classes,
        enable_database=True,
        db_path=str(db_path),
        max_occlusion_frames=60  # Enhanced re-identification parameter
    )

    tracker.run(video_path, save_data=True)


if __name__ == "__main__":
    main()