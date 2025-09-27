#!/usr/bin/env python3
"""Re-generate detections for an existing tracking session.

This helper reuses the IntegratedRealtimeTracker so the resulting detections are
identical to what the real-time pipeline would store, but it targets a specific
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

from integrated_realtime_tracker import IntegratedRealtimeTracker  # noqa: E402
from database_integration import TrackingDatabase  # noqa: E402


def resolve_video_path(video_path: str) -> str:
    """Return an absolute path for the provided video path."""
    potential_path = Path(video_path)
    if potential_path.is_absolute():
        return str(potential_path)

    absolute_path = PROJECT_ROOT / potential_path
    return str(absolute_path)


def clear_session_data(db_path: Path, session_id: int) -> None:
    """Remove existing detections for a session so they can be regenerated."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM tracked_objects WHERE session_id = ?", (session_id,))
        conn.execute(
            "UPDATE tracking_sessions SET total_frames = NULL, end_time = NULL WHERE session_id = ?",
            (session_id,),
        )
        conn.commit()


def fetch_session_details(db_path: Path, session_id: int) -> Optional[tuple[str, Optional[float]]]:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT video_path, fps FROM tracking_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return row if row else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate detections for an existing tracking session")
    parser.add_argument("session_id", type=int, help="Session identifier to process")
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Optional override for the video path. Defaults to the value stored in the database.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(PROJECT_ROOT / "databases" / "tracking_data.db"),
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "yolo11m.pt"),
        help="Path to the YOLO weights file",
    )
    parser.add_argument(
        "--ignore-classes",
        nargs="*",
        default=[],
        help="Optional list of class names to ignore while tracking",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Append to existing detections instead of clearing them first",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Draw class labels while processing (useful for debugging)",
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

    tracker = IntegratedRealtimeTracker(
        model_path=args.model_path,
        show_labels=args.show_labels,
        ignore_classes=args.ignore_classes,
        enable_database=True,
        db_path=str(db_path),
        headless=True,
    )

    tracker.run(video_path, session_id=args.session_id, reset_frame_count=not args.keep_existing)


if __name__ == "__main__":
    main()
