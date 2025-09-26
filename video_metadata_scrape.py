#!/usr/bin/env python3
"""
Scrape metadata from a video file.
"""

import os
import json
import argparse
import cv2

def scrape_metadata(video_path):
    """Scrape metadata from a video file"""
    print(f"Video path: {video_path}")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 25)

# Consider the YUV420p format
# OpenCV will automatically convert, but be aware of potential quality loss


def parse_arguments():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to the video file", nargs='?', default="data/2025_09_17-15_02_07_MovingObjects_44.ts")
    return parser.parse_args()

def main():
    """Main script logic"""
    args = parse_arguments()

    scrape_metadata(args.video_path)

if __name__ == "__main__":
    main()
