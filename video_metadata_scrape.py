#!/usr/bin/env python3
"""
Scrape metadata from a video file.
"""

import os
import json
import argparse
import cv2
import numpy as np
import ffmpeg
import subprocess

def scrape_metadata(video_path):
    """Scrape metadata from a video file"""
    print(f"Video path: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file")
        return None
    else:
        print("Video file opened successfully")

    # Basic video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {width} x {height}")

    # Frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frame Rate: {fps:.2f} FPS")

    # Total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total Frames: {total_frames}")

    # Duration
    duration = total_frames / fps if fps > 0 else 0
    print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

    # Additional OpenCV properties
    print("\nAdditional Video Properties:")
    print(f"Backend: {cap.getBackendName()}")
    print(f"FourCC: {int(cap.get(cv2.CAP_PROP_FOURCC)):08x}")
    print(f"Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    print(f"Contrast: {cap.get(cv2.CAP_PROP_CONTRAST)}")
    print(f"Saturation: {cap.get(cv2.CAP_PROP_SATURATION)}")
    print(f"Hue: {cap.get(cv2.CAP_PROP_HUE)}")
    print(f"Gain: {cap.get(cv2.CAP_PROP_GAIN)}")
    print(f"Exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    print(f"Auto Exposure: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
    print(f"White Balance Blue U: {cap.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U)}")
    print(f"White Balance Red V: {cap.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V)}")
    print(f"Rectification: {cap.get(cv2.CAP_PROP_RECTIFICATION)}")
    print(f"Monochrome: {cap.get(cv2.CAP_PROP_MONOCHROME)}")
    print(f"Sharpness: {cap.get(cv2.CAP_PROP_SHARPNESS)}")
    print(f"Auto Focus: {cap.get(cv2.CAP_PROP_AUTOFOCUS)}")
    print(f"Temperature: {cap.get(cv2.CAP_PROP_TEMPERATURE)}")
    print(f"Focus: {cap.get(cv2.CAP_PROP_FOCUS)}")
    print(f"Zoom: {cap.get(cv2.CAP_PROP_ZOOM)}")
    print(f"Pan: {cap.get(cv2.CAP_PROP_PAN)}")
    print(f"Tilt: {cap.get(cv2.CAP_PROP_TILT)}")
    print(f"Roll: {cap.get(cv2.CAP_PROP_ROLL)}")
    print(f"Iris: {cap.get(cv2.CAP_PROP_IRIS)}")
    print(f"Settings: {cap.get(cv2.CAP_PROP_SETTINGS)}")
    print(f"Buffer Size: {cap.get(cv2.CAP_PROP_BUFFERSIZE)}")
    print(f"Mode: {cap.get(cv2.CAP_PROP_MODE)}")
    print(f"Format: {cap.get(cv2.CAP_PROP_FORMAT)}")
    print(f"Codec: {cap.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT)}")
    print(f"Bitrate: {cap.get(cv2.CAP_PROP_BITRATE)}")
    print(f"Orientation Meta: {cap.get(cv2.CAP_PROP_ORIENTATION_META)}")
    print(f"Orientation Auto: {cap.get(cv2.CAP_PROP_ORIENTATION_AUTO)}")
    print(f"Open Timeout: {cap.get(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC)}")
    print(f"Read Timeout: {cap.get(cv2.CAP_PROP_READ_TIMEOUT_MSEC)}")

    cap.release()
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': total_frames,
        'duration': duration
    }


def extract_ffmpeg_metadata(video_path):
    """Extract detailed metadata using ffmpeg"""
    print("\n" + "=" * 50)
    print("FFMPEG METADATA EXTRACTION")
    print("=" * 50)
    
    try:
        # Get detailed metadata using ffprobe
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
        
        print("Format Information:")
        format_info = metadata.get('format', {})
        print(f"  Format: {format_info.get('format_long_name', 'Unknown')}")
        print(f"  Duration: {format_info.get('duration', 'Unknown')} seconds")
        print(f"  Size: {format_info.get('size', 'Unknown')} bytes")
        print(f"  Bitrate: {format_info.get('bit_rate', 'Unknown')} bps")
        print(f"  Start Time: {format_info.get('start_time', 'Unknown')} seconds")
        
        print("\nStream Information:")
        for i, stream in enumerate(metadata.get('streams', [])):
            print(f"\nStream {i}:")
            print(f"  Codec: {stream.get('codec_long_name', 'Unknown')}")
            print(f"  Type: {stream.get('codec_type', 'Unknown')}")
            if stream.get('codec_type') == 'video':
                print(f"  Resolution: {stream.get('width', 'Unknown')} x {stream.get('height', 'Unknown')}")
                print(f"  Pixel Format: {stream.get('pix_fmt', 'Unknown')}")
                print(f"  Profile: {stream.get('profile', 'Unknown')}")
                print(f"  Level: {stream.get('level', 'Unknown')}")
                print(f"  Color Range: {stream.get('color_range', 'Unknown')}")
                print(f"  Frame Rate: {stream.get('r_frame_rate', 'Unknown')}")
            elif stream.get('codec_type') == 'data':
                print(f"  Codec Tag: {stream.get('codec_tag_string', 'Unknown')}")
                print("  This appears to be a metadata stream (possibly KLV for drone telemetry)")
        
        return metadata
        
    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing ffprobe output: {e}")
        return None


def extract_klv_metadata(video_path):
    """Extract KLV metadata from drone video"""
    print("\n" + "=" * 50)
    print("KLV METADATA EXTRACTION (Drone Telemetry)")
    print("=" * 50)
    
    try:
        # Extract KLV data using ffmpeg
        # KLV data is typically in stream 1 for drone videos
        cmd = [
            'ffmpeg', '-i', video_path, '-map', '0:1', '-c', 'copy', '-f', 'data',
            '-', '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, check=True)
        
        if result.stdout:
            print(f"KLV data extracted: {len(result.stdout)} bytes")
            print("Raw KLV data (first 200 bytes):")
            print(result.stdout[:200].hex())
            print("\nNote: KLV data requires specialized parsing to extract GPS coordinates,")
            print("altitude, speed, and other telemetry data. This is the raw binary data.")
        else:
            print("No KLV data found in stream 1")
            
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        print(f"Error extracting KLV data: {e}")
        return None


def save_metadata_to_file(metadata, output_file):
    """Save metadata to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {output_file}")


def parse_arguments():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to the video file", nargs='?', default="data/2025_09_17-15_02_07_MovingObjects_44.ts")
    return parser.parse_args()

def main():
    """Main script logic"""
    args = parse_arguments()

    # Extract OpenCV metadata
    opencv_metadata = scrape_metadata(args.video_path)
    
    # Extract ffmpeg metadata
    ffmpeg_metadata = extract_ffmpeg_metadata(args.video_path)
    
    # Extract KLV metadata (drone telemetry)
    klv_data = extract_klv_metadata(args.video_path)
    
    # Combine all metadata
    all_metadata = {
        'opencv_metadata': opencv_metadata,
        'ffmpeg_metadata': ffmpeg_metadata,
        'klv_data_available': klv_data is not None,
        'klv_data_size': len(klv_data) if klv_data else 0
    }
    
    # Save to file
    output_file = f"{os.path.splitext(args.video_path)[0]}_metadata.json"
    save_metadata_to_file(all_metadata, output_file)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Video file: {args.video_path}")
    print(f"Resolution: {opencv_metadata['width']}x{opencv_metadata['height']}")
    print(f"Duration: {opencv_metadata['duration']:.2f} seconds")
    print(f"Frame Rate: {opencv_metadata['fps']:.2f} FPS")
    print(f"Total Frames: {opencv_metadata['total_frames']}")
    print(f"KLV Telemetry Data: {'Available' if klv_data else 'Not Available'}")
    if klv_data:
        print(f"KLV Data Size: {len(klv_data)} bytes")
    print(f"Metadata saved to: {output_file}")

if __name__ == "__main__":
    main()
