#!/usr/bin/env python3
"""
Startup script for EDTH Tracker GUI
This script handles the communication between the Tauri frontend and Python backend
"""

import sys
import subprocess
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking Python dependencies...")

    # Check core dependencies
    deps_status = {}

    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
        deps_status['opencv'] = True
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
        deps_status['opencv'] = False

    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        deps_status['torch'] = True
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        deps_status['torch'] = False

    try:
        import supervision
        print("✓ Supervision: Available")
        deps_status['supervision'] = True
    except ImportError as e:
        print(f"✗ Supervision: {e}")
        deps_status['supervision'] = False

    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO: Available")
        deps_status['yolo'] = True
    except ImportError as e:
        print(f"✗ Ultralytics YOLO: {e}")
        deps_status['yolo'] = False

    # Check transformers separately with better error handling
    try:
        import transformers
        print("✓ Transformers: Available (RT-DETR may have compatibility issues)")
        deps_status['transformers'] = True
    except ImportError as e:
        print(f"✗ Transformers: {e}")
        deps_status['transformers'] = False
    except Exception as e:
        print(f"⚠ Transformers: Available but has compatibility issues ({str(e)[:50]}...)")
        print("  Note: RT-DETR tracker may not work due to NumPy/TensorFlow compatibility")
        deps_status['transformers'] = 'partial'

    # Determine if we can run basic functionality
    basic_deps = deps_status.get('opencv', False) and deps_status.get('torch', False)
    yolo_ready = basic_deps and deps_status.get('yolo', False)

    if basic_deps:
        print("\n✓ Core dependencies available - GUI can start")
        if yolo_ready:
            print("✓ YOLO tracker should work")
        else:
            print("⚠ YOLO tracker may not work")

        if deps_status.get('transformers') == True:
            print("✓ Transformer tracker should work")
        elif deps_status.get('transformers') == 'partial':
            print("⚠ Transformer tracker may have issues")
        else:
            print("✗ Transformer tracker will not work")

        return True
    else:
        print("\n✗ Missing critical dependencies")
        print("\nTo fix dependency issues:")
        print("1. Activate your virtual environment:")
        print("   source .venv/bin/activate")
        print("2. Install dependencies:")
        print("   poetry install")
        print("   pip install trackers")
        print("   pip install supervision==0.21.0")
        print("3. Fix NumPy compatibility:")
        print("   pip install 'numpy<2.0'")
        return False

def check_data_directory():
    """Check if data directory exists"""
    data_dir = Path("data")
    if not data_dir.exists():
        print("Creating data directory...")
        data_dir.mkdir()
        print("✓ Data directory created")
        print("Please place your video files in the 'data' directory")
    else:
        video_files = list(data_dir.glob("*.mp4")) + list(data_dir.glob("*.avi")) + list(data_dir.glob("*.mov"))
        print(f"✓ Data directory exists with {len(video_files)} video files")

def start_tauri_app():
    """Start the Tauri application"""
    try:
        print("Starting Tauri GUI...")
        # Check if we're in development or production mode
        if os.path.exists("src-tauri/Cargo.toml"):
            subprocess.run(["npm", "run", "tauri", "dev"], check=True)
        else:
            print("Run 'npm run tauri build' to create a production build")
    except subprocess.CalledProcessError as e:
        print(f"Error starting Tauri app: {e}")
        return False
    except FileNotFoundError:
        print("Error: npm or tauri not found. Please install:")
        print("npm install -g @tauri-apps/cli")
        return False
    return True

def main():
    """Main function"""
    print("EDTH Tracker GUI Startup")
    print("=" * 40)

    # Check Python dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check data directory
    check_data_directory()

    # Start Tauri app
    if not start_tauri_app():
        sys.exit(1)

if __name__ == "__main__":
    main()