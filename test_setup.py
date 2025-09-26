#!/usr/bin/env python3
"""
Test the complete setup of the EDTH Tracker GUI
"""

import subprocess
import os
import sys
import time

def test_python_dependencies():
    """Test if Python dependencies are available"""
    print("Testing Python dependencies...")

    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not available")
        return False

    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not available")
        return False

    try:
        import supervision as sv
        print(f"✓ Supervision available")
    except ImportError:
        print("✗ Supervision not available")
        return False

    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO available")
    except ImportError:
        print("✗ Ultralytics not available")
        return False

    try:
        from transformers import RTDetrV2ForObjectDetection
        print("✓ Transformers available")
    except ImportError:
        print("✗ Transformers not available")
        return False

    return True

def test_tauri_setup():
    """Test if Tauri is properly set up"""
    print("\nTesting Tauri setup...")

    # Check if Cargo.toml exists
    if not os.path.exists("src-tauri/Cargo.toml"):
        print("✗ Cargo.toml not found")
        return False
    print("✓ Cargo.toml exists")

    # Check if tauri.conf.json exists
    if not os.path.exists("src-tauri/tauri.conf.json"):
        print("✗ tauri.conf.json not found")
        return False
    print("✓ tauri.conf.json exists")

    # Check if main.rs exists
    if not os.path.exists("src-tauri/src/main.rs"):
        print("✗ main.rs not found")
        return False
    print("✓ main.rs exists")

    # Check if icons exist
    required_icons = ["32x32.png", "128x128.png", "icon.ico", "icon.icns"]
    for icon in required_icons:
        if not os.path.exists(f"src-tauri/icons/{icon}"):
            print(f"✗ {icon} not found")
            return False
    print("✓ All required icons exist")

    return True

def test_frontend_files():
    """Test if frontend files exist"""
    print("\nTesting frontend files...")

    required_files = ["index.html", "main.js", "package.json"]
    for file in required_files:
        if not os.path.exists(file):
            print(f"✗ {file} not found")
            return False
        print(f"✓ {file} exists")

    if not os.path.exists("dist/index.html"):
        print("✗ dist/index.html not found")
        return False
    print("✓ dist/index.html exists")

    return True

def test_data_directory():
    """Test data directory"""
    print("\nTesting data directory...")

    if not os.path.exists("data"):
        print("Creating data directory...")
        os.makedirs("data", exist_ok=True)

    video_files = []
    if os.path.exists("data"):
        for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
            import glob
            video_files.extend(glob.glob(f"data/{ext}"))

    print(f"✓ Data directory exists with {len(video_files)} video files")

    if len(video_files) == 0:
        print("  Note: No video files found. Place test videos in data/ directory")

    return True

def main():
    """Run all tests"""
    print("EDTH Tracker GUI Setup Test")
    print("=" * 40)

    tests = [
        ("Python Dependencies", test_python_dependencies),
        ("Tauri Setup", test_tauri_setup),
        ("Frontend Files", test_frontend_files),
        ("Data Directory", test_data_directory),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")

    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! The GUI setup is complete.")
        print("\nTo start the GUI:")
        print("  python start_gui.py")
        print("  or")
        print("  npm run tauri dev")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())