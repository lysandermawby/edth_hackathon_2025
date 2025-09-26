#!/usr/bin/env python3
"""
Real-time Object Tracking Launcher

This script allows you to choose between different detection models for real-time tracking:
1. YOLO-based tracking (faster, good for real-time)
2. RT-DETR transformer-based tracking (more accurate, slower)

Usage:
    python run_realtime_tracking.py
"""

import sys
import os

def main():
    print("=" * 50)
    print("Real-time Object Tracking")
    print("=" * 50)
    print()
    print("Choose your detection model:")
    print("1. YOLO (Ultralytics) - Fast, good for real-time")
    print("2. RT-DETR (Transformer) - More accurate, slower")
    print("3. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == "1":
                print("\nStarting YOLO-based real-time tracking...")
                print("Note: Make sure you have 'yolo11m.pt' model file in your directory")
                print("You can download it from: https://github.com/ultralytics/assets/releases")
                print()
                from realtime_tracker import main as yolo_main
                yolo_main()
                break
                
            elif choice == "2":
                print("\nStarting RT-DETR transformer-based real-time tracking...")
                print("Note: This will download the model on first run (~100MB)")
                print()
                from realtime_tracker_transformer import main as transformer_main
                transformer_main()
                break
                
            elif choice == "3":
                print("Exiting...")
                sys.exit(0)
                
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
        except ImportError as e:
            print(f"Error importing required modules: {e}")
            print("Make sure you have installed all required dependencies:")
            print("pip install opencv-python ultralytics supervision transformers torch")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
