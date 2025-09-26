#!/usr/bin/env python3
"""
Simple startup script for EDTH Tracker GUI
Bypasses dependency checking and starts the GUI directly
"""

import subprocess
import os
from pathlib import Path

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
    """Start the Tauri application with React + Vite"""
    try:
        print("Starting React + TypeScript + Vite + Tauri GUI...")
        print("Note: Some Python trackers may not work due to dependency issues")
        print("But the GUI interface should load successfully\n")

        # Change to frontend directory
        os.chdir("frontend")

        # Check if we're in development mode
        if os.path.exists("src-tauri/Cargo.toml") and os.path.exists("package.json"):
            print("Installing npm dependencies first...")
            subprocess.run(["npm", "install"], check=True)
            print("Starting Tauri dev server...")
            subprocess.run(["npm", "run", "tauri:dev"], check=True)
        else:
            print("Error: Missing Tauri configuration files")
            print("Make sure you're in the project root directory")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error starting Tauri app: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure npm is installed")
        print("2. Run 'npm install' to install dependencies")
        print("3. Check that Rust/Cargo is installed")
        print("4. Make sure TypeScript compilation works: npm run build")
        return False
    except FileNotFoundError:
        print("Error: npm not found. Please install Node.js and npm")
        print("Visit: https://nodejs.org/")
        return False
    except KeyboardInterrupt:
        print("\nGUI startup cancelled by user")
        return True
    return True

def main():
    """Main function - simplified startup"""
    print("EDTH Tracker GUI - Simple Startup")
    print("=" * 40)
    print("Skipping dependency checks...")
    print("Starting GUI directly...")
    print()

    # Check data directory
    check_data_directory()
    print()

    # Start Tauri app
    if not start_tauri_app():
        print("Failed to start GUI")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())