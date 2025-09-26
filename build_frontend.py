#!/usr/bin/env python3
"""
Simple build script to prepare frontend files for Tauri
"""

import os
import shutil
from pathlib import Path

def build_frontend():
    """Copy frontend files to dist directory"""
    print("Building frontend for Tauri...")

    # Ensure dist directory exists
    dist_dir = Path("dist")
    dist_dir.mkdir(exist_ok=True)

    # Copy frontend files
    files_to_copy = ["index.html", "main.js"]

    for file in files_to_copy:
        if Path(file).exists():
            shutil.copy2(file, dist_dir / file)
            print(f"✓ Copied {file}")
        else:
            print(f"✗ Missing {file}")

    print(f"✓ Frontend built in {dist_dir}/")

if __name__ == "__main__":
    build_frontend()