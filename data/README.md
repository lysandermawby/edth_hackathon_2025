# Data Directory

This directory contains data files for the EDTH London Hackathon 2025 project.

## Structure

- `__init__.py` - Python package marker
- `example_time_config.json` - Example configuration file for video cropping
- `quantum_drone_flight/` - Quantum drone flight data
- `cropped_videos/` - Output directory for cropped video segments (created by video cropping script)
- Other data files as needed

## Video Cropping

To use the video cropping functionality:

1. Copy `example_time_config.json` to `backend/video_crop/time_config.json`
2. Update the paths in the config to point to your actual video and data files
3. Run the cropping script: `python backend/video_crop/video_crop.py <video_path>`

## Git Ignore

Large binary files (videos, databases, etc.) are ignored by git to prevent repository bloat. Only configuration files, documentation, and small data files are tracked.
