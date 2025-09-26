# EDTH Tracker GUI

A Tauri-based GUI for the EDTH Hackathon object tracking project, providing an intuitive interface for real-time tracking and video processing.

## Features

- **Real-time Object Tracking**: Choose between YOLO and RT-DETR transformer models
- **Video Processing**: Process video files with object tracking and save annotated outputs
- **System Information**: Check Python environment and dependencies
- **File Management**: Browse and select video files from the data directory

## Prerequisites

1. **Rust**: Install from [rustup.rs](https://rustup.rs/)
2. **Node.js**: Install from [nodejs.org](https://nodejs.org/)
3. **Python Dependencies**: Follow the main README setup instructions

## Quick Start

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start the GUI**:
   ```bash
   python start_gui.py
   ```

   Or manually:
   ```bash
   npm run tauri dev
   ```

## Building for Production

```bash
npm run tauri build
```

This will create a standalone executable in `src-tauri/target/release/`.

## GUI Features

### Real-time Tracking
- **YOLO Tracker**: Fast real-time tracking using ultralytics YOLO
- **Transformer Tracker**: More accurate tracking using RT-DETR model
- **Interactive Menu**: Command-line style tracker selection

### Video Processing
- Browse video files in the `data/` directory
- Process videos with object tracking
- Specify custom video paths
- Output videos are saved with `_output.mp4` suffix

### System Information
- Check Python version and OpenCV installation
- Verify that all dependencies are properly installed

## Architecture

The GUI consists of:

- **Frontend**: HTML/CSS/JavaScript interface using Tauri API
- **Backend**: Rust application that calls Python scripts
- **Python Bridge**: Commands that execute the existing tracking scripts

## Troubleshooting

1. **"Failed to execute python script"**: Ensure Python virtual environment is activated and dependencies are installed
2. **"No video files found"**: Place video files in the `data/` directory
3. **Camera access issues**: Grant camera permissions to the application

## File Structure

```
├── index.html          # Main GUI interface
├── main.js            # JavaScript logic and Tauri API calls
├── style.css          # GUI styling
├── vite.config.js     # Vite configuration
├── start_gui.py       # Startup script with dependency checks
├── src-tauri/         # Rust backend
│   ├── Cargo.toml     # Rust dependencies
│   ├── tauri.conf.json # Tauri configuration
│   └── src/main.rs    # Rust application logic
└── package.json       # Node.js dependencies
```

## Commands Available

The Rust backend exposes these commands to the frontend:

- `run_python_tracker(tracker_type)`: Start real-time tracking
- `process_video(video_path)`: Process a video file
- `get_system_info()`: Get Python/OpenCV version info
- `list_video_files()`: List available video files in data/

## Development

To modify the GUI:

1. Edit `index.html`, `main.js`, or `style.css` for frontend changes
2. Edit `src-tauri/src/main.rs` for backend command changes
3. Use `npm run tauri dev` for hot-reload development

The GUI automatically refreshes when frontend files change, but Rust changes require restart.