# EDTH Hackathon London 2025

Real-time object detection and tracking system for the European Defence Tech Hackathon in London 2025.

See [this information spreadsheet](https://docs.google.com/spreadsheets/d/1AT8ndsEe9hgljTUFgmpAJ_pYE9kLxrVdJI-xuT68uto/edit?referrer=luma&gid=2054763765).
Data is available through [this sharepoint](https://quantumdrones-my.sharepoint.com/personal/pzimmermann_quantum-systems_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpzimmermann%5Fquantum%2Dsystems%5Fcom%2FDocuments%2F2025%5FLondon%5FHackathon&ga=1).

## 🎯 Project Features

**Complete real-time object detection and tracking pipeline** tackling Quantum-6 (and Stark-1):

✅ **Real-time object detection** using YOLO11  
✅ **Multi-object tracking** with ByteTrack (persistent IDs)  
✅ **Robust re-identification** with appearance and kinematic features  
✅ **Live web interface** with interactive visualization  
✅ **Camera and video file support**  
✅ **Database integration** for session history  
✅ **Interactive tooltips** with detection details  
✅ **Performance monitoring** (FPS, object count)  
✅ **Drone video analysis** with GPS mapping  
✅ **Video processing and cropping** tools  
✅ **Comprehensive data visualization** and analysis  

### Core Capabilities

1. **Identify objects in video in real time** - YOLO11 detection with 30+ FPS
2. **Estimate velocity, acceleration, and class** - Multi-object tracking with ByteTrack
3. **Maintain live tracking** - Persistent object IDs across frames with re-identification
4. **Interactive web visualization** - Real-time canvas with detection overlays
5. **Drone video analysis** - GPS mapping and telemetry integration
6. **Database storage and analysis** - SQLite with comprehensive visualization tools

## 🚀 Quick Start (Recommended)

### **Option A: Complete Real-time System** 🔴

**For live camera feed and real-time detection:**

```bash
cd standalone-frontend
./start-realtime.sh
```

Then open **http://localhost:5173** and:

1. Toggle to **"🔴 Live Detection"** mode
2. Click **"Connect"** to connect to real-time server
3. Click **"Start Webcam"** for live camera OR **"Start Sample Video"** for video files
4. **Hover over detections** for detailed information!

**Features:**

- 🎯 **Live camera feed** with real-time object detection
- 🎨 **Interactive visualization** with hover tooltips
- 📊 **Performance stats** (FPS, object count, frame numbers)
- 🗄️ **Database storage** of detection sessions
- 🔄 **Dual mode**: Live detection + recorded session playback

### **Option B: Recorded Session Viewer** 📁

**To view pre-processed tracking sessions:**

1. First, process a video (see Backend Processing section)
2. Run the standalone frontend:
   ```bash
   cd standalone-frontend
   ./start-realtime.sh
   ```
3. Open **http://localhost:5173** and stay in **"📁 Recorded Sessions"** mode
4. Select a session to view video with detection overlays

## 🔧 Full Setup Guide

### Prerequisites

**Python 3.11+** and **Node.js 18+** are required.

### 1. Environment Setup

Create and activate a Python virtual environment:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Python Dependencies

Install the backend dependencies using Poetry:

```bash
cd backend
poetry install
```

### 3. **Critical Dependency Fix** ⚠️

There is a known incompatibility between packages. **You must run these commands or the system will not work:**

```bash
pip install trackers
pip install supervision==0.21.0
pip install websockets  # For real-time server
```

### 4. Install Frontend Dependencies

```bash
cd standalone-frontend
npm install
```

### 5. Add Video Files (Optional)

Place video files in the `data/` directory for processing or use the default sample video.

---

## 🎮 Usage Options

### 🔴 **Real-time Detection (Recommended)**

**Start the complete system with one command:**

```bash
cd standalone-frontend
./start-realtime.sh
```

This starts:

- 🎯 **Real-time detection server** (ws://localhost:8765)
- 🗄️ **API server** (http://localhost:3001)
- 🌐 **Frontend interface** (http://localhost:5173)

**Then:**

1. Open **http://localhost:5173** in your browser
2. Toggle to **"🔴 Live Detection"** mode
3. Click **"Connect"** → **"Start Webcam"** or **"Start Sample Video"**
4. **Hover over detections** for detailed tooltips!

---

## 🛠️ Backend Processing (Advanced)

### **Core Tracking Commands**

All commands use the `make` system for easy execution:

#### **Basic Object Detection & Tracking**

```bash
# Run integrated real-time tracker on default video
make analyse

# Run real-time camera tracking with YOLO11
make realtime_tracker

# Run enhanced kinematic re-identification tracker
make kinematic_reid_tracker
```

#### **Re-identification Tracking**

```bash
# Run re-identification tracker on quantum drone video
make reidentification_tracker
```

#### **Drone Video Analysis**

```bash
# Launch drone video and map viewer with GPS tracking
make map_viewer
```

#### **Video Processing**

```bash
# Crop video using time configuration
make crop
```

### **Database Analysis & Visualization**

```bash
# Show database schema and basic tracking data
make db_visualize

# Show data for specific session
make db_visualize_session SESSION_ID=1

# Show database data with plots
make db_visualize_plots

# Comprehensive database analysis with plots
make db_visualize_all
```

### **Frontend Management**

```bash
# Start frontend application (both recorded and real-time)
make frontend

# Install all frontend and Python dependencies
make frontend_install

# Clean up frontend processes
make frontend_clean
```

### **Advanced Video Processing**

**Generate annotated video output:**

```bash
python backend/tracking/trackers_in_video.py data/your_video.mp4
```

Creates `data/your_video_output.mp4` with bounding boxes.

**Generate database + tracking data:**

```bash
python backend/tracking/integrated_realtime_tracker.py data/your_video.mp4 --headless
```

- Writes to `databases/tracking_data.db`
- Creates `data/your_video_tracking_data.json`

### **Metadata Extraction**

**Extract video metadata and telemetry:**

```bash
python backend/metadata_process/video_metadata_scrape.py data/your_video.mp4
```

**Parse drone telemetry data:**

```bash
python backend/metadata_process/enhanced_telemetry_parser.py data/quantum_drone_flight/metadata.csv
```

### **Session Management**

**Generate detections for existing session:**

```bash
python backend/tracking/generate_session_detections.py --session-id 1 --video-path data/your_video.mp4
```

---

## 🏗️ System Architecture

```
📹 Input (Camera/Video)
    ↓
🤖 YOLO11 Detection
    ↓
🎯 ByteTrack Multi-Object Tracking
    ↓
🔄 Re-identification System (Appearance + Kinematic)
    ↓
🌐 WebSocket Streaming
    ↓
⚛️ React Frontend Canvas
    ↓
🗄️ SQLite Database Storage
    ↓
📊 Data Visualization & Analysis
```

## 📋 Project Structure

```
├── backend/
│   ├── tracking/                    # Core detection & tracking
│   │   ├── integrated_realtime_tracker.py      # Main tracking with DB
│   │   ├── realtime_reidentification_tracker.py # Enhanced re-ID tracking
│   │   ├── realtime_tracker.py                # Camera tracking
│   │   ├── trackers_in_video.py               # Video processing
│   │   ├── kinematic_reidentification.py      # Motion-based re-ID
│   │   ├── database_integration.py           # DB management
│   │   └── drone_video_map_viewer.py         # GPS mapping
│   ├── db_query/
│   │   └── data_visualisation.py             # Database analysis
│   ├── metadata_process/
│   │   ├── video_metadata_scrape.py          # Video metadata extraction
│   │   └── enhanced_telemetry_parser.py      # Drone telemetry parsing
│   ├── video_crop/
│   │   └── video_crop.py                     # Video cropping tools
│   └── reidentify/
│       ├── robust_reidentification.py        # Appearance-based re-ID
│       └── reidentification.py              # Core re-ID system
├── standalone-frontend/             # 🔴 Real-time web interface
│   ├── start-realtime.sh           # One-command startup
│   ├── realtime_server.py          # WebSocket detection server
│   ├── src/RealtimeVideoCanvas.tsx # Live detection canvas
│   └── server.js                   # API for recorded sessions
├── data/                           # Video files and datasets
│   ├── quantum_drone_flight/       # Drone video data
│   ├── VisDrone2019-MOT-val/       # Validation dataset
│   └── metadata/                   # Extracted metadata
├── databases/                      # SQLite tracking data
├── models/                         # YOLO model weights
└── Makefile                       # Command automation
```

## ⚡ Key Features

### **Detection & Tracking**
- **🎯 Real-time Object Detection**: YOLO11 at 30+ FPS
- **🔄 Multi-Object Tracking**: ByteTrack with persistent IDs
- **🔍 Robust Re-identification**: Appearance + kinematic features
- **📊 Live Performance Metrics**: FPS, object count, frame numbers
- **📹 Flexible Input**: Camera feed or video files

### **Visualization & Analysis**
- **🎨 Interactive Visualization**: Hover tooltips with detection details
- **🗄️ Session Management**: Database storage and replay
- **🌐 Web-based Interface**: No installation required
- **📊 Comprehensive Analytics**: Class distribution, tracking statistics
- **🗺️ GPS Mapping**: Drone video with location overlays

### **Data Processing**
- **📁 Video Processing**: Cropping, metadata extraction
- **📈 Database Analysis**: SQLite with visualization tools
- **🛰️ Telemetry Integration**: Drone GPS and sensor data
- **💾 Export Capabilities**: JSON, CSV, annotated videos

## 🎮 Interactive Controls

### **Real-time Detection**
- **Web Interface**: Toggle between live detection and recorded sessions
- **Camera Controls**: Start/stop webcam, select video files
- **Detection Info**: Hover over bounding boxes for details

### **Video Processing**
- **Playback Controls**: Play, pause, seek, frame-by-frame
- **Session Selection**: Choose from processed tracking sessions
- **Data Visualization**: View tracking statistics and plots

## 🐛 Troubleshooting

**Real-time server won't start?**

- Ensure Python dependencies are installed: `pip install websockets opencv-python ultralytics`
- Check that no other process is using port 8765

**Camera not working?**

- Grant browser camera permissions
- Try different camera IDs in the interface

**Video file not found?**

- Place videos in `data/` directory
- Use relative paths like `data/your_video.mp4`

**Performance issues?**

- Close other resource-intensive applications
- Lower video resolution for better FPS

**Database connection issues?**

- Ensure SQLite database exists in `databases/` directory
- Check file permissions

---

## 📚 Additional Resources

- **📖 Real-time System Guide**: `standalone-frontend/REALTIME_GUIDE.md`
- **🎯 Tooltip Demo**: `standalone-frontend/TOOLTIP_DEMO.md`
- **🚀 Quick Start**: `standalone-frontend/QUICKSTART.md`
- **📹 Camera Selection**: `standalone-frontend/CAMERA_SELECTION.md`
- **⚡ Features Overview**: `standalone-frontend/FEATURES.md`

## 🔧 Development

### **Adding New Features**

1. **Backend**: Add new scripts in `backend/tracking/` or `backend/db_query/`
2. **Frontend**: Modify React components in `standalone-frontend/src/`
3. **Makefile**: Add new commands to `Makefile` for easy access
4. **Database**: Use `database_integration.py` for data storage

### **Testing**

```bash
# Test basic tracking
make analyse

# Test database visualization
make db_visualize

# Test frontend
make frontend
```

---

## 📄 License

This project is developed for the European Defence Tech Hackathon London 2025.
Developed by Juan Carlos Climent Pardo, Alvaro Ritter, Kenneth Oliver, Kazybek Khairulla, and Lysander Mawby.
