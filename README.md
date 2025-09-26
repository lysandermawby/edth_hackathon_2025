# EDTH Hackathon London 2025

Real-time object detection and tracking system for the European Defence Tech Hackathon in London 2025.

See [this information spreadsheet](https://docs.google.com/spreadsheets/d/1AT8ndsEe9hgljTUFgmpAJ_pYE9kLxrVdJI-xuT68uto/edit?referrer=luma&gid=2054763765#gid=2054763765).
Data is available through [this sharepoint](https://quantumdrones-my.sharepoint.com/personal/pzimmermann_quantum-systems_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpzimmermann%5Fquantum%2Dsystems%5Fcom%2FDocuments%2F2025%5FLondon%5FHackathon&ga=1).

## 🎯 Project Features

**Complete real-time object detection and tracking pipeline** tackling Quantum-6 (and Stark-1):

✅ **Real-time object detection** using YOLO11  
✅ **Multi-object tracking** with ByteTrack (persistent IDs)  
✅ **Live web interface** with interactive visualization  
✅ **Camera and video file support**  
✅ **Database integration** for session history  
✅ **Interactive tooltips** with detection details  
✅ **Performance monitoring** (FPS, object count)  

### Core Capabilities

1. **Identify objects in video in real time** - YOLO11 detection with 30+ FPS
2. **Estimate velocity, acceleration, and class** - Multi-object tracking with ByteTrack
3. **Maintain live tracking** - Persistent object IDs across frames
4. **Interactive web visualization** - Real-time canvas with detection overlays

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

### Process Video Files

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

### Original Tauri Frontend

**Launch the original desktop app:**
```bash
cd frontend
npm install
npm run tauri dev
```
Use **Preview** to view processed videos with tracking overlays.

---

## 🏗️ System Architecture

```
📹 Input (Camera/Video) 
    ↓
🤖 YOLO11 Detection 
    ↓
🎯 ByteTrack Multi-Object Tracking
    ↓
🌐 WebSocket Streaming
    ↓
⚛️ React Frontend Canvas
    ↓
🗄️ SQLite Database Storage
```

## 📋 Project Structure

```
├── backend/tracking/          # Python detection & tracking
│   ├── integrated_realtime_tracker.py
│   ├── trackers_in_video.py
│   └── realtime_tracker.py
├── standalone-frontend/       # 🔴 Real-time web interface
│   ├── start-realtime.sh     # One-command startup
│   ├── realtime_server.py    # WebSocket detection server
│   ├── src/RealtimeVideoCanvas.tsx
│   └── server.js             # API for recorded sessions
├── frontend/                  # Original Tauri desktop app
├── data/                      # Video files
├── databases/                 # SQLite tracking data
└── models/                    # YOLO model weights
```

## ⚡ Key Features

- **🎯 Real-time Object Detection**: YOLO11 at 30+ FPS
- **🔄 Multi-Object Tracking**: ByteTrack with persistent IDs
- **🎨 Interactive Visualization**: Hover tooltips with detection details
- **📊 Live Performance Metrics**: FPS, object count, frame numbers
- **🗄️ Session Management**: Database storage and replay
- **📹 Flexible Input**: Camera feed or video files
- **🌐 Web-based Interface**: No installation required

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

---

## 📚 Additional Resources

- **📖 Real-time System Guide**: `standalone-frontend/REALTIME_GUIDE.md`
- **🎯 Tooltip Demo**: `standalone-frontend/TOOLTIP_DEMO.md`
- **🚀 Quick Start**: `standalone-frontend/QUICKSTART.md`
