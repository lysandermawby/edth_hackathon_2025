# 🎯 Real-time Object Detection & Tracking System

## 🚀 Complete Pipeline Overview

This system provides **end-to-end real-time object detection and tracking** from camera feed or video file to interactive web visualization.

## 📋 System Architecture

```
📹 Video Input → 🤖 YOLO Detection → 🎯 ByteTrack → 🌐 WebSocket → ⚛️ React Canvas
    (Camera/File)     (AI Processing)     (Tracking)    (Streaming)   (Live Visualization)
```

## 🔧 Components

### 🐍 **Backend: Real-time Detection Server**

- **File**: `realtime_server.py`
- **Port**: `ws://localhost:8765` (WebSocket)
- **Features**:
  - YOLO11 object detection
  - ByteTrack multi-object tracking
  - WebSocket streaming to frontend
  - SQLite database integration
  - Support for webcam and video files
  - Base64 frame encoding for web delivery

### ⚛️ **Frontend: Real-time Canvas Component**

- **File**: `RealtimeVideoCanvas.tsx`
- **Features**:
  - WebSocket client connection
  - Live video stream rendering
  - Real-time detection overlays
  - Interactive hover tooltips
  - Performance monitoring (FPS, object count)
  - Connection status indicators

### 🗄️ **API Server: Session Storage**

- **File**: `server.js`
- **Port**: `http://localhost:3001`
- **Purpose**: Serves recorded session data and video files

## 🚀 Quick Start

### 1. **Start the Complete System**

```bash
./start-realtime.sh
```

This starts all three components:

- Real-time detection server (ws://localhost:8765)
- API server (http://localhost:3001)
- Frontend interface (http://localhost:5173)

### 2. **Use the Interface**

1. Open http://localhost:5173
2. Toggle to **"🔴 Live Detection"** mode
3. Click **"Connect"** to connect to real-time server
4. Choose your input:
   - **"Start Webcam"** for live camera feed
   - **"Start Sample Video"** for video file processing

### 3. **Interact with Detections**

- **Hover** over any detection bounding box for detailed info
- Watch **real-time statistics**: FPS, object count, frame number
- Observe **tracking consistency** as objects move across frames

## 📊 Real-time Data Flow

### Input Sources

- **Webcam**: Default camera (ID 0) at 1280x720, 30 FPS
- **Video File**: Any supported format (MP4, AVI, etc.)

### Processing Pipeline

1. **Frame Capture**: Video frames from input source
2. **YOLO Detection**: AI identifies objects in each frame
3. **ByteTrack Tracking**: Assigns consistent IDs across frames
4. **Data Extraction**: Bounding boxes, classes, confidence scores
5. **WebSocket Streaming**: JSON + base64 frame data to frontend
6. **Database Storage**: Session and detection data saved to SQLite

### Frontend Rendering

1. **WebSocket Reception**: Receives frame and detection data
2. **Canvas Rendering**: Displays video frame on HTML5 canvas
3. **Overlay Drawing**: Draws bounding boxes with labels
4. **Tooltip Interaction**: Shows detailed info on hover
5. **Performance Display**: Live FPS and statistics

## 🎯 Detection Information

Each detection provides:

- **Class**: Object type (car, person, bike, etc.)
- **Tracker ID**: Consistent ID across frames for tracking
- **Confidence**: AI certainty percentage (0-100%)
- **Bounding Box**: Position and size (x, y, width, height)
- **Center Point**: Object center coordinates

## ⚡ Performance Features

### Optimization

- **30 FPS Rate Limiting**: Prevents frontend overwhelm
- **JPEG Compression**: Efficient frame streaming
- **Async Processing**: Non-blocking WebSocket handling
- **Memory Management**: Automatic cleanup of old data

### Monitoring

- **Live FPS Counter**: Real processing speed
- **Object Count**: Number of detections per frame
- **Frame Number**: Processing progress tracking
- **Connection Status**: WebSocket health indicator

## 🔧 Advanced Usage

### Custom Video Processing

```javascript
// Connect to WebSocket and start custom video
const ws = new WebSocket("ws://localhost:8765");
ws.send(
  JSON.stringify({
    command: "start_video",
    video_path: "path/to/your/video.mp4",
  })
);
```

### Camera Selection

```javascript
// Start specific camera (ID 1 instead of default 0)
ws.send(
  JSON.stringify({
    command: "start_camera",
    camera_id: 1,
  })
);
```

## 🐛 Troubleshooting

### WebSocket Connection Issues

- Ensure Python server is running on port 8765
- Check browser console for connection errors
- Verify no firewall blocking WebSocket connections

### Camera Access Problems

- Grant browser camera permissions
- Check if other apps are using the camera
- Try different camera IDs (0, 1, 2...)

### Performance Issues

- Lower video resolution for better FPS
- Close other resource-intensive applications
- Check GPU availability for YOLO processing

## 🎮 Demo Scenarios

### 1. **Live Object Tracking**

- Start webcam feed
- Move objects in view
- Watch consistent tracker IDs
- Observe real-time performance metrics

### 2. **Video Analysis**

- Load sample video with multiple objects
- Observe detection accuracy across frames
- Check tracking consistency for moving objects
- Analyze confidence scores for different object types

### 3. **Interactive Inspection**

- Hover over different detections
- Compare confidence scores
- Track object movement via center coordinates
- Monitor size changes as objects move closer/farther

This system demonstrates the complete pipeline from raw video input to interactive web visualization with real-time AI processing! 🎉
