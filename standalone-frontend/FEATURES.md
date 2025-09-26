# EDTH Object Tracker - Feature Overview

## 🎯 Core Features

### Real-time Video Playback with Detection Overlays

- **Video Canvas**: Custom HTML5 canvas rendering for optimal performance
- **Synchronized Overlays**: Detection bounding boxes perfectly synced with video frames
- **Color-coded Objects**: Consistent colors per tracker ID for easy following
- **Class Labels**: Shows object class (car, bus, airplane, etc.) with confidence scores
- **Tracker IDs**: Unique identifier for each tracked object across frames

### Database Integration

- **SQLite Backend**: Fast, reliable storage of tracking data
- **Session Management**: Multiple tracking sessions with metadata
- **Frame-based Storage**: Efficient storage of detection data per video frame
- **Real-time Queries**: Fast retrieval of detection data for visualization

### User Interface

- **Session Selection**: Browse and select from available tracking sessions
- **Video Controls**: Play, pause, seek through video with standard controls
- **Detection Statistics**: Real-time count of objects and frames with detections
- **Responsive Design**: Works on desktop and tablet devices
- **Modern UI**: Built with Tailwind CSS for clean, professional appearance

## 🔧 Technical Features

### Frontend (React TypeScript)

- **Vite Build System**: Fast development and production builds
- **TypeScript**: Type-safe development with excellent IDE support
- **Component Architecture**: Modular, reusable components
- **Performance Optimized**: Efficient canvas rendering and data management

### Backend (Node.js Express)

- **RESTful API**: Clean, standard API endpoints
- **CORS Enabled**: Cross-origin requests supported
- **Static File Serving**: Direct video file serving with proper headers
- **SQLite Integration**: Native database connectivity

### Video Processing Pipeline

- **YOLO Detection**: Advanced object detection using YOLOv11
- **ByteTrack Tracking**: State-of-the-art multi-object tracking
- **Real-time Processing**: Live detection and tracking during video playback
- **Multiple Object Classes**: Supports cars, buses, airplanes, and more

## 📊 Data Flow

1. **Video Processing**: YOLO detects objects → ByteTrack assigns IDs → Data saved to SQLite
2. **API Layer**: Express server serves tracking sessions and detection data
3. **Frontend Rendering**: React app loads data → Canvas renders video + overlays
4. **User Interaction**: Video controls → Real-time overlay updates

## 🎨 Visual Features

### Detection Overlays

- **Bounding Boxes**: Precise rectangular boundaries around detected objects
- **Color Consistency**: Same color per tracker ID throughout video
- **Labels**: Class name, tracker ID, and confidence percentage
- **Smooth Animation**: 60fps canvas rendering for fluid experience

### UI Components

- **Session List**: Organized view of all tracking sessions with metadata
- **Video Player**: Full-featured player with timeline scrubbing
- **Statistics Panel**: Live updates of detection counts and frame info
- **Status Indicators**: Loading states and error handling

## 🚀 Performance

- **Efficient Rendering**: Canvas-based rendering for optimal performance
- **Lazy Loading**: Data loaded only when needed
- **Memory Management**: Efficient handling of large video files
- **Fast Queries**: Indexed database queries for quick data retrieval

## 🔮 Extensibility

The system is designed to be easily extended:

- **New Object Classes**: Add support for additional YOLO classes
- **Custom Visualizations**: Extend canvas rendering for new overlay types
- **Additional Metadata**: Store and display more tracking information
- **Export Features**: Add video export with burned-in overlays
- **Real-time Streaming**: Extend to support live camera feeds
