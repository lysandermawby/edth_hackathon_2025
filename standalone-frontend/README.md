# EDTH Object Tracker - Standalone Frontend

A standalone React TypeScript frontend for viewing video tracking data with detection overlays.

## Features

- 🎥 Video playback with real-time detection overlays
- 🎯 Bounding boxes with class labels and confidence scores
- 📊 Session-based tracking data from SQLite database
- 🎨 Modern, responsive UI with Tailwind CSS
- ⚡ Fast development with Vite

## Setup

1. Install dependencies:

```bash
npm install
```

2. Make sure you have tracking data in the database:

```bash
# From the project root, run the tracking script
cd ../
python backend/tracking/integrated_realtime_tracker.py data/Individual_2.mp4 --headless
```

## Running

### Quick Start

```bash
./start.sh
```

This will start both the API server (port 3001) and the frontend dev server (port 5173).

### Manual Start

```bash
# Terminal 1: Start API server
npm run server

# Terminal 2: Start frontend
npm run dev
```

Then open http://localhost:5173 in your browser.

## API Endpoints

- `GET /api/sessions` - Get all tracking sessions
- `GET /api/sessions/:id/detections` - Get detections for a session
- `GET /api/video/:path` - Serve video files
- `GET /api/health` - Health check

## Project Structure

```
src/
├── App.tsx              # Main application component
├── VideoCanvas.tsx      # Video player with detection overlay
├── types.ts            # TypeScript interfaces
├── main.tsx            # React entry point
└── index.css           # Tailwind CSS
```

## Usage

1. Select a tracking session from the left panel
2. The video will load with detection overlays
3. Use the video controls to play/pause and seek
4. Bounding boxes show detected objects with class labels and confidence scores

## Data Flow

1. Backend serves tracking sessions from SQLite database
2. Frontend loads session data and converts to frame-based format
3. Video canvas synchronizes detection overlays with video playback
4. Detections are rendered in real-time as the video plays
