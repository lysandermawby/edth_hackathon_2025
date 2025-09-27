#!/bin/bash

echo "🚀 Starting EDTH Object Tracker Standalone Frontend"
echo "=================================================="

# Change to the standalone frontend directory
cd standalone-frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Kill any existing processes
echo "🧹 Cleaning up any existing processes..."
pkill -f "node server.js" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true

# Check Python dependencies using Poetry
echo "🔧 Checking Python dependencies..."
cd backend
poetry run python -c "import sqlite3, cv2" 2>/dev/null || {
    echo "❌ Missing Python dependencies. Installing with Poetry..."
    poetry install
}
cd standalone-frontend

echo "🔧 Starting API server on port 3001..."
node server.js &
API_PID=$!

# Wait for API server to start
sleep 3

# Test if API server is working
if curl -s http://localhost:3001/api/health > /dev/null; then
    echo "✅ API server started successfully"
else
    echo "❌ API server failed to start"
    exit 1
fi

echo "🌐 Starting frontend development server on port 5173..."

# Start the frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "🎉 Application is starting up!"
echo ""
echo "📊 API Server: http://localhost:3001"
echo "🖥️  Frontend:   http://localhost:5173"
echo ""
echo "📝 Available API endpoints:"
echo "   GET /api/sessions           - List all tracking sessions"
echo "   GET /api/sessions/:id/detections - Get detections for a session"
echo "   GET /api/video/:path        - Serve video files"
echo "   GET /api/health            - Health check"
echo ""
echo "💡 To use:"
echo "   1. Open http://localhost:5173 in your browser"
echo "   2. Select a tracking session from the left panel"
echo "   3. Watch the video with detection overlays!"
echo ""
echo "🛑 Press Ctrl+C to stop both servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    kill $API_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Stopped all servers"
    exit 0
}

# Trap the cleanup function
trap cleanup EXIT INT TERM

# Wait for user to press Ctrl+C
wait
