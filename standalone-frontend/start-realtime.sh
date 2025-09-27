#!/bin/bash

echo "🚀 Starting EDTH Real-time Object Tracking System"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "realtime_server.py" ]; then
    echo "❌ Please run this script from the standalone-frontend directory"
    exit 1
fi

# Kill any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "realtime_server.py" 2>/dev/null || true
pkill -f "node server.js" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true

# Check Python dependencies using Poetry
echo "🔧 Checking Python dependencies..."
cd ../backend
poetry run python -c "import websockets, cv2, ultralytics" 2>/dev/null || {
    echo "❌ Missing Python dependencies. Installing with Poetry..."
    poetry install
}
cd ../standalone-frontend

# Start the WebSocket real-time detection server
echo "🎯 Starting real-time detection server on ws://localhost:8765..."
cd ../backend
poetry run python ../standalone-frontend/realtime_server.py &
REALTIME_PID=$!
cd ../standalone-frontend

# Wait for real-time server to start
sleep 3

# Start the API server for recorded sessions
echo "🔧 Starting API server on http://localhost:3001..."
node server.js &
API_PID=$!

# Wait for API server to start
sleep 2

# Test if API server is working
if curl -s http://localhost:3001/api/health > /dev/null; then
    echo "✅ API server started successfully"
else
    echo "❌ API server failed to start"
    kill $REALTIME_PID $API_PID 2>/dev/null
    exit 1
fi

echo "🌐 Starting frontend development server on http://localhost:5173..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "🎉 All systems ready!"
echo ""
echo "📊 API Server (recorded sessions): http://localhost:3001"
echo "🎯 Real-time Detection Server: ws://localhost:8765"
echo "🖥️  Frontend Interface: http://localhost:5173"
echo ""
echo "💡 How to use:"
echo "   1. Open http://localhost:5173 in your browser"
echo "   2. Toggle between 'Recorded Sessions' and 'Live Detection'"
echo "   3. For live detection:"
echo "      - Click 'Connect' to connect to real-time server"
echo "      - Click 'Start Webcam' for live camera feed"
echo "      - Click 'Start Sample Video' for video file processing"
echo "   4. Hover over detections for detailed information"
echo ""
echo "🛑 Press Ctrl+C to stop all servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    kill $REALTIME_PID $API_PID $FRONTEND_PID 2>/dev/null
    echo "✅ Stopped all servers"
    exit 0
}

# Trap the cleanup function
trap cleanup EXIT INT TERM

# Wait for user to press Ctrl+C
wait
