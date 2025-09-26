#!/bin/bash

# Start the API server in the background
echo "Starting API server..."
node server.js &
API_PID=$!

# Wait a moment for the API server to start
sleep 2

# Start the Vite dev server
echo "Starting frontend development server..."
npm run dev

# Clean up: kill the API server when the frontend server stops
trap "kill $API_PID" EXIT
