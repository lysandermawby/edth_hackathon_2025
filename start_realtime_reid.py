#!/usr/bin/env python3
"""
Quick start script for the real-time re-identification system.

This script launches both the backend re-identification tracker server
and the frontend in the correct order.
"""

import subprocess
import time
import sys
import os
import signal
from pathlib import Path

def start_backend():
    """Start the realtime server with re-identification"""
    print("🚀 Starting real-time re-identification server...")
    frontend_dir = Path(__file__).parent / "standalone-frontend"
    server_script = frontend_dir / "realtime_server.py"
    
    # Start the backend server
    backend_process = subprocess.Popen([
        sys.executable, str(server_script)
    ], cwd=str(frontend_dir))
    
    return backend_process

def start_frontend():
    """Start the frontend development server"""
    print("🌐 Starting frontend development server...")
    frontend_dir = Path(__file__).parent / "standalone-frontend"
    
    # Start the frontend server
    frontend_process = subprocess.Popen([
        "npm", "run", "dev"
    ], cwd=str(frontend_dir))
    
    return frontend_process

def main():
    """Main function to start both services"""
    print("🎯 EDTH Real-time Re-identification System")
    print("=" * 50)
    
    backend_process = None
    frontend_process = None
    
    try:
        # Start backend first
        backend_process = start_backend()
        print("✅ Backend server started")
        
        # Wait a moment for backend to initialize
        time.sleep(3)
        
        # Start frontend
        frontend_process = start_frontend()
        print("✅ Frontend server started")
        
        print("\n🌟 System ready!")
        print("📱 Frontend: http://localhost:5173")
        print("🔌 WebSocket: ws://localhost:8765")
        print("\nFeatures enabled:")
        print("  • Real-time object detection with YOLO")
        print("  • Enhanced re-identification system")
        print("  • Object tracking through occlusions")
        print("  • Live performance statistics")
        print("  • Database integration")
        print("\nPress Ctrl+C to stop all services")
        
        # Wait for user interruption
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        
        if frontend_process:
            frontend_process.terminate()
            print("✅ Frontend stopped")
            
        if backend_process:
            backend_process.terminate()
            print("✅ Backend stopped")
            
        print("👋 All services stopped successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        if frontend_process:
            frontend_process.terminate()
            
        if backend_process:
            backend_process.terminate()
            
        sys.exit(1)

if __name__ == "__main__":
    main()
