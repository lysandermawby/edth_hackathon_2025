#!/usr/bin/env python3
"""
Real-time object detection and tracking server with WebSocket streaming.

Processes video frames (from file or webcam) with YOLO detection and ByteTrack,
then streams the results to the frontend via WebSocket.
"""

import asyncio
import websockets
import json
import cv2
import numpy as np
import base64
import time
import threading
from pathlib import Path
import sys
import os

# Add the backend tracking module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'tracking'))

try:
    from integrated_realtime_tracker import IntegratedRealtimeTracker
    from database_integration import TrackingDatabase, RealTimeDataProcessor
except ImportError as e:
    print(f"Error importing tracking modules: {e}")
    print("Please ensure you're in the project root and have installed dependencies")
    sys.exit(1)

# Import depth estimation
try:
    from depth_estimator import create_depth_estimator
    DEPTH_AVAILABLE = True
except ImportError as e:
    print(f"Depth estimation not available: {e}")
    print("Depth Pro functionality will be disabled")
    DEPTH_AVAILABLE = False

class RealtimeDetectionServer:
    def __init__(self, model_path=None, db_path=None, enable_depth=True):
        """Initialize the real-time detection server"""
        # Set up paths relative to project root
        project_root = Path(__file__).parent.parent

        if model_path is None:
            model_path = project_root / "models" / "yolo11m.pt"
        if db_path is None:
            db_path = project_root / "databases" / "tracking_data.db"

        # Initialize tracker
        self.tracker = IntegratedRealtimeTracker(
            model_path=str(model_path),
            db_path=str(db_path),
            headless=True,
            enable_database=True
        )

        # Initialize depth estimator if available and enabled
        self.depth_estimator = None
        if DEPTH_AVAILABLE and enable_depth:
            try:
                self.depth_estimator = create_depth_estimator()
                if self.depth_estimator.is_available():
                    print("Depth estimation enabled using Apple Depth Pro")
                else:
                    print("Depth Pro model not available - depth estimation disabled")
                    self.depth_estimator = None
            except Exception as e:
                print(f"Failed to initialize depth estimator: {e}")
                self.depth_estimator = None
        else:
            print("Depth estimation disabled")
        
        # WebSocket connections
        self.clients = set()
        
        # Video capture
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.start_time = None
        
        # Performance tracking
        self.fps_tracker = []
        
    async def register_client(self, websocket):
        """Register a new WebSocket client"""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial status
        await websocket.send(json.dumps({
            "type": "status",
            "message": "Connected to detection server",
            "clients": len(self.clients)
        }))
        
    async def unregister_client(self, websocket):
        """Unregister a WebSocket client"""
        self.clients.discard(websocket)
        print(f"Client disconnected. Total clients: {len(self.clients)}")
        
    async def broadcast(self, message):
        """Broadcast message to all connected clients"""
        if self.clients:
            # Remove disconnected clients
            disconnected = set()
            for client in self.clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            
            # Clean up disconnected clients
            self.clients -= disconnected
            
    def frame_to_base64(self, frame):
        """Convert OpenCV frame to base64 string"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64
        
    async def start_video_processing(self, source=0):
        """Start processing video from source (camera ID or video file path)"""
        try:
            # Initialize video capture
            if isinstance(source, str) and source.isdigit():
                source = int(source)
                
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                await self.broadcast(json.dumps({
                    "type": "error",
                    "message": f"Could not open video source: {source}"
                }))
                return
                
            # Set camera properties if using webcam
            if isinstance(source, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            await self.broadcast(json.dumps({
                "type": "video_info",
                "fps": fps,
                "width": width,
                "height": height,
                "source": str(source)
            }))
            
            self.running = True
            self.frame_count = 0
            self.start_time = time.time()
            
            # Start database session
            source_path = str(source) if isinstance(source, int) else source
            session_id = self.tracker.data_processor.db.start_session(f"realtime_{source_path}", fps)
            self.tracker.data_processor.start_processing(f"realtime_{source_path}", fps, session_id)
            
            print(f"Started video processing from source: {source}")
            print(f"Resolution: {width}x{height}, FPS: {fps}")
            
            # Main processing loop
            while self.running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    if isinstance(source, str):  # Video file ended
                        await self.broadcast(json.dumps({
                            "type": "video_ended",
                            "message": "Video file ended"
                        }))
                        break
                    else:  # Camera error
                        continue
                        
                # Process frame for detection and tracking
                frame_start_time = time.time()
                current_timestamp = time.time() - self.start_time
                
                try:
                    annotated_frame, detections, tracking_data = self.tracker.process_frame(frame, current_timestamp)

                    # Process depth estimation if available
                    depth_map = None
                    enhanced_objects = []

                    if self.depth_estimator is not None:
                        try:
                            frame_id = f"frame_{self.frame_count}"
                            depth_map, enhanced_detections = self.depth_estimator.process_frame_with_detections(
                                frame, tracking_data['objects'], frame_id
                            )
                            enhanced_objects = enhanced_detections
                        except Exception as e:
                            print(f"Depth estimation error: {e}")
                            enhanced_objects = tracking_data['objects']
                    else:
                        enhanced_objects = tracking_data['objects']

                    # Calculate FPS
                    processing_time = time.time() - frame_start_time
                    current_fps = 1.0 / processing_time if processing_time > 0 else 0
                    self.fps_tracker.append(current_fps)
                    if len(self.fps_tracker) > 30:  # Keep last 30 frames for average
                        self.fps_tracker.pop(0)
                    avg_fps = sum(self.fps_tracker) / len(self.fps_tracker)

                    # Add depth indicator to info overlay if available
                    depth_status = "Depth: ON" if self.depth_estimator is not None else "Depth: OFF"
                    info_text = f"Session: {session_id} | {depth_status}"

                    # Add info overlay
                    info_frame = self.tracker.add_info_overlay(
                        annotated_frame.copy(),
                        avg_fps,
                        len(enhanced_objects),
                        info_text
                    )

                    # Convert frame to base64
                    frame_base64 = self.frame_to_base64(info_frame)

                    # Prepare detection data for frontend
                    detection_objects = []
                    for obj in enhanced_objects:
                        detection_obj = {
                            "tracker_id": obj.get('tracker_id'),
                            "class_id": obj['class_id'],
                            "class_name": obj['class_name'],
                            "confidence": obj['confidence'],
                            "bbox": obj['bbox'],
                            "center": obj['center']
                        }

                        # Add depth information if available
                        if 'depth' in obj:
                            detection_obj['depth'] = obj['depth']

                        detection_objects.append(detection_obj)
                    
                    # Send frame and detection data to clients
                    message = {
                        "type": "frame",
                        "frame_number": self.frame_count,
                        "timestamp": current_timestamp,
                        "frame_data": frame_base64,
                        "detections": detection_objects,
                        "fps": avg_fps,
                        "session_id": session_id
                    }
                    
                    await self.broadcast(json.dumps(message))
                    
                    self.frame_count += 1
                    
                    # Control frame rate to prevent overwhelming clients
                    await asyncio.sleep(0.033)  # ~30 FPS max
                    
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    await self.broadcast(json.dumps({
                        "type": "error",
                        "message": f"Frame processing error: {str(e)}"
                    }))
                    
        except Exception as e:
            print(f"Video processing error: {e}")
            await self.broadcast(json.dumps({
                "type": "error",
                "message": f"Video processing error: {str(e)}"
            }))
        finally:
            await self.stop_processing()
            
    async def stop_processing(self):
        """Stop video processing"""
        self.running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # Stop database processing
        if hasattr(self.tracker, 'data_processor') and self.tracker.data_processor:
            self.tracker.data_processor.stop_processing(self.frame_count)
            
        await self.broadcast(json.dumps({
            "type": "status",
            "message": "Video processing stopped"
        }))
        
        print("Video processing stopped")
        
    def detect_available_cameras(self, max_cameras=10):
        """Detect available cameras by trying to open them"""
        available_cameras = []
        
        for camera_id in range(max_cameras):
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                # Test if we can actually read a frame
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(camera_id)
                cap.release()
            else:
                # If we can't open this camera, assume no more cameras exist
                break
                
        return available_cameras

    async def handle_client_message(self, websocket, message):
        """Handle incoming messages from clients"""
        try:
            data = json.loads(message)
            command = data.get('command')
            
            if command == 'list_cameras':
                cameras = self.detect_available_cameras()
                await websocket.send(json.dumps({
                    "type": "camera_list",
                    "cameras": cameras,
                    "message": f"Found {len(cameras)} available cameras"
                }))
                
            elif command == 'start_camera':
                camera_id = data.get('camera_id', 0)
                await self.start_video_processing(camera_id)
                
            elif command == 'start_video':
                video_path = data.get('video_path')
                if video_path:
                    # Convert relative path to absolute
                    if not os.path.isabs(video_path):
                        project_root = Path(__file__).parent.parent
                        video_path = project_root / video_path
                    await self.start_video_processing(str(video_path))
                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "No video path provided"
                    }))
                    
            elif command == 'stop':
                await self.stop_processing()
                
            elif command == 'ping':
                await websocket.send(json.dumps({
                    "type": "pong",
                    "timestamp": time.time()
                }))
                
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Invalid JSON message"
            }))
        except Exception as e:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Error handling message: {str(e)}"
            }))
            
    async def handle_client(self, websocket, path):
        """Handle new client connection"""
        await self.register_client(websocket)
        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)

# Global server instance
server = RealtimeDetectionServer()

async def main():
    """Main function to start the WebSocket server"""
    print("Starting Real-time Object Detection Server...")
    print("WebSocket server will run on ws://localhost:8765")
    
    # Start WebSocket server
    start_server = websockets.serve(server.handle_client, "localhost", 8765)
    
    print("Server ready! Connect from frontend to start detection.")
    await start_server
    await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Server error: {e}")
