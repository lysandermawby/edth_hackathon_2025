#!/usr/bin/env python3
"""
Database integration for real-time tracking data storage and retrieval.

This module provides examples of how to store tracking data in various databases
for real-time processing and decision making.
"""

import sqlite3
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional
import queue

class TrackingDatabase:
    """SQLite database for storing tracking data"""
    
    def __init__(self, db_path="tracking_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tracking sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracking_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                total_frames INTEGER,
                fps REAL
            )
        ''')
        
        # Create tracked objects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracked_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                frame_number INTEGER,
                timestamp REAL,
                tracker_id INTEGER,
                class_id INTEGER,
                class_name TEXT,
                confidence REAL,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                center_x REAL,
                center_y REAL,
                FOREIGN KEY (session_id) REFERENCES tracking_sessions (session_id)
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_frame ON tracked_objects (session_id, frame_number)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracker_id ON tracked_objects (tracker_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_class_name ON tracked_objects (class_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON tracked_objects (timestamp)')
        
        conn.commit()
        conn.close()
    
    def start_session(self, video_path: str, fps: float) -> int:
        """Start a new tracking session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert absolute path to relative path from project root for storage
        relative_path = self._get_relative_path(video_path)
        
        cursor.execute('''
            INSERT INTO tracking_sessions (video_path, fps)
            VALUES (?, ?)
        ''', (relative_path, fps))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return session_id
    
    def _get_relative_path(self, video_path: str) -> str:
        """Convert absolute path to relative path from project root"""
        import os
        
        # Get the project root directory (assuming this file is in backend/tracking/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        # Convert to absolute path first
        abs_path = os.path.abspath(video_path)
        
        # Try to make it relative to project root
        try:
            relative_path = os.path.relpath(abs_path, project_root)
            # Ensure it uses forward slashes for consistency
            return relative_path.replace(os.sep, '/')
        except ValueError:
            # If we can't make it relative (different drives on Windows), return as-is
            return video_path
    
>>>>>>> main
    def end_session(self, session_id: int, total_frames: int):
        """End a tracking session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE tracking_sessions 
            SET end_time = CURRENT_TIMESTAMP, total_frames = ?
            WHERE session_id = ?
        ''', (total_frames, session_id))
        
        conn.commit()
        conn.close()
    
    def insert_frame_data(self, session_id: int, frame_data: Dict[str, Any]):
        """Insert tracking data for a single frame"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for obj in frame_data['objects']:
            cursor.execute('''
                INSERT INTO tracked_objects (
                    session_id, frame_number, timestamp, tracker_id, class_id, class_name,
                    confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, center_x, center_y
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id, frame_data['frame_number'], frame_data['timestamp'],
                obj['tracker_id'], obj['class_id'], obj['class_name'], obj['confidence'],
                obj['bbox']['x1'], obj['bbox']['y1'], obj['bbox']['x2'], obj['bbox']['y2'],
                obj['center']['x'], obj['center']['y']
            ))
        
        conn.commit()
        conn.close()
    
    def get_objects_in_frame(self, session_id: int, frame_number: int) -> List[Dict]:
        """Get all objects tracked in a specific frame"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM tracked_objects 
            WHERE session_id = ? AND frame_number = ?
            ORDER BY tracker_id
        ''', (session_id, frame_number))
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_object_trajectory(self, session_id: int, tracker_id: int) -> List[Dict]:
        """Get the complete trajectory of a tracked object"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM tracked_objects 
            WHERE session_id = ? AND tracker_id = ?
            ORDER BY frame_number
        ''', (session_id, tracker_id))
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_objects_by_class(self, session_id: int, class_name: str) -> List[Dict]:
        """Get all objects of a specific class"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM tracked_objects 
            WHERE session_id = ? AND class_name = ?
            ORDER BY frame_number, tracker_id
        ''', (session_id, class_name))
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_session_summary(self, session_id: int) -> Dict:
        """Get summary statistics for a tracking session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get basic session info
        cursor.execute('SELECT * FROM tracking_sessions WHERE session_id = ?', (session_id,))
        session = cursor.fetchone()
        
        if not session:
            conn.close()
            return None
        
        # Get object counts by class
        cursor.execute('''
            SELECT class_name, COUNT(DISTINCT tracker_id) as unique_objects,
                   COUNT(*) as total_detections
            FROM tracked_objects 
            WHERE session_id = ?
            GROUP BY class_name
        ''', (session_id,))
        
        class_stats = cursor.fetchall()
        
        # Get total unique objects
        cursor.execute('''
            SELECT COUNT(DISTINCT tracker_id) as total_unique_objects
            FROM tracked_objects 
            WHERE session_id = ?
        ''', (session_id,))
        
        total_objects = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'session_id': session[0],
            'video_path': session[1],
            'start_time': session[2],
            'end_time': session[3],
            'total_frames': session[4],
            'fps': session[5],
            'total_unique_objects': total_objects,
            'class_statistics': [{'class': row[0], 'unique_objects': row[1], 'total_detections': row[2]} 
                               for row in class_stats]
        }

class RealTimeDataProcessor:
    """Real-time processor for tracking data"""
    
    def __init__(self, db: TrackingDatabase):
        self.db = db
        self.data_queue = queue.Queue()
        self.running = False
        self.session_id = None
    
    def start_processing(self, video_path: str, fps: float, session_id: Optional[int] = None):
        """Start real-time data processing"""
        if session_id is not None:
            self.session_id = session_id
        else:
            self.session_id = self.db.start_session(video_path, fps)
        
        self.running = True
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_data)
        self.process_thread.start()
    
    def stop_processing(self, total_frames: int):
        """Stop real-time data processing"""
        self.running = False
        if self.process_thread:
            self.process_thread.join()
        
        if self.session_id:
            self.db.end_session(self.session_id, total_frames)
    
    def add_frame_data(self, frame_data: Dict[str, Any]):
        """Add frame data to processing queue"""
        if self.running:
            self.data_queue.put(frame_data)
    
    def _process_data(self):
        """Process data from queue in real-time"""
        while self.running:
            try:
                frame_data = self.data_queue.get(timeout=1.0)
                self.db.insert_frame_data(self.session_id, frame_data)
                
                # Real-time analysis examples
                self._analyze_frame_data(frame_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing data: {e}")
    
    def _analyze_frame_data(self, frame_data: Dict[str, Any]):
        """Perform real-time analysis on frame data"""
        # Example: Alert if too many objects detected
        if len(frame_data['objects']) > 10:
            print(f"ALERT: High object count detected at frame {frame_data['frame_number']}: {len(frame_data['objects'])} objects")
        
        # Example: Track specific object types
        cars = [obj for obj in frame_data['objects'] if obj['class_name'] == 'car']
        if cars:
            print(f"Frame {frame_data['frame_number']}: {len(cars)} cars detected")
        
        # Example: Monitor object movement
        for obj in frame_data['objects']:
            if obj['tracker_id'] is not None:
                # Could implement speed calculation, trajectory analysis, etc.
                pass

# Example usage functions
def example_queries(db: TrackingDatabase, session_id: int):
    """Example database queries"""
    print("\n=== Database Query Examples ===")
    
    # Get session summary
    summary = db.get_session_summary(session_id)
    print(f"Session Summary: {summary}")
    
    # Get objects in frame 100
    frame_objects = db.get_objects_in_frame(session_id, 100)
    print(f"Objects in frame 100: {len(frame_objects)}")
    
    # Get trajectory of object with tracker_id 1
    trajectory = db.get_object_trajectory(session_id, 1)
    print(f"Trajectory of object 1: {len(trajectory)} detections")
    
    # Get all cars
    cars = db.get_objects_by_class(session_id, 'car')
    print(f"All car detections: {len(cars)}")

if __name__ == "__main__":
    # Example usage
    db = TrackingDatabase("example_tracking.db")
    
    # This would typically be called from the real-time video tracker
    session_id = db.start_session("test_video.mp4", 30.0)
    
    # Example frame data
    frame_data = {
        'frame_number': 1,
        'timestamp': 0.033,
        'objects': [
            {
                'tracker_id': 1,
                'class_id': 2,
                'class_name': 'car',
                'confidence': 0.95,
                'bbox': {'x1': 100, 'y1': 200, 'x2': 300, 'y2': 400},
                'center': {'x': 200, 'y': 300}
            }
        ]
    }
    
    db.insert_frame_data(session_id, frame_data)
    db.end_session(session_id, 1000)
    
    # Run example queries
    example_queries(db, session_id)
