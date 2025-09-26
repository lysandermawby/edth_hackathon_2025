import cv2
import supervision as sv
from trackers import SORTTracker
from ultralytics import YOLO
import numpy as np
import time

class RealtimeTracker:
    def __init__(self, model_path="yolo11m.pt", camera_id=0):
        """
        Initialize the real-time tracker
        
        Args:
            model_path (str): Path to YOLO model weights
            camera_id (int): Camera device ID (usually 0 for default camera)
        """
        self.tracker = SORTTracker()
        self.model = YOLO(model_path)
        self.annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
        self.box_annotator = sv.BoxAnnotator()
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        
    def initialize_camera(self):
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera initialized successfully")
        print(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
        
    def process_frame(self, frame):
        """
        Process a single frame for detection and tracking
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Annotated frame with detections and tracking IDs
        """
        # Run YOLO detection
        result = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Update tracker
        detections = self.tracker.update(detections)
        
        # Create labels with tracker IDs
        labels = []
        for i in range(len(detections)):
            if detections.tracker_id[i] is not None:
                labels.append(f"ID: {detections.tracker_id[i]}")
            else:
                labels.append("New")
        
        # Annotate frame
        annotated_frame = self.box_annotator.annotate(frame, detections)
        annotated_frame = self.annotator.annotate(annotated_frame, detections, labels=labels)
        
        return annotated_frame, detections
    
    def run(self):
        """Main loop for real-time tracking"""
        try:
            self.initialize_camera()
            self.running = True
            
            print("\n=== Real-time Object Tracking ===")
            print("Press 'q' to quit")
            print("Press 's' to save current frame")
            print("Press 'r' to reset tracker")
            print("Press SPACE to pause/resume")
            print("================================\n")
            
            paused = False
            frame_count = 0
            start_time = time.time()
            
            while self.running:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to read frame from camera")
                        break
                    
                    # Process frame
                    annotated_frame, detections = self.process_frame(frame)
                    
                    # Calculate FPS
                    frame_count += 1
                    if frame_count % 30 == 0:  # Update FPS every 30 frames
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time
                        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display frame
                    cv2.imshow('Real-time Object Tracking', annotated_frame)
                else:
                    # Show paused message
                    paused_frame = frame.copy() if 'frame' in locals() else np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(paused_frame, "PAUSED - Press SPACE to resume", 
                              (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Real-time Object Tracking', paused_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    if not paused and 'annotated_frame' in locals():
                        filename = f"captured_frame_{int(time.time())}.jpg"
                        cv2.imwrite(filename, annotated_frame)
                        print(f"Frame saved as {filename}")
                elif key == ord('r'):
                    # Reset tracker
                    self.tracker = SORTTracker()
                    print("Tracker reset")
                elif key == ord(' '):
                    # Toggle pause
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")

def main():
    """Main function to run the real-time tracker"""
    camera_id = 0
    
    # Available models: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    model_path = "yolo11m.pt"
    
    tracker = RealtimeTracker(model_path=model_path, camera_id=camera_id)
    tracker.run()

if __name__ == "__main__":
    main()
