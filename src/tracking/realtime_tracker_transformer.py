import cv2
import torch
import supervision as sv
from trackers import SORTTracker
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
import numpy as np
import time

class RealtimeTrackerTransformer:
    def __init__(self, model_name="PekingU/rtdetr_v2_r18vd", camera_id=0):
        """
        Initialize the real-time tracker with RT-DETR transformer model
        
        Args:
            model_name (str): Hugging Face model name for RT-DETR
            camera_id (int): Camera device ID (usually 0 for default camera)
        """
        self.tracker = SORTTracker()
        self.image_processor = RTDetrImageProcessor.from_pretrained(model_name)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(model_name)
        self.annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
        self.box_annotator = sv.BoxAnnotator()
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        
        # Set model to evaluation mode
        self.model.eval()
        
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
        Process a single frame for detection and tracking using RT-DETR
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Annotated frame with detections and tracking IDs
        """
        # Preprocess frame for RT-DETR
        inputs = self.image_processor(images=frame, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        h, w, _ = frame.shape
        results = self.image_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(h, w)]),
            threshold=0.5
        )[0]
        
        # Convert to supervision detections
        detections = sv.Detections.from_transformers(
            transformers_results=results,
            id2label=self.model.config.id2label
        )
        
        # Update tracker
        detections = self.tracker.update(detections)
        
        # Create labels with tracker IDs and class names
        labels = []
        for i in range(len(detections)):
            class_name = detections.data.get('class_name', ['Unknown'])[i] if detections.data else 'Unknown'
            if detections.tracker_id[i] is not None:
                labels.append(f"ID: {detections.tracker_id[i]} - {class_name}")
            else:
                labels.append(f"New - {class_name}")
        
        # Annotate frame
        annotated_frame = self.box_annotator.annotate(frame, detections)
        annotated_frame = self.annotator.annotate(annotated_frame, detections, labels=labels)
        
        return annotated_frame, detections
    
    def run(self):
        """Main loop for real-time tracking"""
        try:
            self.initialize_camera()
            self.running = True
            
            print("\n=== Real-time Object Tracking (RT-DETR) ===")
            print("Press 'q' to quit")
            print("Press 's' to save current frame")
            print("Press 'r' to reset tracker")
            print("Press SPACE to pause/resume")
            print("Press 't' to toggle detection threshold")
            print("==========================================\n")
            
            paused = False
            frame_count = 0
            start_time = time.time()
            detection_threshold = 0.5
            
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
                        cv2.putText(annotated_frame, f"Threshold: {detection_threshold:.2f}", 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Display frame
                    cv2.imshow('Real-time Object Tracking (RT-DETR)', annotated_frame)
                else:
                    # Show paused message
                    paused_frame = frame.copy() if 'frame' in locals() else np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(paused_frame, "PAUSED - Press SPACE to resume", 
                              (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Real-time Object Tracking (RT-DETR)', paused_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    if not paused and 'annotated_frame' in locals():
                        filename = f"captured_frame_transformer_{int(time.time())}.jpg"
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
                elif key == ord('t'):
                    # Toggle detection threshold
                    detection_threshold = 0.3 if detection_threshold == 0.5 else 0.5
                    print(f"Detection threshold changed to {detection_threshold}")
                    
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
    """Main function to run the real-time tracker with transformer model"""
    # You can change the camera_id if you have multiple cameras
    camera_id = 0
    
    # Available RT-DETR models:
    # "PekingU/rtdetr_v2_r18vd" - Faster, less accurate
    # "PekingU/rtdetr_v2_r50vd" - Balanced
    # "PekingU/rtdetr_v2_r101vd" - Slower, more accurate
    model_name = "PekingU/rtdetr_v2_r18vd"
    
    tracker = RealtimeTrackerTransformer(model_name=model_name, camera_id=camera_id)
    tracker.run()

if __name__ == "__main__":
    main()
