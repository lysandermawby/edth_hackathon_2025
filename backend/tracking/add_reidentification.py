#!/usr/bin/env python3
"""
Add Re-identification to Existing IntegratedRealtimeTracker

This script shows how to add re-identification capabilities to your existing
IntegratedRealtimeTracker with minimal code changes.
"""

import os
import sys

# Add the reidentify module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'reidentify'))

from robust_reidentification import RobustReidentificationSystem

def add_reidentification_to_tracker(tracker_instance):
    """
    Add re-identification capabilities to an existing IntegratedRealtimeTracker
    
    Args:
        tracker_instance: Instance of IntegratedRealtimeTracker
        
    Returns:
        Modified tracker with re-identification capabilities
    """
    # Add re-identification system to the tracker
    tracker_instance.reid_system = RobustReidentificationSystem(max_occlusion_frames=30)
    
    # Store the original process_frame method
    original_process_frame = tracker_instance.process_frame
    
    def enhanced_process_frame(frame, frame_timestamp):
        """Enhanced process_frame with re-identification"""
        # Run YOLO detection
        result = tracker_instance.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter ignored classes
        detections = tracker_instance.filter_detections(detections)
        
        # Update tracker
        detections = tracker_instance.tracker.update_with_detections(detections)
        
        # NEW: Apply re-identification
        import time
        timestamp = time.time()
        detections = tracker_instance.reid_system.process_detections(detections, frame, timestamp)
        
        # Extract tracking data for database
        tracking_data = tracker_instance.extract_tracking_data(detections, frame_timestamp)
        tracker_instance.tracking_data.append(tracking_data)
        
        # Send to real-time data processor
        if tracker_instance.data_processor:
            tracker_instance.data_processor.add_frame_data(tracking_data)
        
        # Create labels with re-identification info
        labels = []
        for i in range(len(detections)):
            if tracker_instance.show_labels and detections.class_id[i] is not None:
                class_name = tracker_instance.model.names[detections.class_id[i]]
                if detections.tracker_id[i] is not None:
                    # Check if this is a reidentified track
                    track_id = detections.tracker_id[i]
                    is_reidentified = track_id in tracker_instance.reid_system.occlusion_handler.lost_tracks
                    
                    label = f"{class_name} ID:{track_id}"
                    if is_reidentified:
                        label += " (REID)"
                    labels.append(label)
                else:
                    labels.append(f"{class_name} New")
            else:
                if detections.tracker_id[i] is not None:
                    track_id = detections.tracker_id[i]
                    is_reidentified = track_id in tracker_instance.reid_system.occlusion_handler.lost_tracks
                    
                    label = f"ID: {track_id}"
                    if is_reidentified:
                        label += " (REID)"
                    labels.append(label)
                else:
                    labels.append("New")
        
        # Annotate frame
        annotated_frame = tracker_instance.box_annotator.annotate(frame, detections)
        if labels:
            annotated_frame = tracker_instance.annotator.annotate(annotated_frame, detections, labels=labels)
        
        return annotated_frame, detections, tracking_data
    
    # Replace the process_frame method
    tracker_instance.process_frame = enhanced_process_frame
    
    # Add re-identification info to overlay
    original_add_info_overlay = tracker_instance.add_info_overlay
    
    def enhanced_add_info_overlay(frame, fps, total_objects, db_status=""):
        """Enhanced overlay with re-identification stats"""
        # Call original overlay
        frame = original_add_info_overlay(frame, fps, total_objects, db_status)
        
        # Add re-identification stats
        reid_stats = tracker_instance.reid_system.get_statistics()
        cv2.putText(frame, f"ReID: {reid_stats['successful_reidentifications']}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Active: {reid_stats['active_tracks']}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Lost: {reid_stats['lost_tracks_count']}", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        return frame
    
    # Replace the add_info_overlay method
    tracker_instance.add_info_overlay = enhanced_add_info_overlay
    
    print("✅ Re-identification capabilities added to tracker!")
    return tracker_instance

# Example usage
if __name__ == "__main__":
    # Import your existing tracker
    from integrated_realtime_tracker import IntegratedRealtimeTracker
    
    # Create your existing tracker
    tracker = IntegratedRealtimeTracker(
        show_labels=True,
        enable_database=True
    )
    
    # Add re-identification capabilities
    enhanced_tracker = add_reidentification_to_tracker(tracker)
    
    # Run with re-identification
    video_path = "/home/juanqui55/git/edth_hackathon_2025/data/Individual_2.mp4"
    enhanced_tracker.run(video_path)
