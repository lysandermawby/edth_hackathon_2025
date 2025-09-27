#!/usr/bin/env python3
"""
Test script for real-time depth estimation integration.
"""

import sys
import os
import cv2
import time

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'tracking'))

def test_depth_integration():
    print("Testing depth estimation integration...")

    # Test 1: Import depth estimator
    try:
        from depth_estimator import create_depth_estimator
        print("✓ Depth estimator import successful")
    except Exception as e:
        print(f"✗ Depth estimator import failed: {e}")
        return False

    # Test 2: Create depth estimator
    try:
        estimator = create_depth_estimator()
        print("✓ Depth estimator created")
    except Exception as e:
        print(f"✗ Depth estimator creation failed: {e}")
        return False

    # Test 3: Check availability
    if estimator.is_available():
        print("✓ Depth estimator is available")
    else:
        print("✗ Depth estimator not available")
        return False

    # Test 4: Test with a sample frame
    try:
        frame = cv2.imread('/Users/kenny/GitHub/edth_hackathon_2025/test_frame.jpg')
        if frame is None:
            print("✗ Could not load test frame")
            return False

        print("✓ Test frame loaded")

        # Create mock detection data
        mock_detections = [{
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.9,
            'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200},
            'center': {'x': 150, 'y': 150}
        }]

        print("Processing frame with depth estimation (optimized)...")
        start_time = time.time()

        # Test with optimized settings
        depth_map = estimator.estimate_depth(frame, "test_frame", resize_factor=0.3)

        if depth_map is not None:
            # Manually add depth to mock detection
            depth_stats = estimator.get_object_depth(depth_map, mock_detections[0]['bbox'])
            mock_detections[0]['depth'] = depth_stats
            enhanced_detections = mock_detections
        else:
            enhanced_detections = mock_detections

        processing_time = time.time() - start_time
        print(f"✓ Frame processed in {processing_time:.2f} seconds")

        if depth_map is not None:
            print(f"✓ Depth map generated: {depth_map.shape}")
            print(f"  Depth range: {depth_map.min():.2f} - {depth_map.max():.2f} meters")
        else:
            print("✗ No depth map generated")
            return False

        if enhanced_detections and 'depth' in enhanced_detections[0]:
            depth_info = enhanced_detections[0]['depth']
            print(f"✓ Object depth estimated: {depth_info['mean_depth']:.2f}m")
        else:
            print("✗ No depth information in enhanced detections")
            return False

        print("✓ All tests passed!")
        return True

    except Exception as e:
        print(f"✗ Frame processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_depth_integration()
    if success:
        print("\n🎉 Depth estimation integration is working correctly!")
    else:
        print("\n❌ Depth estimation integration has issues.")

    sys.exit(0 if success else 1)