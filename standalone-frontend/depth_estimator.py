#!/usr/bin/env python3
"""
Depth estimation module using Apple's Depth Pro model.

This module provides depth estimation functionality for detected objects
in the video tracking pipeline.
"""

import cv2
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys
import os
import threading
import queue
import time

# Add the depth_pro module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml-depth-pro', 'src'))

try:
    import depth_pro
except ImportError as e:
    logging.error(f"Error importing depth_pro: {e}")
    logging.error("Please ensure Apple Depth Pro is properly installed")
    depth_pro = None

class DepthEstimator:
    """Depth estimation using Apple Depth Pro model."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the depth estimator.

        Args:
            model_path: Path to the Depth Pro model checkpoint. If None, uses default.
        """
        self.model = None
        self.transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.is_initialized = False

        # Cache for depth maps to avoid recomputation for same frame
        self.depth_cache = {}
        self.cache_size_limit = 10  # Keep last 10 depth maps

        # Threading for non-blocking depth processing
        self.processing_queue = queue.Queue(maxsize=2)  # Small queue to avoid memory issues
        self.result_queue = queue.Queue()
        self.stop_processing = False
        self.processing_thread = None

        # Initialize the model
        self._initialize_model()

        # Start background processing thread if initialized
        if self.is_initialized:
            self._start_background_processing()

    def _initialize_model(self):
        """Initialize the Depth Pro model."""
        if depth_pro is None:
            logging.error("Depth Pro not available - depth estimation disabled")
            return

        try:
            # Set model path if not provided
            if self.model_path is None:
                project_root = Path(__file__).parent.parent
                self.model_path = project_root / "ml-depth-pro" / "checkpoints" / "depth_pro.pt"

            # Expand path to absolute
            self.model_path = Path(self.model_path).resolve()

            # Check if model file exists
            if not os.path.exists(self.model_path):
                logging.warning(f"Depth Pro model not found at {self.model_path}")
                logging.warning("Download the model using: cd ml-depth-pro && ./get_pretrained_models.sh")
                return

            # Create config with correct checkpoint path
            from depth_pro.depth_pro import DepthProConfig
            config = DepthProConfig(
                patch_encoder_preset='dinov2l16_384',
                image_encoder_preset='dinov2l16_384',
                decoder_features=256,
                checkpoint_uri=str(self.model_path),
                fov_encoder_preset='dinov2l16_384',
                use_fov_head=True
            )

            # Create model and transforms
            self.model, self.transform = depth_pro.create_model_and_transforms(
                config=config,
                device=self.device,
                precision=torch.half if self.device.type == 'cuda' else torch.float32
            )

            self.model.eval()

            self.is_initialized = True
            logging.info(f"Depth Pro model initialized successfully on {self.device}")

        except Exception as e:
            logging.error(f"Failed to initialize Depth Pro model: {e}")
            self.is_initialized = False

    def _start_background_processing(self):
        """Start background thread for depth processing."""
        self.processing_thread = threading.Thread(target=self._background_processor, daemon=True)
        self.processing_thread.start()
        logging.info("Background depth processing thread started")

    def _background_processor(self):
        """Background thread that processes depth estimation requests."""
        while not self.stop_processing:
            try:
                # Get frame from queue (with timeout to check stop condition)
                try:
                    frame_data = self.processing_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                frame, frame_id, resize_factor = frame_data

                # Process depth estimation
                try:
                    depth_map = self._compute_depth_sync(frame, resize_factor)
                    self.result_queue.put((frame_id, depth_map, time.time()))

                    # Cache the result
                    if frame_id:
                        self._add_to_cache(frame_id, depth_map)

                except Exception as e:
                    logging.error(f"Background depth processing error: {e}")
                    self.result_queue.put((frame_id, None, time.time()))

                finally:
                    self.processing_queue.task_done()

            except Exception as e:
                logging.error(f"Background processor error: {e}")

    def _compute_depth_sync(self, frame: np.ndarray, resize_factor: float) -> Optional[np.ndarray]:
        """Synchronously compute depth map (for background thread)."""
        try:
            # Resize frame for faster processing
            if resize_factor != 1.0:
                height, width = frame.shape[:2]
                new_height, new_width = int(height * resize_factor), int(width * resize_factor)
                frame_resized = cv2.resize(frame, (new_width, new_height))
            else:
                frame_resized = frame

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image for Depth Pro
            from PIL import Image
            pil_image = Image.fromarray(rgb_frame)

            # Apply transforms and move to device
            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                prediction = self.model.infer(image_tensor)
                depth_map_small = prediction["depth"].cpu().numpy().squeeze()

            # Resize depth map back to original size if needed
            if resize_factor != 1.0:
                original_height, original_width = frame.shape[:2]
                depth_map = cv2.resize(depth_map_small, (original_width, original_height))
            else:
                depth_map = depth_map_small

            return depth_map

        except Exception as e:
            logging.error(f"Sync depth computation failed: {e}")
            return None

    def estimate_depth_async(self, frame: np.ndarray, frame_id: Optional[str] = None, resize_factor: float = 0.15) -> bool:
        """
        Submit frame for asynchronous depth estimation.

        Args:
            frame: Input BGR frame from OpenCV
            frame_id: Optional frame identifier for caching
            resize_factor: Factor to resize frame for faster processing

        Returns:
            True if submitted successfully, False otherwise
        """
        if not self.is_initialized or self.processing_thread is None:
            return False

        # Check cache first
        if frame_id and frame_id in self.depth_cache:
            return True

        try:
            # Try to add to processing queue (non-blocking)
            self.processing_queue.put_nowait((frame.copy(), frame_id, resize_factor))
            return True
        except queue.Full:
            # Queue is full, skip this frame
            logging.debug("Depth processing queue full, skipping frame")
            return False

    def get_latest_depth_result(self) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Get the latest depth result from background processing.

        Returns:
            Tuple of (frame_id, depth_map) or (None, None) if no result available
        """
        try:
            # Get all available results and return the latest
            latest_result = None
            while True:
                try:
                    result = self.result_queue.get_nowait()
                    latest_result = result
                except queue.Empty:
                    break

            if latest_result:
                frame_id, depth_map, timestamp = latest_result
                return frame_id, depth_map
            else:
                return None, None

        except Exception as e:
            logging.error(f"Error getting depth result: {e}")
            return None, None

    def estimate_depth(self, frame: np.ndarray, frame_id: Optional[str] = None, resize_factor: float = 0.15) -> Optional[np.ndarray]:
        """
        Estimate depth for the entire frame (synchronous fallback).

        Args:
            frame: Input BGR frame from OpenCV
            frame_id: Optional frame identifier for caching
            resize_factor: Factor to resize frame for faster processing

        Returns:
            Depth map as numpy array (height, width) with depth in meters, or None if failed
        """
        if not self.is_initialized:
            return None

        # Check cache first
        if frame_id and frame_id in self.depth_cache:
            return self.depth_cache[frame_id]

        # Use synchronous computation for fallback
        return self._compute_depth_sync(frame, resize_factor)

    def get_object_depth(self, depth_map: np.ndarray, bbox: Dict[str, float]) -> Dict[str, float]:
        """
        Extract depth information for a specific bounding box.

        Args:
            depth_map: Full frame depth map
            bbox: Bounding box with keys x1, y1, x2, y2

        Returns:
            Dictionary with depth statistics: mean, median, min, max, std
        """
        try:
            # Extract bounding box coordinates
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])

            # Ensure coordinates are within bounds
            h, w = depth_map.shape
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            # Extract depth values within bounding box
            object_depth = depth_map[y1:y2, x1:x2]

            if object_depth.size == 0:
                return self._get_default_depth_stats()

            # Filter out invalid depth values (assuming 0 or negative are invalid)
            valid_depths = object_depth[object_depth > 0]

            if valid_depths.size == 0:
                return self._get_default_depth_stats()

            # Calculate statistics
            depth_stats = {
                'mean_depth': float(np.mean(valid_depths)),
                'median_depth': float(np.median(valid_depths)),
                'min_depth': float(np.min(valid_depths)),
                'max_depth': float(np.max(valid_depths)),
                'std_depth': float(np.std(valid_depths)),
                'valid_pixels': int(valid_depths.size),
                'total_pixels': int(object_depth.size)
            }

            return depth_stats

        except Exception as e:
            logging.error(f"Object depth extraction failed: {e}")
            return self._get_default_depth_stats()

    def _get_default_depth_stats(self) -> Dict[str, float]:
        """Return default depth statistics when estimation fails."""
        return {
            'mean_depth': 0.0,
            'median_depth': 0.0,
            'min_depth': 0.0,
            'max_depth': 0.0,
            'std_depth': 0.0,
            'valid_pixels': 0,
            'total_pixels': 0
        }

    def _add_to_cache(self, frame_id: str, depth_map: np.ndarray):
        """Add depth map to cache with size limit."""
        if len(self.depth_cache) >= self.cache_size_limit:
            # Remove oldest entry
            oldest_key = next(iter(self.depth_cache))
            del self.depth_cache[oldest_key]

        self.depth_cache[frame_id] = depth_map

    def create_depth_visualization(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Create a colorized visualization of the depth map.

        Args:
            depth_map: Depth map as numpy array

        Returns:
            Colorized depth map as BGR image for OpenCV display
        """
        try:
            # Normalize depth map to 0-255 range
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            # Apply colormap (INFERNO gives nice depth visualization)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

            return depth_colored

        except Exception as e:
            logging.error(f"Depth visualization creation failed: {e}")
            # Return a blank image of same size as input
            return np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)

    def process_frame_with_detections(self, frame: np.ndarray, detections: List[Dict],
                                    frame_id: Optional[str] = None) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """
        Process a frame with detections to add depth information.

        Args:
            frame: Input BGR frame
            detections: List of detection dictionaries with bounding boxes
            frame_id: Optional frame identifier for caching

        Returns:
            Tuple of (depth_map, enhanced_detections_with_depth)
        """
        # Estimate depth for the entire frame
        depth_map = self.estimate_depth(frame, frame_id)

        # Enhance detections with depth information
        enhanced_detections = []
        for detection in detections:
            enhanced_detection = detection.copy()

            if depth_map is not None and 'bbox' in detection:
                # Get depth statistics for this detection
                depth_stats = self.get_object_depth(depth_map, detection['bbox'])
                enhanced_detection['depth'] = depth_stats
            else:
                # Add default depth info if estimation failed
                enhanced_detection['depth'] = self._get_default_depth_stats()

            enhanced_detections.append(enhanced_detection)

        return depth_map, enhanced_detections

    def is_available(self) -> bool:
        """Check if depth estimation is available."""
        return self.is_initialized and depth_pro is not None


def create_depth_estimator(model_path: Optional[str] = None) -> DepthEstimator:
    """
    Factory function to create a DepthEstimator instance.

    Args:
        model_path: Optional path to the Depth Pro model

    Returns:
        DepthEstimator instance
    """
    return DepthEstimator(model_path)


if __name__ == "__main__":
    # Test the depth estimator
    import sys

    if len(sys.argv) < 2:
        print("Usage: python depth_estimator.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Initialize depth estimator
    estimator = create_depth_estimator()

    if not estimator.is_available():
        print("Depth estimator not available")
        sys.exit(1)

    # Load test image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load image: {image_path}")
        sys.exit(1)

    print(f"Testing depth estimation on {image_path}")

    # Estimate depth
    depth_map = estimator.estimate_depth(frame)

    if depth_map is not None:
        print(f"Depth map shape: {depth_map.shape}")
        print(f"Depth range: {np.min(depth_map):.2f} - {np.max(depth_map):.2f} meters")

        # Create visualization
        depth_vis = estimator.create_depth_visualization(depth_map)

        # Save results
        output_dir = Path("depth_test_output")
        output_dir.mkdir(exist_ok=True)

        cv2.imwrite(str(output_dir / "original.jpg"), frame)
        cv2.imwrite(str(output_dir / "depth_visualization.jpg"), depth_vis)

        print(f"Results saved to {output_dir}/")

    else:
        print("Depth estimation failed")