# variable declarations

TARGET_VIDEO := data/Individual_2.mp4 # default video to process
ANALYSIS_SCRIPT := backend/tracking/integrated_realtime_tracker.py

# Video cropping variables
VIDEO_PATH := data/quantum_drone_flight/2025_09_17-15_02_07_MovingObjects_44.ts
CONFIG_PATH := data/quantum_drone_flight/time_config.json
CROP_SCRIPT := backend/video_crop/video_crop.py

# default target
analyse: 
	@if [ -z "$(TARGET_VIDEO)" ]; then echo "Error: TARGET_VIDEO is not set"; exit 1; fi
	@if [ -z "$(ANALYSIS_SCRIPT)" ]; then echo "Error: ANALYSIS_SCRIPT is not set"; exit 1; fi
	@python $(ANALYSIS_SCRIPT) $(TARGET_VIDEO)

# Video cropping target
crop:
	@if [ -z "$(VIDEO_PATH)" ]; then echo "Error: VIDEO_PATH is not set"; exit 1; fi
	@if [ -z "$(CONFIG_PATH)" ]; then echo "Error: CONFIG_PATH is not set"; exit 1; fi
	@if [ -z "$(CROP_SCRIPT)" ]; then echo "Error: CROP_SCRIPT is not set"; exit 1; fi
	@echo "Cropping video: $(VIDEO_PATH) with config: $(CONFIG_PATH)"
	@python $(CROP_SCRIPT) $(VIDEO_PATH) --config $(CONFIG_PATH)

