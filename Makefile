# Setup target - activate Poetry environment and check dependencies
.PHONY: setup
setup:
	@echo "Setting up Poetry environment..."
	@cd backend && poetry install
	@echo "Poetry environment ready!"

# Help target - show available commands
.PHONY: help
help:
	@echo "Available targets:"
	@echo ""
	@echo "  setup                   - Set up Poetry environment and install dependencies"
	@echo "  analyse                 - Run integrated real-time tracker on default video ($(TARGET_VIDEO))"
	@echo "  crop                    - Crop video using time configuration"
	@echo "  map_viewer              - Process drone video and GPS metadata (headless)"
	@echo "  reidentification_tracker - Run re-identification tracker on quantum drone video"
	@echo "  kinematic_reid_tracker   - Run enhanced kinematic re-identification tracker"
	@echo "  realtime_tracker        - Run real-time camera tracking with YOLO11 model"
	@echo "  demo_appearance         - Demo appearance-only re-identification"
	@echo "  demo_kinematic          - Demo kinematic-only re-identification"
	@echo "  demo_combined           - Demo combined appearance + kinematic re-identification"
	@echo "  demo_aggressive         - Demo with aggressive detection (low confidence, relaxed NMS)"
	@echo "  demo_ultra_aggressive   - Demo with ultra-aggressive detection (very low confidence)"
	@echo "  db_visualize            - Show database schema and basic tracking data"
	@echo "  db_visualize_session    - Show data for specific session (use SESSION_ID=1)"
	@echo "  db_visualize_plots      - Show database data with plots"
	@echo "  db_visualize_all        - Show comprehensive database analysis with plots"
	@echo "  frontend                - Start frontend application (both recorded and real-time)"
	@echo "  frontend_install        - Install all frontend and Python dependencies"
	@echo "  frontend_clean          - Clean up frontend processes"
	@echo "  help                    - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  TARGET_VIDEO=$(TARGET_VIDEO)"
	@echo "  VIDEO_PATH=$(VIDEO_PATH)"
	@echo "  CONFIG_PATH=$(CONFIG_PATH)"
	@echo "  METADATA_PATH=$(METADATA_PATH)"
	@echo "  DB_PATH=$(DB_PATH)"
	@echo "  MODEL_PATH=$(MODEL_PATH)"
	@echo ""
	@echo "Usage examples:"
	@echo "  make setup                   # Set up Poetry environment (run this first!)"
	@echo "  make help                    # Show this help"
	@echo "  make analyse                 # Run basic tracking"
	@echo "  make reidentification_tracker # Run re-identification tracker on quantum drone video"
	@echo "  make kinematic_reid_tracker   # Run enhanced kinematic re-identification tracker"
	@echo "  make realtime_tracker        # Run real-time camera tracking"
	@echo "  make demo_appearance         # Demo appearance-only re-identification"
	@echo "  make demo_kinematic          # Demo kinematic-only re-identification"
	@echo "  make demo_combined           # Demo combined re-identification"
	@echo "  make demo_aggressive         # Demo with aggressive detection settings"
	@echo "  make demo_ultra_aggressive   # Demo with ultra-aggressive detection settings"
	@echo "  make realtime_tracker MODEL_PATH=models/yolo11m.pt # Use specific model"
	@echo "  make crop                    # Crop video to specific time segments"
	@echo "  make map_viewer              # Process drone video and GPS metadata"
	@echo "  make db_visualize            # Show database contents"
	@echo "  make db_visualize_session SESSION_ID=1 # Show data for session 1"
	@echo "  make db_visualize_plots      # Show database with plots"
	@echo "  make db_visualize_all        # Comprehensive database analysis"
	@echo "  make frontend                # Start frontend application"
	@echo "  make frontend_install        # Install all dependencies"
	@echo "  make frontend_clean          # Clean up frontend processes"

# variable declarations

TARGET_VIDEO := ../data/Individual_2.mp4 # default video to process
MODEL_PATH := ../models/yolo11m.pt # default model for realtime tracker
ANALYSIS_SCRIPT := tracking/integrated_realtime_tracker.py
REIDENTIFICATION_TRACKER_SCRIPT := tracking/realtime_reidentification_tracker.py
KINEMATIC_REID_TRACKER_SCRIPT := tracking/realtime_reidentification_tracker.py
REALTIMETRACKER_SCRIPT := tracking/realtime_tracker.py
DEMO_TRACKER_SCRIPT := tracking/demo_reidentification_tracker.py

# Video cropping variables
VIDEO_PATH := ../data/quantum_drone_flight/2025_09_17-15_02_07_MovingObjects_44.ts
CONFIG_PATH := ../data/quantum_drone_flight/time_config.json
CROP_SCRIPT := video_crop/video_crop.py

# Video map viewer variables
MAP_VIEWER_SCRIPT := tracking/drone_video_map_viewer.py
METADATA_PATH := ../data/quantum_drone_flight/metadata.csv

# Database visualization variables
DB_VISUALIZATION_SCRIPT := db_query/data_visualisation.py
DB_PATH := ../databases/tracking_data.db

# Frontend variables
FRONTEND_DIR := standalone-frontend

# default target (help is shown if no target specified)
.DEFAULT_GOAL := help

# Python targets that depend on setup
analyse: setup
	@if [ -z "$(TARGET_VIDEO)" ]; then echo "Error: TARGET_VIDEO is not set"; exit 1; fi
	@if [ -z "$(ANALYSIS_SCRIPT)" ]; then echo "Error: ANALYSIS_SCRIPT is not set"; exit 1; fi
	@cd backend && poetry run python $(ANALYSIS_SCRIPT) $(TARGET_VIDEO)

# Video cropping target
crop: setup
	@if [ -z "$(VIDEO_PATH)" ]; then echo "Error: VIDEO_PATH is not set"; exit 1; fi
	@if [ -z "$(CONFIG_PATH)" ]; then echo "Error: CONFIG_PATH is not set"; exit 1; fi
	@if [ -z "$(CROP_SCRIPT)" ]; then echo "Error: CROP_SCRIPT is not set"; exit 1; fi
	@echo "Cropping video: $(VIDEO_PATH) with config: $(CONFIG_PATH)"
	@cd backend && poetry run python $(CROP_SCRIPT) $(VIDEO_PATH) --config $(CONFIG_PATH)

# Video map viewer target (headless)
map_viewer: setup
	@if [ -z "$(VIDEO_PATH)" ]; then echo "Error: VIDEO_PATH is not set"; exit 1; fi
	@if [ -z "$(METADATA_PATH)" ]; then echo "Error: METADATA_PATH is not set"; exit 1; fi
	@echo "Processing drone video and GPS metadata: $(VIDEO_PATH) with metadata: $(METADATA_PATH)"
	@cd backend && poetry run python $(MAP_VIEWER_SCRIPT) $(VIDEO_PATH) $(METADATA_PATH)

reidentification_tracker: setup
	@if [ -z "$(VIDEO_PATH)" ]; then echo "Error: VIDEO_PATH is not set"; exit 1; fi
	@echo "Running reidentification tracker: $(VIDEO_PATH)"
	@cd backend && poetry run python $(REIDENTIFICATION_TRACKER_SCRIPT) $(VIDEO_PATH)

kinematic_reid_tracker: setup
	@if [ -z "$(TARGET_VIDEO)" ]; then echo "Error: TARGET_VIDEO is not set"; exit 1; fi
	@if [ -z "$(KINEMATIC_REID_TRACKER_SCRIPT)" ]; then echo "Error: KINEMATIC_REID_TRACKER_SCRIPT is not set"; exit 1; fi
	@echo "Running enhanced kinematic re-identification tracker: $(TARGET_VIDEO)"
	@echo "Features: Position projection, velocity/acceleration tracking, multi-modal matching"
	@cd backend && poetry run python $(KINEMATIC_REID_TRACKER_SCRIPT) $(TARGET_VIDEO) --show-labels --conf-threshold 0.15 --enable-preprocessing

# Database visualization targets
db_visualize: setup
	@if [ -z "$(DB_VISUALIZATION_SCRIPT)" ]; then echo "Error: DB_VISUALIZATION_SCRIPT is not set"; exit 1; fi
	@if [ -z "$(DB_PATH)" ]; then echo "Error: DB_PATH is not set"; exit 1; fi
	@echo "Running database visualization: $(DB_PATH)"
	@cd backend && poetry run python $(DB_VISUALIZATION_SCRIPT) --db-path $(DB_PATH)

db_visualize_session: setup
	@if [ -z "$(DB_VISUALIZATION_SCRIPT)" ]; then echo "Error: DB_VISUALIZATION_SCRIPT is not set"; exit 1; fi
	@if [ -z "$(DB_PATH)" ]; then echo "Error: DB_PATH is not set"; exit 1; fi
	@if [ -z "$(SESSION_ID)" ]; then echo "Error: SESSION_ID is not set. Usage: make db_visualize_session SESSION_ID=1"; exit 1; fi
	@echo "Running database visualization for session $(SESSION_ID): $(DB_PATH)"
	@cd backend && poetry run python $(DB_VISUALIZATION_SCRIPT) --db-path $(DB_PATH) --session-id $(SESSION_ID)

db_visualize_plots: setup
	@if [ -z "$(DB_VISUALIZATION_SCRIPT)" ]; then echo "Error: DB_VISUALIZATION_SCRIPT is not set"; exit 1; fi
	@if [ -z "$(DB_PATH)" ]; then echo "Error: DB_PATH is not set"; exit 1; fi
	@echo "Running database visualization with plots: $(DB_PATH)"
	@cd backend && poetry run python $(DB_VISUALIZATION_SCRIPT) --db-path $(DB_PATH) --show-plots

db_visualize_all: setup
	@if [ -z "$(DB_VISUALIZATION_SCRIPT)" ]; then echo "Error: DB_VISUALIZATION_SCRIPT is not set"; exit 1; fi
	@if [ -z "$(DB_PATH)" ]; then echo "Error: DB_PATH is not set"; exit 1; fi
	@echo "Running comprehensive database visualization: $(DB_PATH)"
	@cd backend && poetry run python $(DB_VISUALIZATION_SCRIPT) --db-path $(DB_PATH) --show-plots --object-limit 0

# Frontend targets
frontend:
	@echo "Starting frontend application with both recorded and real-time capabilities..."
	@(cd $(FRONTEND_DIR) && ./start-realtime.sh)

frontend_install:
	@echo "Installing frontend dependencies..."
	@cd $(FRONTEND_DIR) && npm install
	@echo "Installing Python dependencies with Poetry..."
	@cd backend && poetry install

frontend_clean:
	@echo "Cleaning up frontend processes..."
	@pkill -f "node server.js" 2>/dev/null || true
	@pkill -f "vite" 2>/dev/null || true
	@pkill -f "realtime_server.py" 2>/dev/null || true
	@echo "Frontend processes cleaned up"

# Realtime tracker target
realtime_tracker: setup
	@if [ -z "$(REALTIMETRACKER_SCRIPT)" ]; then echo "Error: REALTIMETRACKER_SCRIPT is not set"; exit 1; fi
	@if [ -z "$(MODEL_PATH)" ]; then echo "Error: MODEL_PATH is not set"; exit 1; fi
	@echo "Stopping any existing realtime tracker processes..."
	@pkill -f "realtime_tracker.py" 2>/dev/null || true
	@echo "Running realtime tracker with model: $(MODEL_PATH)"
	@cd backend && poetry run python "$(REALTIMETRACKER_SCRIPT)" "$(MODEL_PATH)"

# Demo re-identification targets
demo_appearance: setup
	@if [ -z "$(TARGET_VIDEO)" ]; then echo "Error: TARGET_VIDEO is not set"; exit 1; fi
	@if [ -z "$(DEMO_TRACKER_SCRIPT)" ]; then echo "Error: DEMO_TRACKER_SCRIPT is not set"; exit 1; fi
	@echo "Running appearance-only re-identification demo: $(TARGET_VIDEO)"
	@echo "Features: Color histograms, HOG, deep learning features"
	@cd backend && poetry run python $(DEMO_TRACKER_SCRIPT) $(TARGET_VIDEO) --mode appearance --show-labels --conf-threshold 0.15

demo_kinematic: setup
	@if [ -z "$(TARGET_VIDEO)" ]; then echo "Error: TARGET_VIDEO is not set"; exit 1; fi
	@if [ -z "$(DEMO_TRACKER_SCRIPT)" ]; then echo "Error: DEMO_TRACKER_SCRIPT is not set"; exit 1; fi
	@echo "Running kinematic-only re-identification demo: $(TARGET_VIDEO)"
	@echo "Features: Position prediction, velocity/acceleration tracking, motion analysis"
	@cd backend && poetry run python $(DEMO_TRACKER_SCRIPT) $(TARGET_VIDEO) --mode kinematic --show-labels --conf-threshold 0.15

demo_combined: setup
	@if [ -z "$(TARGET_VIDEO)" ]; then echo "Error: TARGET_VIDEO is not set"; exit 1; fi
	@if [ -z "$(DEMO_TRACKER_SCRIPT)" ]; then echo "Error: DEMO_TRACKER_SCRIPT is not set"; exit 1; fi
	@echo "Running combined re-identification demo: $(TARGET_VIDEO)"
	@echo "Features: Appearance + Kinematic features with weighted fusion"
	@cd backend && poetry run python $(DEMO_TRACKER_SCRIPT) $(TARGET_VIDEO) --mode combined --show-labels --conf-threshold 0.15

# Aggressive detection demos
demo_aggressive: setup
	@if [ -z "$(TARGET_VIDEO)" ]; then echo "Error: TARGET_VIDEO is not set"; exit 1; fi
	@if [ -z "$(DEMO_TRACKER_SCRIPT)" ]; then echo "Error: DEMO_TRACKER_SCRIPT is not set"; exit 1; fi
	@echo "Running AGGRESSIVE detection demo: $(TARGET_VIDEO)"
	@echo "Features: Very low confidence threshold, enhanced preprocessing, relaxed NMS"
	@cd backend && poetry run python $(DEMO_TRACKER_SCRIPT) $(TARGET_VIDEO) --mode combined --show-labels --conf-threshold 0.05 --iou-threshold 0.5 --enable-preprocessing --max-det 2000

demo_ultra_aggressive: setup
	@if [ -z "$(TARGET_VIDEO)" ]; then echo "Error: TARGET_VIDEO is not set"; exit 1; fi
	@if [ -z "$(DEMO_TRACKER_SCRIPT)" ]; then echo "Error: DEMO_TRACKER_SCRIPT is not set"; exit 1; fi
	@echo "Running ULTRA-AGGRESSIVE detection demo: $(TARGET_VIDEO)"
	@echo "Features: Extremely low confidence, class-agnostic NMS, maximum detections"
	@cd backend && poetry run python $(DEMO_TRACKER_SCRIPT) $(TARGET_VIDEO) --mode combined --show-labels --conf-threshold 0.02 --iou-threshold 0.3 --agnostic-nms --enable-preprocessing --max-det 5000
