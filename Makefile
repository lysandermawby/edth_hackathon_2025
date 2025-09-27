# Help target - show available commands
.PHONY: help
help:
	@echo "Available targets:"
	@echo ""
	@echo "  analyse                 - Run integrated real-time tracker on default video ($(TARGET_VIDEO))"
	@echo "  crop                    - Crop video using time configuration"
	@echo "  map_viewer              - Launch drone video and map viewer with GPS tracking"
	@echo "  reidentification_tracker - Run re-identification tracker on quantum drone video"
	@echo "  db_visualize            - Show database schema and basic tracking data"
	@echo "  db_visualize_session    - Show data for specific session (use SESSION_ID=1)"
	@echo "  db_visualize_plots      - Show database data with plots"
	@echo "  db_visualize_all        - Show comprehensive database analysis with plots"
	@echo "  frontend                - Start frontend application (recorded sessions)"
	@echo "  frontend_realtime       - Start frontend with real-time detection"
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
	@echo ""
	@echo "Usage examples:"
	@echo "  make help                    # Show this help"
	@echo "  make analyse                 # Run basic tracking"
	@echo "  make map_viewer              # View drone video with GPS map overlay"
	@echo "  make reidentification_tracker # Run re-identification tracker on quantum drone video"
	@echo "  make crop                    # Crop video to specific time segments"
	@echo "  make db_visualize            # Show database contents"
	@echo "  make db_visualize_session SESSION_ID=1 # Show data for session 1"
	@echo "  make db_visualize_plots      # Show database with plots"
	@echo "  make db_visualize_all        # Comprehensive database analysis"
	@echo "  make frontend                # Start frontend application"
	@echo "  make frontend_realtime       # Start frontend with real-time detection"
	@echo "  make frontend_install        # Install all dependencies"
	@echo "  make frontend_clean          # Clean up frontend processes"

# variable declarations

TARGET_VIDEO := data/Individual_2.mp4 # default video to process
ANALYSIS_SCRIPT := backend/tracking/integrated_realtime_tracker.py
REIDENTIFICATION_TRACKER_SCRIPT := backend/tracking/realtime_reidentification_tracker.py

# Video cropping variables
VIDEO_PATH := data/quantum_drone_flight/2025_09_17-15_02_07_MovingObjects_44.ts
CONFIG_PATH := data/quantum_drone_flight/time_config.json
CROP_SCRIPT := backend/video_crop/video_crop.py

# Video map viewer variables
MAP_VIEWER_SCRIPT := backend/tracking/drone_video_map_viewer.py
METADATA_PATH := data/quantum_drone_flight/metadata.csv

# Database visualization variables
DB_VISUALIZATION_SCRIPT := backend/db_query/data_visualisation.py
DB_PATH := databases/tracking_data.db

# Frontend variables
FRONTEND_DIR := standalone-frontend
FRONTEND_SCRIPT := run-standalone-frontend.sh
REALTIME_SCRIPT := standalone-frontend/start-realtime.sh

# default target (help is shown if no target specified)
.DEFAULT_GOAL := help

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

# Video map viewer target
map_viewer:
	@if [ -z "$(VIDEO_PATH)" ]; then echo "Error: VIDEO_PATH is not set"; exit 1; fi
	@if [ -z "$(METADATA_PATH)" ]; then echo "Error: METADATA_PATH is not set"; exit 1; fi
	@echo "Running video map viewer: $(VIDEO_PATH) with metadata: $(METADATA_PATH)"
	@python $(MAP_VIEWER_SCRIPT) $(VIDEO_PATH) $(METADATA_PATH)

reidentification_tracker:
	@if [ -z "$(VIDEO_PATH)" ]; then echo "Error: VIDEO_PATH is not set"; exit 1; fi
	@echo "Running reidentification tracker: $(VIDEO_PATH)"
	@python $(REIDENTIFICATION_TRACKER_SCRIPT) $(VIDEO_PATH)

# Database visualization targets
db_visualize:
	@if [ -z "$(DB_VISUALIZATION_SCRIPT)" ]; then echo "Error: DB_VISUALIZATION_SCRIPT is not set"; exit 1; fi
	@if [ -z "$(DB_PATH)" ]; then echo "Error: DB_PATH is not set"; exit 1; fi
	@echo "Running database visualization: $(DB_PATH)"
	@python $(DB_VISUALIZATION_SCRIPT) --db-path $(DB_PATH)

db_visualize_session:
	@if [ -z "$(DB_VISUALIZATION_SCRIPT)" ]; then echo "Error: DB_VISUALIZATION_SCRIPT is not set"; exit 1; fi
	@if [ -z "$(DB_PATH)" ]; then echo "Error: DB_PATH is not set"; exit 1; fi
	@if [ -z "$(SESSION_ID)" ]; then echo "Error: SESSION_ID is not set. Usage: make db_visualize_session SESSION_ID=1"; exit 1; fi
	@echo "Running database visualization for session $(SESSION_ID): $(DB_PATH)"
	@python $(DB_VISUALIZATION_SCRIPT) --db-path $(DB_PATH) --session-id $(SESSION_ID)

db_visualize_plots:
	@if [ -z "$(DB_VISUALIZATION_SCRIPT)" ]; then echo "Error: DB_VISUALIZATION_SCRIPT is not set"; exit 1; fi
	@if [ -z "$(DB_PATH)" ]; then echo "Error: DB_PATH is not set"; exit 1; fi
	@echo "Running database visualization with plots: $(DB_PATH)"
	@python $(DB_VISUALIZATION_SCRIPT) --db-path $(DB_PATH) --show-plots

db_visualize_all:
	@if [ -z "$(DB_VISUALIZATION_SCRIPT)" ]; then echo "Error: DB_VISUALIZATION_SCRIPT is not set"; exit 1; fi
	@if [ -z "$(DB_PATH)" ]; then echo "Error: DB_PATH is not set"; exit 1; fi
	@echo "Running comprehensive database visualization: $(DB_PATH)"
	@python $(DB_VISUALIZATION_SCRIPT) --db-path $(DB_PATH) --show-plots --object-limit 0

# Frontend targets
frontend:
	@if [ -z "$(FRONTEND_SCRIPT)" ]; then echo "Error: FRONTEND_SCRIPT is not set"; exit 1; fi
	@echo "Starting frontend application..."
	@./$(FRONTEND_SCRIPT)

frontend_realtime:
	@if [ -z "$(REALTIME_SCRIPT)" ]; then echo "Error: REALTIME_SCRIPT is not set"; exit 1; fi
	@echo "Starting frontend with real-time detection..."
	@./$(REALTIME_SCRIPT)

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


