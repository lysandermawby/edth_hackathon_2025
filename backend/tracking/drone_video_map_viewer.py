#!/usr/bin/env python3
"""
Drone Video and Map Viewer

Displays drone video synchronized with real-time map visualization showing:
- Current camera position on satellite map
- Camera orientation and field of view cone
- Historical flight path trace
- Real-time synchronization between video frames and GPS metadata

Usage:
    python drone_video_map_viewer.py <video_path> <metadata_csv_path>
"""

import os
import sys
import csv
import cv2
import numpy as np
import math
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import requests
from PIL import Image, ImageTk
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Wedge, Circle
import matplotlib.patches as patches
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue


class DroneVideoMapViewer:
    def __init__(self, video_path, metadata_path):
        self.video_path = video_path
        self.metadata_path = metadata_path

        # Video properties
        self.cap = None
        self.frame_rate = 25.0  # From video analysis
        self.frame_count = 0
        self.current_frame = 0

        # Metadata
        self.metadata = []
        self.load_metadata()

        # Map properties
        self.map_center_lat = None
        self.map_center_lon = None
        self.map_zoom = 16  # Higher zoom for more detail (fixed base zoom)
        self.map_size = 800
        self.satellite_image = None
        self.map_extent = None

        # Adaptive tile system
        self.current_tile_center = None  # Current center of loaded tiles
        self.tile_cache = {}  # Cache for downloaded tiles
        self.tile_grid_size = 3  # Reduced to 3x3 for better performance
        self.needs_tile_update = True
        self.download_executor = ThreadPoolExecutor(max_workers=8)  # Concurrent downloads
        self.pending_downloads = set()  # Track pending downloads

        # GUI
        self.root = tk.Tk()
        self.root.title("Drone Video & Map Viewer")
        self.root.geometry("1600x1000")

        # Control variables
        self.is_playing = False
        self.playback_speed = 1.0

        # Initialize GUI components
        self.calculate_map_bounds()  # Calculate bounds before GUI setup
        self.setup_gui()
        self.load_video()

    def load_metadata(self):
        """Load GPS and orientation metadata from CSV file"""
        print(f"Loading metadata from {self.metadata_path}")

        try:
            with open(self.metadata_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert string values to appropriate types
                    metadata_entry = {
                        'timestamp': int(row['timestamp']),
                        'latitude': float(row['latitude']),
                        'longitude': float(row['longitude']),
                        'altitude': float(row['altitude']),
                        'roll': float(row['roll']),
                        'pitch': float(row['pitch']),
                        'yaw': float(row['yaw']),
                        'gimbal_elevation': float(row['gimbal_elevation']),
                        'gimbal_azimuth': float(row['gimbal_azimuth']),
                        'vfov': float(row['vfov']),
                        'hfov': float(row['hfov'])
                    }
                    self.metadata.append(metadata_entry)

            print(f"Loaded {len(self.metadata)} metadata entries")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load metadata: {e}")
            sys.exit(1)

    def load_video(self):
        """Load video file"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise Exception("Could not open video file")

            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)

            print(f"Video loaded: {self.frame_count} frames at {self.frame_rate} FPS")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {e}")
            sys.exit(1)

    def calculate_map_bounds(self):
        """Calculate map center and bounds from GPS data"""
        if not self.metadata:
            # Default center if no metadata
            self.map_center_lat = 48.0
            self.map_center_lon = 11.0
            return

        # Filter out invalid coordinates
        valid_entries = [entry for entry in self.metadata
                        if -90 < entry['latitude'] < 90 and -180 < entry['longitude'] < 180]

        if not valid_entries:
            # Default center if no valid coordinates
            self.map_center_lat = 48.0
            self.map_center_lon = 11.0
            return

        lats = [entry['latitude'] for entry in valid_entries]
        lons = [entry['longitude'] for entry in valid_entries]

        self.map_center_lat = sum(lats) / len(lats)
        self.map_center_lon = sum(lons) / len(lons)

        print(f"Debug: lats range: {min(lats)} to {max(lats)}")
        print(f"Debug: lons range: {min(lons)} to {max(lons)}")
        print(f"Debug: map center: {self.map_center_lat}, {self.map_center_lon}")

        # Calculate bounds for zoom level
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        max_range = max(lat_range, lon_range)

        print(f"Map center: {self.map_center_lat:.6f}, {self.map_center_lon:.6f}")
        print(f"Map zoom: {self.map_zoom}")
        print(f"Valid GPS entries: {len(valid_entries)}/{len(self.metadata)}")

        # Initialize with first drone position if available
        if valid_entries:
            first_pos = valid_entries[0]
            self.update_satellite_tiles(first_pos['latitude'], first_pos['longitude'])

    def deg2num(self, lat_deg, lon_deg, zoom):
        """Convert lat/lon to tile numbers"""
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (xtile, ytile)

    def num2deg(self, xtile, ytile, zoom):
        """Convert tile numbers to lat/lon"""
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return (lat_deg, lon_deg)

    def download_single_tile(self, tile_x, tile_y):
        """Download a single tile (for use in thread pool)"""
        tile_key = (tile_x, tile_y, self.map_zoom)

        # Try multiple satellite tile sources for reliability
        tile_sources = [
            # ArcGIS World Imagery (most reliable satellite imagery)
            f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{self.map_zoom}/{tile_y}/{tile_x}",
            # Alternative ArcGIS servers for redundancy
            f"https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{self.map_zoom}/{tile_y}/{tile_x}",
            # Bing Maps satellite (via different endpoint)
            f"https://t0.ssl.ak.dynamic.tiles.virtualearth.net/comp/ch/{self.quadkey_from_tile(tile_x, tile_y, self.map_zoom)}?mkt=en-us&it=A,G,L&shading=hill"
        ]

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.arcgis.com/'
        }

        for i, url in enumerate(tile_sources):
            try:
                print(f"Trying source {i+1} for tile {tile_x},{tile_y}: {url[:50]}...")
                response = requests.get(url, timeout=8, headers=headers)
                if response.status_code == 200 and len(response.content) > 1000:  # Ensure we got actual image data
                    tile_image = Image.open(BytesIO(response.content))
                    # Verify image is valid
                    tile_image.verify()
                    tile_image = Image.open(BytesIO(response.content))  # Reload after verify
                    print(f"Successfully downloaded tile {tile_x},{tile_y} from source {i+1}")
                    return tile_key, tile_image
                else:
                    print(f"Source {i+1} failed: HTTP {response.status_code}, size: {len(response.content)}")
            except Exception as e:
                print(f"Source {i+1} error for tile {tile_x},{tile_y}: {e}")
                continue

        print(f"All sources failed for tile {tile_x},{tile_y}")
        # Return blank tile as fallback
        blank_tile = Image.new('RGB', (256, 256), (200, 200, 200))
        return tile_key, blank_tile

    def quadkey_from_tile(self, tile_x, tile_y, zoom):
        """Convert tile coordinates to Bing quadkey"""
        quadkey = ""
        for i in range(zoom, 0, -1):
            digit = 0
            mask = 1 << (i - 1)
            if (tile_x & mask) != 0:
                digit += 1
            if (tile_y & mask) != 0:
                digit += 2
            quadkey += str(digit)
        return quadkey

    def get_tile_from_cache_or_download(self, tile_x, tile_y):
        """Get tile from cache or return placeholder if downloading"""
        tile_key = (tile_x, tile_y, self.map_zoom)

        if tile_key in self.tile_cache:
            return self.tile_cache[tile_key]

        # Return loading placeholder - download should have happened in update_satellite_tiles
        loading_tile = Image.new('RGB', (256, 256), (180, 180, 180))
        # Add some visual indication this is a placeholder
        from PIL import ImageDraw
        draw = ImageDraw.Draw(loading_tile)
        draw.text((100, 120), "Loading...", fill=(100, 100, 100))
        return loading_tile

    def is_in_comfort_zone(self, drone_lat, drone_lon):
        """Check if drone is within the comfort zone of current tiles"""
        if self.current_tile_center is None:
            return False

        center_lat, center_lon = self.current_tile_center
        lat_diff = abs(drone_lat - center_lat)
        lon_diff = abs(drone_lon - center_lon)

        # Scale comfort zone with zoom level - smaller zone at higher zoom
        base_comfort_zone = 0.005  # Base comfort zone for zoom level 16 (current default)
        zoom_factor = 2 ** (self.map_zoom - 16)  # Use actual default zoom as base
        comfort_zone_size = base_comfort_zone / zoom_factor

        return lat_diff < comfort_zone_size and lon_diff < comfort_zone_size

    def update_satellite_tiles(self, drone_lat, drone_lon, force_update=False):
        """Update satellite tiles based on drone position with concurrent downloads"""
        if not force_update and self.is_in_comfort_zone(drone_lat, drone_lon):
            return  # No need to update tiles

        print(f"Updating satellite tiles for position: {drone_lat:.6f}, {drone_lon:.6f}")

        try:
            # Calculate tile coordinates for drone position
            center_x, center_y = self.deg2num(drone_lat, drone_lon, self.map_zoom)

            tile_size = 256
            grid_half = self.tile_grid_size // 2

            # Identify tiles to download
            tiles_to_download = []
            tile_positions = {}

            for dy in range(-grid_half, grid_half + 1):
                for dx in range(-grid_half, grid_half + 1):
                    tile_x = center_x + dx
                    tile_y = center_y + dy
                    tile_key = (tile_x, tile_y, self.map_zoom)
                    tile_positions[tile_key] = (dy + grid_half, dx + grid_half)

                    if tile_key not in self.tile_cache and tile_key not in self.pending_downloads:
                        tiles_to_download.append((tile_x, tile_y))
                        self.pending_downloads.add(tile_key)

            # Submit concurrent download tasks
            if tiles_to_download:
                download_futures = {}
                for tile_x, tile_y in tiles_to_download:
                    future = self.download_executor.submit(self.download_single_tile, tile_x, tile_y)
                    download_futures[future] = (tile_x, tile_y)

                # Wait for downloads to complete (with reasonable timeout)
                completed_tiles = 0
                try:
                    for future in as_completed(download_futures, timeout=5.0):
                        try:
                            tile_key, tile_image = future.result()
                            self.tile_cache[tile_key] = tile_image
                            self.pending_downloads.discard(tile_key)
                            completed_tiles += 1
                        except Exception as e:
                            tile_x, tile_y = download_futures[future]
                            tile_key = (tile_x, tile_y, self.map_zoom)
                            self.pending_downloads.discard(tile_key)
                            print(f"Failed to download tile {tile_x},{tile_y}: {e}")
                except Exception as timeout_e:
                    print(f"Download timeout, got {completed_tiles} tiles")

                if completed_tiles > 0:
                    print(f"Downloaded {completed_tiles} new tiles")

            # Build image from available tiles (cached or loading placeholders)
            tiles = []
            for dy in range(-grid_half, grid_half + 1):
                tile_row = []
                for dx in range(-grid_half, grid_half + 1):
                    tile_x = center_x + dx
                    tile_y = center_y + dy
                    tile_image = self.get_tile_from_cache_or_download(tile_x, tile_y)
                    tile_row.append(tile_image)
                tiles.append(tile_row)

            # Combine tiles into a single image
            combined_width = len(tiles[0]) * tile_size
            combined_height = len(tiles) * tile_size
            combined_image = Image.new('RGB', (combined_width, combined_height))

            for row_idx, tile_row in enumerate(tiles):
                for col_idx, tile in enumerate(tile_row):
                    x = col_idx * tile_size
                    y = row_idx * tile_size
                    combined_image.paste(tile, (x, y))

            self.satellite_image = combined_image
            self.current_tile_center = (drone_lat, drone_lon)
            print(f"Created combined satellite image: {combined_width}x{combined_height}, actual tiles in cache: {len([k for k in self.tile_cache.keys() if k[2] == self.map_zoom])}")

            # Calculate the geographic extent of the combined image
            top_left_lat, top_left_lon = self.num2deg(center_x - grid_half, center_y - grid_half, self.map_zoom)
            bottom_right_lat, bottom_right_lon = self.num2deg(center_x + grid_half + 1, center_y + grid_half + 1, self.map_zoom)

            self.map_extent = [top_left_lon, bottom_right_lon, bottom_right_lat, top_left_lat]

            # Intelligent cache management - keep more relevant tiles
            if len(self.tile_cache) > 150:
                # Remove tiles furthest from current center
                current_center_x, current_center_y = center_x, center_y
                tiles_with_distance = []

                for (tile_x, tile_y, zoom), tile_img in self.tile_cache.items():
                    if zoom == self.map_zoom:  # Only consider tiles from current zoom
                        distance = abs(tile_x - current_center_x) + abs(tile_y - current_center_y)
                        tiles_with_distance.append((distance, (tile_x, tile_y, zoom)))

                # Sort by distance and remove furthest 30 tiles
                tiles_with_distance.sort(reverse=True)
                for _, tile_key in tiles_with_distance[:30]:
                    if tile_key in self.tile_cache:
                        del self.tile_cache[tile_key]

        except Exception as e:
            print(f"Failed to update satellite tiles: {e}")

    def setup_gui(self):
        """Setup the GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top section: Video and Map side by side
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.BOTH, expand=True)

        # Video frame
        video_frame = ttk.LabelFrame(top_frame, text="Drone Video", padding=10)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.video_label = ttk.Label(video_frame, text="Video will appear here")
        self.video_label.pack()

        # Map frame
        map_frame = ttk.LabelFrame(top_frame, text="Real-time Map", padding=10)
        map_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Matplotlib figure for map
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.map_canvas = FigureCanvasTkAgg(self.fig, map_frame)
        self.map_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom section: Controls and info
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))

        # Video controls
        controls_subframe = ttk.Frame(control_frame)
        controls_subframe.pack(fill=tk.X)

        # Playback controls
        ttk.Button(controls_subframe, text="Play/Pause", command=self.toggle_playback).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_subframe, text="Reset", command=self.reset_video).pack(side=tk.LEFT, padx=(0, 5))

        # Speed control
        ttk.Label(controls_subframe, text="Speed:").pack(side=tk.LEFT, padx=(10, 5))
        self.speed_var = tk.StringVar(value="1.0x")
        speed_combo = ttk.Combobox(controls_subframe, textvariable=self.speed_var,
                                  values=["0.25x", "0.5x", "1.0x", "2.0x", "4.0x"],
                                  width=8, state="readonly")
        speed_combo.pack(side=tk.LEFT)
        speed_combo.bind('<<ComboboxSelected>>', self.on_speed_change)

        # Zoom control
        ttk.Label(controls_subframe, text="Zoom:").pack(side=tk.LEFT, padx=(10, 5))
        self.zoom_var = tk.StringVar(value=str(self.map_zoom))
        zoom_combo = ttk.Combobox(controls_subframe, textvariable=self.zoom_var,
                                 values=["13", "14", "15", "16", "17", "18"],
                                 width=5, state="readonly")
        zoom_combo.pack(side=tk.LEFT)
        zoom_combo.bind('<<ComboboxSelected>>', self.on_zoom_change)

        # Progress bar
        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Scale(progress_frame, from_=0, to=100,
                                     variable=self.progress_var, orient=tk.HORIZONTAL,
                                     command=self.on_progress_change)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

        # Info display
        info_frame = ttk.Frame(control_frame)
        info_frame.pack(fill=tk.X, pady=(5, 0))

        self.info_label = ttk.Label(info_frame, text="Ready to play")
        self.info_label.pack()

        # Initialize map
        self.setup_map()

    def setup_map(self):
        """Initialize the map display"""
        self.ax.clear()

        # Set map bounds
        map_range = 0.01
        if self.map_extent:
            # Use satellite image extent
            self.ax.set_xlim(self.map_extent[0], self.map_extent[1])
            self.ax.set_ylim(self.map_extent[2], self.map_extent[3])
        else:
            # Fallback to center-based bounds
            self.ax.set_xlim(self.map_center_lon - map_range, self.map_center_lon + map_range)
            self.ax.set_ylim(self.map_center_lat - map_range, self.map_center_lat + map_range)

        # Display satellite imagery as background with better quality settings
        if self.satellite_image and self.map_extent:
            print(f"SETUP_MAP: Displaying satellite image: {self.satellite_image.size}, extent: {self.map_extent}")
            self.ax.imshow(self.satellite_image, extent=self.map_extent, aspect='auto',
                          alpha=0.9, interpolation='bilinear', resample=True)
        else:
            print(f"SETUP_MAP: No satellite image to display: image={self.satellite_image is not None}, extent={self.map_extent}")

        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_title('Drone Position and Orientation - Satellite View')
        self.ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        self.ax.set_aspect('equal')

        # Plot flight path (only valid coordinates)
        if self.metadata:
            valid_entries = [entry for entry in self.metadata
                           if -90 < entry['latitude'] < 90 and -180 < entry['longitude'] < 180]
            if valid_entries:
                lats = [entry['latitude'] for entry in valid_entries]
                lons = [entry['longitude'] for entry in valid_entries]
                self.ax.plot(lons, lats, 'cyan', alpha=0.7, linewidth=2, label='Flight path')

        self.map_canvas.draw()

    def get_current_metadata(self):
        """Get metadata for current video frame"""
        if not self.metadata or self.current_frame >= len(self.metadata):
            return None

        # Simple frame-to-metadata mapping (assuming synchronized)
        metadata_index = min(self.current_frame, len(self.metadata) - 1)
        return self.metadata[metadata_index]

    def update_map(self, metadata_entry):
        """Update map with current drone position and orientation"""
        if not metadata_entry:
            return

        # Skip invalid coordinates
        if not (-90 < metadata_entry['latitude'] < 90 and -180 < metadata_entry['longitude'] < 180):
            return

        # Check if we need to update satellite tiles for this position
        current_lat = metadata_entry['latitude']
        current_lon = metadata_entry['longitude']
        self.update_satellite_tiles(current_lat, current_lon)

        self.ax.clear()

        # Set map bounds - scale viewport based on zoom level for appropriate field of view
        # Higher zoom = smaller viewport for more detail, lower zoom = larger viewport
        base_view_range = 0.01  # Base range for zoom level 16 (current default)
        zoom_factor = 2 ** (self.map_zoom - 16)  # Each zoom level doubles the detail
        view_range = base_view_range / zoom_factor  # Smaller viewport at higher zoom
        if self.map_extent:
            # Center view around drone position but constrained by available tiles
            min_lon = max(self.map_extent[0], current_lon - view_range)
            max_lon = min(self.map_extent[1], current_lon + view_range)
            min_lat = max(self.map_extent[2], current_lat - view_range)
            max_lat = min(self.map_extent[3], current_lat + view_range)

            self.ax.set_xlim(min_lon, max_lon)
            self.ax.set_ylim(min_lat, max_lat)
        else:
            # Fallback to drone-centered bounds
            self.ax.set_xlim(current_lon - view_range, current_lon + view_range)
            self.ax.set_ylim(current_lat - view_range, current_lat + view_range)

        # Display satellite imagery as background with better quality settings
        if self.satellite_image and self.map_extent:
            print(f"UPDATE_MAP: Displaying satellite image: {self.satellite_image.size}, extent: {self.map_extent}")
            self.ax.imshow(self.satellite_image, extent=self.map_extent, aspect='auto',
                          alpha=0.9, interpolation='bilinear', resample=True)
        else:
            print(f"UPDATE_MAP: No satellite image to display: image={self.satellite_image is not None}, extent={self.map_extent}")

        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_title('Drone Position and Orientation - Satellite View')
        self.ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        self.ax.set_aspect('equal')

        # Plot full flight path (only valid coordinates)
        valid_entries = [entry for entry in self.metadata
                        if -90 < entry['latitude'] < 90 and -180 < entry['longitude'] < 180]
        if valid_entries:
            lats = [entry['latitude'] for entry in valid_entries]
            lons = [entry['longitude'] for entry in valid_entries]
            self.ax.plot(lons, lats, 'cyan', alpha=0.5, linewidth=1, label='Flight path')

            # Plot path up to current position
            current_index = self.current_frame
            if current_index > 0:
                # Get valid entries up to current frame
                past_valid = [entry for i, entry in enumerate(self.metadata[:current_index+1])
                             if -90 < entry['latitude'] < 90 and -180 < entry['longitude'] < 180]
                if past_valid:
                    past_lats = [entry['latitude'] for entry in past_valid]
                    past_lons = [entry['longitude'] for entry in past_valid]
                    self.ax.plot(past_lons, past_lats, 'lime', linewidth=3, label='Completed path')

        # Current position
        current_lat = metadata_entry['latitude']
        current_lon = metadata_entry['longitude']

        # Draw current position marker
        self.ax.plot(current_lon, current_lat, 'red', marker='o', markersize=12, markeredgecolor='white', markeredgewidth=2, label='Current position')

        # Draw orientation indicator (simplified)
        yaw = metadata_entry['yaw']
        gimbal_azimuth = metadata_entry['gimbal_azimuth']

        # Convert yaw to direction vector
        direction_rad = math.radians(-yaw)  # Negative for map coordinate system
        arrow_length = 0.002

        end_lon = current_lon + arrow_length * math.sin(direction_rad)
        end_lat = current_lat + arrow_length * math.cos(direction_rad)

        self.ax.annotate('', xy=(end_lon, end_lat), xytext=(current_lon, current_lat),
                        arrowprops=dict(arrowstyle='->', color='yellow', lw=3),
                        label='Heading')

        # Draw field of view cone (simplified)
        hfov = metadata_entry['hfov']
        cone_length = 0.003
        cone_angle = math.radians(hfov / 2)

        # Left edge of FOV cone
        left_angle = direction_rad - cone_angle
        left_lon = current_lon + cone_length * math.sin(left_angle)
        left_lat = current_lat + cone_length * math.cos(left_angle)

        # Right edge of FOV cone
        right_angle = direction_rad + cone_angle
        right_lon = current_lon + cone_length * math.sin(right_angle)
        right_lat = current_lat + cone_length * math.cos(right_angle)

        # Draw FOV cone
        cone_x = [current_lon, left_lon, right_lon, current_lon]
        cone_y = [current_lat, left_lat, right_lat, current_lat]
        self.ax.plot(cone_x, cone_y, 'yellow', alpha=0.8, linewidth=2, linestyle='--')
        self.ax.fill(cone_x, cone_y, 'yellow', alpha=0.2)

        self.ax.legend()
        self.map_canvas.draw()

    def update_frame(self):
        """Update video frame and corresponding map"""
        if not self.cap:
            return

        # Set video position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()

        if ret:
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize frame for display
            height, width = frame_rgb.shape[:2]
            max_width = 640
            if width > max_width:
                scale = max_width / width
                new_width = max_width
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))

            # Convert to PIL Image and then to PhotoImage
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)

            # Update video display
            self.video_label.configure(image=photo)
            self.video_label.image = photo  # Keep a reference

            # Update map with current metadata
            metadata = self.get_current_metadata()
            if metadata:
                self.update_map(metadata)

            # Update progress bar
            progress = (self.current_frame / max(self.frame_count - 1, 1)) * 100
            self.progress_var.set(progress)

            # Update info display
            if metadata:
                # Calculate current view range for display
                base_view_range = 0.01  # Updated base for zoom 16
                zoom_factor = 2 ** (self.map_zoom - 16)
                view_range = base_view_range / zoom_factor
                view_size_km = view_range * 111  # Rough conversion to km

                info_text = (f"Frame: {self.current_frame}/{self.frame_count} | "
                           f"Position: {metadata['latitude']:.6f}, {metadata['longitude']:.6f} | "
                           f"Altitude: {metadata['altitude']:.1f}m | "
                           f"Yaw: {metadata['yaw']:.1f}° | "
                           f"Zoom: {self.map_zoom} | "
                           f"View: {view_size_km:.1f}km")
                self.info_label.configure(text=info_text)

    def toggle_playback(self):
        """Toggle play/pause"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_video()

    def play_video(self):
        """Play video with map synchronization"""
        if not self.is_playing:
            return

        start_time = time.time()
        start_frame = self.current_frame

        def update_loop():
            while self.is_playing and self.current_frame < self.frame_count - 1:
                elapsed_time = time.time() - start_time
                expected_frame = start_frame + int(elapsed_time * self.frame_rate * self.playback_speed)

                if expected_frame > self.current_frame:
                    self.current_frame = min(expected_frame, self.frame_count - 1)
                    self.root.after(0, self.update_frame)

                time.sleep(0.01)  # Small delay to prevent excessive CPU usage

            self.is_playing = False

        threading.Thread(target=update_loop, daemon=True).start()

    def reset_video(self):
        """Reset video to beginning"""
        self.is_playing = False
        self.current_frame = 0
        self.update_frame()

    def on_speed_change(self, event=None):
        """Handle playback speed change"""
        speed_text = self.speed_var.get()
        self.playback_speed = float(speed_text.replace('x', ''))

    def on_zoom_change(self, event=None):
        """Handle zoom level change"""
        new_zoom = int(self.zoom_var.get())
        if new_zoom != self.map_zoom:
            self.map_zoom = new_zoom
            print(f"Zoom level changed to: {self.map_zoom}")

            # Smart cache clearing - only remove tiles from different zoom levels
            keys_to_remove = [key for key in self.tile_cache.keys() if key[2] != self.map_zoom]
            for key in keys_to_remove:
                del self.tile_cache[key]
            self.current_tile_center = None

            # Force tile update with new zoom level
            current_metadata = self.get_current_metadata()
            if current_metadata and (-90 < current_metadata['latitude'] < 90 and -180 < current_metadata['longitude'] < 180):
                self.update_satellite_tiles(current_metadata['latitude'], current_metadata['longitude'], force_update=True)
                self.update_map(current_metadata)  # Refresh the map display

    def on_progress_change(self, value):
        """Handle progress bar change"""
        if not self.is_playing:  # Only allow manual seeking when not playing
            progress = float(value)
            self.current_frame = int((progress / 100) * (self.frame_count - 1))
            self.update_frame()

    def run(self):
        """Start the application"""
        # Initialize with first frame
        self.update_frame()

        # Start GUI main loop
        self.root.mainloop()

        # Cleanup
        if self.cap:
            self.cap.release()
        self.download_executor.shutdown(wait=False)


def main():
    if len(sys.argv) != 3:
        print("Usage: python drone_video_map_viewer.py <video_path> <metadata_csv_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    metadata_path = sys.argv[2]

    # Check if files exist
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found: {metadata_path}")
        sys.exit(1)

    print("Starting Drone Video & Map Viewer...")
    print(f"Video: {video_path}")
    print(f"Metadata: {metadata_path}")

    try:
        viewer = DroneVideoMapViewer(video_path, metadata_path)
        viewer.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()