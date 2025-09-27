export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface TrackingObject {
  tracker_id?: number;
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: BoundingBox;
  center?: { x: number; y: number };
  is_reidentified?: boolean;
}

export interface FrameDetections {
  frame_number: number;
  timestamp: number;
  objects: TrackingObject[];
}

export interface Session {
  session_id: number;
  video_path: string;
  start_time: string;
  end_time?: string;
  total_frames?: number;
  fps: number;
}

export interface DetectionData {
  id: number;
  session_id: number;
  frame_number: number;
  timestamp: number;
  tracker_id?: number;
  class_id: number;
  class_name: string;
  confidence: number;
  bbox_x1: number;
  bbox_y1: number;
  bbox_x2: number;
  bbox_y2: number;
  center_x: number;
  center_y: number;
}

export interface DroneMetadata {
  timestamp: number;
  latitude: number;
  longitude: number;
  altitude: number;
  roll: number;
  pitch: number;
  yaw: number;
  gimbal_elevation: number;
  gimbal_azimuth: number;
  vfov: number;
  hfov: number;
}

export interface EnhancedTelemetryPoint {
  timestamp: number;
  latitude: number;
  longitude: number;
  altitude: number;
  roll: number;
  pitch: number;
  yaw: number;
  gimbal_elevation: number;
  gimbal_azimuth: number;
  center_latitude: number;
  center_longitude: number;
  slant_range: number;
  hfov: number;
  vfov: number;
  footprint?: number[][];  // [lon, lat] polygon coordinates
}

export interface FlightAnalytics {
  total_distance: number;
  max_altitude: number;
  min_altitude: number;
  avg_speed: number;
  max_speed: number;
  flight_duration: number;
  coverage_area: number;
  stability_metrics: {
    roll_variance: number;
    pitch_variance: number;
    yaw_variance: number;
  };
}

export interface EnhancedTelemetryData {
  telemetry: EnhancedTelemetryPoint[];
  analytics: FlightAnalytics | null;
}

export interface ReidentificationStats {
  total_reidentifications: number;
  active_tracks: number;
  lost_tracks: number;
  success_rate: number;
}

export interface SessionWithMetadata extends Session {
  metadata?: DroneMetadata[];
  enhanced_telemetry?: EnhancedTelemetryData;
}
