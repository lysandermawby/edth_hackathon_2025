import React, { useState, useEffect, useCallback, useMemo } from "react";
import {
  HiVideoCamera,
  HiMap,
  HiCog,
  HiLink,
  HiChartBar,
  HiWifi,
  HiLocationMarker,
} from "react-icons/hi";
import VideoCanvas from "./VideoCanvas";
import DroneMapViewer from "./DroneMapViewer";
import DroneStatusDisplay from "./DroneStatusDisplay";
import type {
  FrameDetections,
  DroneMetadata,
  SessionWithMetadata,
} from "./types";

interface VideoMapViewerProps {
  session: SessionWithMetadata;
  trackingData: FrameDetections[];
  videoSrc: string;
}

const VideoMapViewer: React.FC<VideoMapViewerProps> = ({
  session,
  trackingData,
  videoSrc,
}) => {
  const [currentVideoTime, setCurrentVideoTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMapSynced, setIsMapSynced] = useState(true);
  const [mapFrame, setMapFrame] = useState(0);

  // Find the best matching metadata entry based on video progress
  const getCurrentMetadataIndex = useCallback(
    (videoTime: number, metadata: DroneMetadata[]): number => {
      if (duration === 0 || metadata.length === 0) return 0;

      // Calculate the progress through the video (0 to 1)
      const videoProgress = Math.min(videoTime / duration, 1);

      // Map video progress to metadata array index
      const metadataIndex = Math.floor(videoProgress * (metadata.length - 1));

      // Ensure we don't exceed array bounds
      return Math.max(0, Math.min(metadataIndex, metadata.length - 1));
    },
    [duration]
  );

  // Use enhanced telemetry if available, otherwise fall back to legacy metadata or sample data
  const actualMetadata: DroneMetadata[] = useMemo(() => {
    // Priority 1: Enhanced telemetry (most accurate with camera footprints)
    if (
      session.enhanced_telemetry?.telemetry &&
      session.enhanced_telemetry.telemetry.length > 0
    ) {
      return session.enhanced_telemetry.telemetry.map((point) => ({
        timestamp: point.timestamp,
        latitude: point.latitude,
        longitude: point.longitude,
        altitude: point.altitude,
        roll: point.roll,
        pitch: point.pitch,
        yaw: point.yaw,
        gimbal_elevation: point.gimbal_elevation,
        gimbal_azimuth: point.gimbal_azimuth,
        vfov: point.vfov,
        hfov: point.hfov,
      }));
    }

    // Priority 2: Legacy metadata
    if (session.metadata && session.metadata.length > 0) {
      return session.metadata;
    }

    // Priority 3: Generate sample data proportional to video duration
    return Array.from({ length: 50 }, (_, i) => ({
      timestamp: i * (duration / 50),
      latitude: 48.1351 + (i / 50) * 0.01,
      longitude: 11.582 + (i / 50) * 0.01,
      altitude: 100 + Math.sin(i / 10) * 20,
      roll: Math.sin(i / 8) * 5,
      pitch: -15 + Math.cos(i / 6) * 3,
      yaw: 45 + (i / 50) * 180,
      gimbal_elevation: -30 + Math.sin(i / 4) * 10,
      gimbal_azimuth: Math.cos(i / 5) * 15,
      vfov: 60,
      hfov: 90,
    }));
  }, [session.enhanced_telemetry, session.metadata, duration]);

  // Update map frame when video time changes
  useEffect(() => {
    if (isMapSynced) {
      const metadataIndex = getCurrentMetadataIndex(
        currentVideoTime,
        actualMetadata
      );
      setMapFrame(metadataIndex);
    }
  }, [currentVideoTime, isMapSynced, getCurrentMetadataIndex, actualMetadata]);

  // Handle video time updates
  const handleVideoTimeUpdate = useCallback((time: number) => {
    setCurrentVideoTime(time);
  }, []);

  // Handle duration loaded
  const handleDurationLoad = useCallback((dur: number) => {
    setDuration(dur);
  }, []);

  const formatTime = (time: number): string => {
    if (!Number.isFinite(time)) return "0:00";
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  };

  const hasRealGpsData = session.metadata && session.metadata.length > 0;
  const hasEnhancedTelemetry =
    session.enhanced_telemetry?.telemetry &&
    session.enhanced_telemetry.telemetry.length > 0;

  return (
    <div className="space-y-6">
      {/* Sync Control & Telemetry Panel */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Sync Controls */}
        <div className="xl:col-span-1">
          <div className="card">
            <div className="card-header">
              <div className="flex items-center gap-3">
                <div className="w-6 h-6 bg-secondary-600 rounded flex items-center justify-center">
                  <HiCog className="text-white text-sm" />
                </div>
                <h4 className="font-semibold text-tactical-text">
                  Synchronization Controls
                </h4>
              </div>
            </div>
            <div className="p-4">
              <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                <div className="flex items-center gap-6">
                  <div className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      id="sync-map"
                      checked={isMapSynced}
                      onChange={(e) => setIsMapSynced(e.target.checked)}
                      className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
                    />
                    <label
                      htmlFor="sync-map"
                      className="text-sm font-medium text-tactical-text flex items-center gap-2"
                    >
                      <HiLink className="w-4 h-4" />
                      Auto-sync with video
                    </label>
                  </div>

                  {!isMapSynced && (
                    <div className="flex items-center gap-3">
                      <label className="text-sm text-tactical-muted">
                        Manual frame:
                      </label>
                      <input
                        type="range"
                        min={0}
                        max={Math.max(actualMetadata.length - 1, 0)}
                        value={mapFrame}
                        onChange={(e) => setMapFrame(Number(e.target.value))}
                        className="w-32 h-2 bg-neutral-200 rounded-lg appearance-none cursor-pointer"
                      />
                      <span className="text-sm font-mono text-neutral-700 min-w-[60px] px-2 py-1 bg-neutral-100 rounded">
                        {mapFrame}
                      </span>
                    </div>
                  )}
                </div>

                <div className="text-sm text-tactical-muted space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">Video:</span>
                    <span className="font-mono">
                      {formatTime(currentVideoTime)} / {formatTime(duration)}
                    </span>
                    <span className="px-2 py-1 bg-primary-100 text-primary-700 rounded text-xs">
                      {duration > 0
                        ? ((currentVideoTime / duration) * 100).toFixed(1)
                        : 0}
                      %
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium">Map:</span>
                    <span className="font-mono">
                      {mapFrame}/{actualMetadata.length - 1}
                    </span>
                    {actualMetadata.length > 0 &&
                      mapFrame < actualMetadata.length && (
                        <span className="px-2 py-1 bg-secondary-100 text-secondary-700 rounded text-xs">
                          GPS#{mapFrame} @{" "}
                          {actualMetadata[mapFrame].timestamp.toFixed(2)}s
                        </span>
                      )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Drone Telemetry */}
        <div className="xl:col-span-2">
          <DroneStatusDisplay
            metadata={
              hasRealGpsData && mapFrame < actualMetadata.length
                ? actualMetadata[mapFrame]
                : undefined
            }
          />
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Video Player Panel */}
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 bg-primary-600 rounded flex items-center justify-center">
                <HiVideoCamera className="text-white text-sm" />
              </div>
              <div>
                <h3 className="font-semibold text-tactical-text">
                  Video Analysis
                </h3>
                <p className="text-xs text-tactical-muted">
                  {session.video_path.split("/").pop()}
                </p>
              </div>
            </div>
          </div>
          <div className="card-body space-y-4">
            {/* Video Metadata */}
            <div className="grid grid-cols-2 gap-4 p-3 bg-neutral-50 rounded-lg">
              <div className="text-center">
                <div className="text-lg font-bold text-primary-600">
                  {trackingData.length}
                </div>
                <div className="text-xs text-tactical-muted">Frames</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-success-600">
                  {trackingData.reduce(
                    (acc, frame) => acc + frame.objects.length,
                    0
                  )}
                </div>
                <div className="text-xs text-tactical-muted">Detections</div>
              </div>
            </div>

            {/* Enhanced Telemetry Info */}
            {hasEnhancedTelemetry && (
              <div className="p-3 bg-success-50 border border-success-200 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <HiWifi className="text-success-600 w-4 h-4" />
                  <span className="text-sm font-medium text-success-800">
                    Enhanced Telemetry Active
                  </span>
                </div>
                <div className="text-xs text-success-700">
                  {session.enhanced_telemetry!.telemetry.length} GPS points •
                  Camera footprints • Flight analytics
                </div>
                {session.enhanced_telemetry!.analytics && (
                  <div className="mt-2 text-xs text-success-700 space-y-1">
                    <div>
                      Duration:{" "}
                      {session.enhanced_telemetry!.analytics.flight_duration.toFixed(
                        1
                      )}
                      s • Distance:{" "}
                      {session.enhanced_telemetry!.analytics.total_distance.toFixed(
                        0
                      )}
                      m • Speed:{" "}
                      {session.enhanced_telemetry!.analytics.avg_speed.toFixed(
                        1
                      )}
                      m/s
                    </div>
                    <div>
                      Coverage:{" "}
                      {(
                        session.enhanced_telemetry!.analytics.coverage_area /
                        10000
                      ).toFixed(1)}{" "}
                      hectares • Altitude:{" "}
                      {session.enhanced_telemetry!.analytics.min_altitude.toFixed(
                        0
                      )}
                      -
                      {session.enhanced_telemetry!.analytics.max_altitude.toFixed(
                        0
                      )}
                      m
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Basic GPS Info */}
            {!hasEnhancedTelemetry &&
              session.metadata &&
              session.metadata.length > 0 && (
                <div className="p-3 bg-primary-50 border border-primary-200 rounded-lg">
                  <div className="flex items-center gap-2 mb-1">
                    <HiLocationMarker className="text-primary-600 w-4 h-4" />
                    <span className="text-sm font-medium text-primary-800">
                      GPS Data Available
                    </span>
                  </div>
                  <div className="text-xs text-primary-700">
                    {session.metadata.length} GPS points •
                    {session.metadata[0].timestamp.toFixed(2)}s -{" "}
                    {session.metadata[
                      session.metadata.length - 1
                    ].timestamp.toFixed(2)}
                    s
                  </div>
                </div>
              )}

            <VideoCanvas
              videoSrc={videoSrc}
              trackingData={trackingData}
              onTimeUpdate={handleVideoTimeUpdate}
              onDurationLoad={handleDurationLoad}
            />
          </div>
        </div>

        {/* Drone Map Panel */}
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 bg-secondary-600 rounded flex items-center justify-center">
                <HiMap className="text-white text-sm" />
              </div>
              <div>
                <h3 className="font-semibold text-tactical-text">
                  {hasRealGpsData ? "Flight Path & Position" : "Map View"}
                </h3>
                <p className="text-xs text-tactical-muted">
                  {hasRealGpsData
                    ? "Real-time GPS tracking"
                    : "No GPS data available"}
                </p>
              </div>
            </div>
          </div>
          <div className="card-body">
            <div className="aspect-square rounded-xl overflow-hidden border border-neutral-200">
              <DroneMapViewer
                metadata={actualMetadata}
                currentFrame={mapFrame}
                className="w-full h-full"
                enhancedTelemetry={
                  hasEnhancedTelemetry
                    ? session.enhanced_telemetry!.telemetry
                    : undefined
                }
                showFootprints={hasEnhancedTelemetry}
              />
            </div>
          </div>
        </div>
      </div>

    </div>
  );
};

export default VideoMapViewer;
