import React, { useState, useEffect, useCallback, useMemo } from "react";
import {
  HiVideoCamera,
  HiMap,
  HiCog,
  HiLink,
  HiChartBar,
  HiWifi,
  HiLocationMarker,
  HiRefresh,
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
  onRegenerateDetections?: () => void;
  isGeneratingDetections?: boolean;
  generationMessage?: string | null;
  generationError?: string | null;
}

const VideoMapViewer: React.FC<VideoMapViewerProps> = ({
  session,
  trackingData,
  videoSrc,
  onRegenerateDetections,
  isGeneratingDetections = false,
  generationMessage,
  generationError,
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
      {/* Two Column Layout: Session/Sync Controls + Telemetry */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Left Column: Session Info + Sync Controls */}
        <div className="space-y-6">
          {/* Session Details */}
          <div className="card">
            <div className="card-header">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-6 h-6 bg-primary-600 rounded flex items-center justify-center">
                    <HiVideoCamera className="text-white text-sm" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-tactical-text">
                      Session #{session.session_id}
                    </h3>
                    <p className="text-xs text-tactical-muted">
                      {session.video_path.split("/").pop()}
                    </p>
                  </div>
                </div>
                {onRegenerateDetections && (
                  <button
                    onClick={onRegenerateDetections}
                    disabled={isGeneratingDetections}
                    className={`btn flex items-center gap-2 ${
                      isGeneratingDetections
                        ? "btn-secondary cursor-not-allowed"
                        : "btn-primary"
                    }`}
                  >
                    {isGeneratingDetections ? (
                      <>
                        <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full"></div>
                        Generating...
                      </>
                    ) : (
                      <>
                        <HiRefresh className="w-4 h-4" />
                        Regenerate Detections
                      </>
                    )}
                  </button>
                )}
              </div>
            </div>
            <div className="p-4">
              {/* Status Messages */}
              {generationMessage && (
                <div className="mb-4 p-3 bg-success-50 border border-success-200 text-success-800 rounded-lg text-sm">
                  <div className="font-medium">Success</div>
                  <div className="text-xs mt-1">{generationMessage}</div>
                </div>
              )}
              {generationError && (
                <div className="mb-4 p-3 bg-accent-50 border border-accent-200 text-accent-800 rounded-lg text-sm">
                  <div className="font-medium">Error</div>
                  <div className="text-xs mt-1">{generationError}</div>
                </div>
              )}

              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-lg font-bold text-primary-600">
                    {trackingData.length}
                  </div>
                  <div className="text-xs text-tactical-muted">
                    Total Frames
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-success-600">
                    {trackingData.filter((f) => f.objects.length > 0).length}
                  </div>
                  <div className="text-xs text-tactical-muted">
                    With Detections
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-neon-cyan">
                    {trackingData.reduce((acc, f) => acc + f.objects.length, 0)}
                  </div>
                  <div className="text-xs text-tactical-muted">
                    Total Objects
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-warning-600">
                    {
                      new Set(
                        trackingData.flatMap((f) =>
                          f.objects.map((o) => o.tracker_id).filter(Boolean)
                        )
                      ).size
                    }
                  </div>
                  <div className="text-xs text-tactical-muted">
                    Objects Tracked
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-secondary-600">
                    {session.fps || "N/A"}
                  </div>
                  <div className="text-xs text-tactical-muted">FPS</div>
                </div>
              </div>

              {trackingData.length === 0 && (
                <div className="text-center py-4 text-tactical-muted mt-4">
                  <HiChartBar className="text-neutral-400 text-3xl mb-2 mx-auto" />
                  <div className="font-medium">No detection data available</div>
                  <div className="text-sm mt-1">
                    Click "Regenerate" to process this session
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Sync Controls */}
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
              {/* 3-Column Layout */}
              <div className="grid grid-cols-3 gap-4">
                {/* Column 1: Sync Controls */}
                <div className="space-y-3">
                  <button
                    onClick={() => setIsMapSynced(!isMapSynced)}
                    className={`w-full px-3 py-2 text-sm font-medium rounded transition-all duration-200 flex items-center justify-center gap-2 ${
                      isMapSynced
                        ? "bg-primary-600 text-white border border-primary-600 shadow-md"
                        : "bg-neutral-100 text-neutral-600 border border-neutral-300 hover:bg-neutral-200"
                    }`}
                  >
                    <HiLink
                      className={`w-3 h-3 ${
                        isMapSynced ? "text-white" : "text-neutral-500"
                      }`}
                    />
                    {isMapSynced ? "Auto-sync ON" : "Auto-sync OFF"}
                  </button>

                  {!isMapSynced && (
                    <div className="space-y-2">
                      <label className="text-xs text-tactical-muted block">
                        Manual frame:
                      </label>
                      <input
                        type="range"
                        min={0}
                        max={Math.max(actualMetadata.length - 1, 0)}
                        value={mapFrame}
                        onChange={(e) => setMapFrame(Number(e.target.value))}
                        className="w-full h-2 bg-neutral-200 rounded-lg appearance-none cursor-pointer"
                      />
                      <span className="text-xs font-mono text-neutral-700 px-2 py-1 bg-neutral-100 rounded block text-center">
                        {mapFrame}
                      </span>
                    </div>
                  )}
                </div>

                {/* Column 2: Video Info */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="text-xs text-tactical-muted font-medium">
                      VIDEO
                    </div>
                    <div className="text-sm font-mono text-tactical-text">
                      {formatTime(currentVideoTime)} / {formatTime(duration)}
                    </div>
                  </div>
                  <div className="px-2 py-1 bg-primary-100 text-primary-700 rounded text-xs text-center">
                    {duration > 0
                      ? ((currentVideoTime / duration) * 100).toFixed(1)
                      : 0}
                    %
                  </div>
                </div>

                {/* Column 3: Map Info */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="text-xs text-tactical-muted font-medium">
                      MAP
                    </div>
                    <div className="text-sm font-mono text-tactical-text">
                      {mapFrame}/{actualMetadata.length - 1}
                    </div>
                  </div>
                  {actualMetadata.length > 0 &&
                    mapFrame < actualMetadata.length && (
                      <div className="px-2 py-1 bg-secondary-100 text-secondary-700 rounded text-xs text-center">
                        GPS#{mapFrame} @{" "}
                        {actualMetadata[mapFrame].timestamp.toFixed(1)}s
                      </div>
                    )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Drone Telemetry */}
        <div className="h-full">
          <DroneStatusDisplay
            metadata={
              hasRealGpsData && mapFrame < actualMetadata.length
                ? actualMetadata[mapFrame]
                : undefined
            }
            hasEnhancedTelemetry={hasEnhancedTelemetry}
            session={session}
            className="h-full"
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
            <div className="aspect-[4/3] rounded-xl overflow-hidden border border-neutral-200">
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
