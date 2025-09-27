import React, { useState, useEffect, useCallback, useMemo } from 'react';
import VideoCanvas from './VideoCanvas';
import DroneMapViewer from './DroneMapViewer';
import type { FrameDetections, DroneMetadata, SessionWithMetadata, EnhancedTelemetryPoint } from './types';

interface VideoMapViewerProps {
  session: SessionWithMetadata;
  trackingData: FrameDetections[];
  videoSrc: string;
}

const VideoMapViewer: React.FC<VideoMapViewerProps> = ({
  session,
  trackingData,
  videoSrc
}) => {
  const [currentVideoTime, setCurrentVideoTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMapSynced, setIsMapSynced] = useState(true);
  const [mapFrame, setMapFrame] = useState(0);

  // Find the best matching metadata entry based on video progress
  const getCurrentMetadataIndex = useCallback((videoTime: number, metadata: DroneMetadata[]): number => {
    if (duration === 0 || metadata.length === 0) return 0;

    // Calculate the progress through the video (0 to 1)
    const videoProgress = Math.min(videoTime / duration, 1);

    // Map video progress to metadata array index
    const metadataIndex = Math.floor(videoProgress * (metadata.length - 1));

    // Ensure we don't exceed array bounds
    return Math.max(0, Math.min(metadataIndex, metadata.length - 1));
  }, [duration]);

  // Use enhanced telemetry if available, otherwise fall back to legacy metadata or sample data
  const actualMetadata: DroneMetadata[] = useMemo(() => {
    // Priority 1: Enhanced telemetry (most accurate with camera footprints)
    if (session.enhanced_telemetry?.telemetry && session.enhanced_telemetry.telemetry.length > 0) {
      return session.enhanced_telemetry.telemetry.map(point => ({
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
        hfov: point.hfov
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
      longitude: 11.5820 + (i / 50) * 0.01,
      altitude: 100 + Math.sin(i / 10) * 20,
      roll: Math.sin(i / 8) * 5,
      pitch: -15 + Math.cos(i / 6) * 3,
      yaw: 45 + (i / 50) * 180,
      gimbal_elevation: -30 + Math.sin(i / 4) * 10,
      gimbal_azimuth: Math.cos(i / 5) * 15,
      vfov: 60,
      hfov: 90
    }));
  }, [session.enhanced_telemetry, session.metadata, duration]);

  // Update map frame when video time changes
  useEffect(() => {
    if (isMapSynced) {
      const metadataIndex = getCurrentMetadataIndex(currentVideoTime, actualMetadata);
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
  const hasEnhancedTelemetry = session.enhanced_telemetry?.telemetry && session.enhanced_telemetry.telemetry.length > 0;

  return (
    <div className="space-y-6">
      {/* Sync Control */}
      <div className="bg-white rounded-lg p-4 shadow-md border border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="sync-map"
                checked={isMapSynced}
                onChange={(e) => setIsMapSynced(e.target.checked)}
                className="rounded"
              />
              <label htmlFor="sync-map" className="text-sm font-medium">
                🔗 Sync Map with Video
              </label>
            </div>

            {!isMapSynced && (
              <div className="flex items-center space-x-2">
                <label className="text-sm">Manual Frame:</label>
                <input
                  type="range"
                  min={0}
                  max={Math.max(actualMetadata.length - 1, 0)}
                  value={mapFrame}
                  onChange={(e) => setMapFrame(Number(e.target.value))}
                  className="w-32"
                />
                <span className="text-sm text-gray-600 min-w-[60px]">
                  {mapFrame}
                </span>
              </div>
            )}
          </div>

          <div className="text-sm text-gray-600">
            Video: {formatTime(currentVideoTime)} / {formatTime(duration)} •
            Progress: {duration > 0 ? ((currentVideoTime / duration) * 100).toFixed(1) : 0}% •
            Map: {mapFrame}/{actualMetadata.length - 1}
            {actualMetadata.length > 0 && mapFrame < actualMetadata.length && (
              <span className="ml-2 text-blue-600">
                @ GPS#{mapFrame} ({actualMetadata[mapFrame].timestamp.toFixed(2)}s)
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Video Player */}
        <div className="bg-white rounded-xl p-6 shadow-xl">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">
            📹 Video with Object Detection
          </h3>
          <div className="space-y-4">
            <div className="text-sm text-gray-600 border-b pb-2">
              <p><strong>Video:</strong> {session.video_path.split('/').pop()}</p>
              <p><strong>Frames:</strong> {trackingData.length}</p>
              <p><strong>FPS:</strong> {session.fps}</p>
              <p>
                <strong>Detections:</strong>{" "}
                {trackingData.reduce((acc, frame) => acc + frame.objects.length, 0)}
              </p>
              {hasEnhancedTelemetry && (
                <p>
                  <strong>📡 Enhanced Telemetry:</strong> {session.enhanced_telemetry!.telemetry.length} points
                  <span className="text-xs ml-2 text-green-600">
                    ✓ Camera footprints, flight analytics
                  </span>
                </p>
              )}
              {!hasEnhancedTelemetry && session.metadata && session.metadata.length > 0 && (
                <p>
                  <strong>📍 GPS Points:</strong> {session.metadata.length}
                  <span className="text-xs ml-2">
                    ({session.metadata[0].timestamp.toFixed(2)}s - {session.metadata[session.metadata.length - 1].timestamp.toFixed(2)}s)
                  </span>
                </p>
              )}
              {hasEnhancedTelemetry && session.enhanced_telemetry!.analytics && (
                <div className="text-xs text-gray-600 mt-2 space-y-1">
                  <p>
                    <strong>Flight:</strong> {session.enhanced_telemetry!.analytics.flight_duration.toFixed(1)}s • 
                    {session.enhanced_telemetry!.analytics.total_distance.toFixed(0)}m distance • 
                    {session.enhanced_telemetry!.analytics.avg_speed.toFixed(1)}m/s avg speed
                  </p>
                  <p>
                    <strong>Coverage:</strong> {(session.enhanced_telemetry!.analytics.coverage_area / 10000).toFixed(1)} hectares • 
                    Alt: {session.enhanced_telemetry!.analytics.min_altitude.toFixed(0)}-{session.enhanced_telemetry!.analytics.max_altitude.toFixed(0)}m
                  </p>
                </div>
              )}
            </div>

            <VideoCanvas
              videoSrc={videoSrc}
              trackingData={trackingData}
              onTimeUpdate={handleVideoTimeUpdate}
              onDurationLoad={handleDurationLoad}
            />
          </div>
        </div>

        {/* Drone Map */}
        <div className="bg-white rounded-xl p-6 shadow-xl">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">
            {hasRealGpsData ? "🗺️ Drone Position & Flight Path" : "🗺️ No GPS data available"}
          </h3>
          <div className="aspect-square">
            <DroneMapViewer
              metadata={actualMetadata}
              currentFrame={mapFrame}
              className="w-full h-full"
              enhancedTelemetry={hasEnhancedTelemetry ? session.enhanced_telemetry!.telemetry : undefined}
              showFootprints={hasEnhancedTelemetry}
            />
          </div>
        </div>
      </div>

      {/* Flight Data Display */}
      {hasRealGpsData && (
        <div className="bg-white rounded-xl p-6 shadow-xl">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">
            📊 Flight Telemetry Data
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {mapFrame < actualMetadata.length && (
              <>
                <div className="text-center p-3 bg-blue-50 rounded-lg">
                  <div className="text-sm text-gray-600">Latitude</div>
                  <div className="font-mono text-lg">
                    {actualMetadata[mapFrame].latitude.toFixed(6)}°
                  </div>
                </div>
                <div className="text-center p-3 bg-green-50 rounded-lg">
                  <div className="text-sm text-gray-600">Longitude</div>
                  <div className="font-mono text-lg">
                    {actualMetadata[mapFrame].longitude.toFixed(6)}°
                  </div>
                </div>
                <div className="text-center p-3 bg-purple-50 rounded-lg">
                  <div className="text-sm text-gray-600">Altitude</div>
                  <div className="font-mono text-lg">
                    {actualMetadata[mapFrame].altitude.toFixed(1)}m
                  </div>
                </div>
                <div className="text-center p-3 bg-yellow-50 rounded-lg">
                  <div className="text-sm text-gray-600">Yaw</div>
                  <div className="font-mono text-lg">
                    {actualMetadata[mapFrame].yaw.toFixed(1)}°
                  </div>
                </div>
                <div className="text-center p-3 bg-red-50 rounded-lg">
                  <div className="text-sm text-gray-600">Pitch</div>
                  <div className="font-mono text-lg">
                    {actualMetadata[mapFrame].pitch.toFixed(1)}°
                  </div>
                </div>
                <div className="text-center p-3 bg-orange-50 rounded-lg">
                  <div className="text-sm text-gray-600">Roll</div>
                  <div className="font-mono text-lg">
                    {actualMetadata[mapFrame].roll.toFixed(1)}°
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="bg-blue-50 rounded-xl p-6 border border-blue-200">
        <h3 className="text-lg font-semibold text-blue-700 mb-3">
          🎮 How to Use
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-700">
          <div>
            <h4 className="font-medium mb-2">Video Controls</h4>
            <ul className="space-y-1">
              <li>• Click Play/Pause to control video playback</li>
              <li>• Use the timeline slider to seek</li>
              <li>• Hover over detected objects for details</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium mb-2">Map Controls</h4>
            <ul className="space-y-1">
              <li>• Toggle sync to link map with video</li>
              <li>• Adjust zoom level in map controls</li>
              <li>• Toggle flight path and FOV display</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoMapViewer;
