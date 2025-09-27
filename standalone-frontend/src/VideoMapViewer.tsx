import React, { useState, useEffect, useRef, useCallback } from 'react';
import VideoCanvas from './VideoCanvas';
import DroneMapViewer from './DroneMapViewer';
import type { FrameDetections, DroneMetadata, SessionWithMetadata } from './types';

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
  // const videoRef = useRef<HTMLVideoElement | null>(null); // Removed unused ref

  // Calculate current frame based on video time and FPS
  const getCurrentFrame = useCallback((time: number): number => {
    if (!session.fps || duration === 0) return 0;
    return Math.floor(time * session.fps);
  }, [session.fps, duration]);

  // Update map frame when video time changes
  useEffect(() => {
    if (isMapSynced) {
      const frame = getCurrentFrame(currentVideoTime);
      setMapFrame(frame);
    }
  }, [currentVideoTime, isMapSynced, getCurrentFrame]);

  // Handle video time updates
  const handleVideoTimeUpdate = useCallback((time: number) => {
    setCurrentVideoTime(time);
  }, []);

  // Handle duration loaded
  const handleDurationLoad = useCallback((dur: number) => {
    setDuration(dur);
  }, []);

  // Enhanced VideoCanvas that exposes time updates
  const EnhancedVideoCanvas = () => {
    const videoCanvasRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
      const videoElement = videoCanvasRef.current?.querySelector('video');
      if (!videoElement) return;

      const handleTimeUpdate = () => {
        handleVideoTimeUpdate(videoElement.currentTime);
      };

      const handleLoadedMetadata = () => {
        handleDurationLoad(videoElement.duration);
      };

      videoElement.addEventListener('timeupdate', handleTimeUpdate);
      videoElement.addEventListener('loadedmetadata', handleLoadedMetadata);

      return () => {
        videoElement.removeEventListener('timeupdate', handleTimeUpdate);
        videoElement.removeEventListener('loadedmetadata', handleLoadedMetadata);
      };
    }, []);

    return (
      <div ref={videoCanvasRef}>
        <VideoCanvas videoSrc={videoSrc} trackingData={trackingData} />
      </div>
    );
  };

  const formatTime = (time: number): string => {
    if (!Number.isFinite(time)) return "0:00";
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  };

  // Use actual metadata from session, or fallback to sample data
  const actualMetadata: DroneMetadata[] = session.metadata && session.metadata.length > 0
    ? session.metadata
    : [
        {
          timestamp: 0,
          latitude: 48.1351,
          longitude: 11.5820,
          altitude: 100,
          roll: 0,
          pitch: -15,
          yaw: 45,
          gimbal_elevation: -30,
          gimbal_azimuth: 0,
          vfov: 60,
          hfov: 90
        },
        {
          timestamp: 1,
          latitude: 48.1352,
          longitude: 11.5821,
          altitude: 105,
          roll: 2,
          pitch: -12,
          yaw: 47,
          gimbal_elevation: -28,
          gimbal_azimuth: 5,
          vfov: 60,
          hfov: 90
        }
      ];

  const hasRealGpsData = session.metadata && session.metadata.length > 0;

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
                  max={Math.max(trackingData.length - 1, 100)}
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
            Frame: {getCurrentFrame(currentVideoTime)} •
            Map Frame: {mapFrame}
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
            </div>

            <EnhancedVideoCanvas />
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