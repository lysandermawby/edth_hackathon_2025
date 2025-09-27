import { useState, useEffect } from "react";
import {
  HiFolder,
  HiRefresh,
  HiChartBar,
  HiLocationMarker,
  HiServer,
} from "react-icons/hi";
import { RiLiveLine, RiDashboardFill, RiFileListLine } from "react-icons/ri";
import VideoMapViewer from "./VideoMapViewer";
import RealtimeVideoCanvas from "./RealtimeVideoCanvas";
import type {
  Session,
  FrameDetections,
  DetectionData,
  SessionWithMetadata,
} from "./types";

const convertDetectionsToFrames = (
  detections: DetectionData[],
  fps?: number
): FrameDetections[] => {
  const frameMap = new Map<number, FrameDetections>();

  detections.forEach((detection) => {
    const assumedFps = Math.max(fps ?? 30, 1);
    const fallbackTimestamp = detection.frame_number / assumedFps;
    const timestamp = Number.isFinite(detection.timestamp)
      ? detection.timestamp
      : fallbackTimestamp;

    if (!frameMap.has(detection.frame_number)) {
      frameMap.set(detection.frame_number, {
        frame_number: detection.frame_number,
        timestamp,
        objects: [],
      });
    }

    const frame = frameMap.get(detection.frame_number)!;

    frame.objects.push({
      tracker_id: detection.tracker_id || undefined,
      class_id: detection.class_id,
      class_name: detection.class_name,
      confidence: detection.confidence,
      bbox: {
        x1: detection.bbox_x1,
        y1: detection.bbox_y1,
        x2: detection.bbox_x2,
        y2: detection.bbox_y2,
      },
      center: {
        x: detection.center_x,
        y: detection.center_y,
      },
    });

    if (!Number.isFinite(frame.timestamp) || frame.timestamp > timestamp) {
      frame.timestamp = timestamp;
    }
  });

  const frames = Array.from(frameMap.values()).sort(
    (a, b) => a.frame_number - b.frame_number
  );

  if (frames.length > 0) {
    const minTimestamp = Math.min(...frames.map((frame) => frame.timestamp));
    if (Number.isFinite(minTimestamp) && minTimestamp > 0) {
      frames.forEach((frame) => {
        frame.timestamp = frame.timestamp - minTimestamp;
      });
    }
  }

  return frames;
};

function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedSession, setSelectedSession] =
    useState<SessionWithMetadata | null>(null);
  const [trackingData, setTrackingData] = useState<FrameDetections[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"recorded" | "realtime">("recorded");
  const [isGeneratingDetections, setIsGeneratingDetections] = useState(false);
  const [generationMessage, setGenerationMessage] = useState<string | null>(
    null
  );
  const [generationError, setGenerationError] = useState<string | null>(null);

  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    try {
      const response = await fetch("/api/sessions");
      if (!response.ok) throw new Error("Failed to fetch sessions");
      const data = await response.json();
      setSessions(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch sessions");
    }
  };

  const loadSessionData = async (session: Session) => {
    setLoading(true);
    setError(null);
    setGenerationMessage(null);
    setGenerationError(null);

    try {
      // Load both detections and metadata in parallel
      const [detectionsResponse, metadataResponse] = await Promise.all([
        fetch(`/api/sessions/${session.session_id}/detections`),
        fetch(`/api/sessions/${session.session_id}/metadata`),
      ]);

      if (!detectionsResponse.ok)
        throw new Error("Failed to fetch tracking data");

      const detections: DetectionData[] = await detectionsResponse.json();
      const frames = convertDetectionsToFrames(detections, session.fps);
      setTrackingData(frames);

      // Load enhanced telemetry data if available
      let enhancedTelemetry = null;
      try {
        const telemetryResponse = await fetch(
          `/api/sessions/${session.session_id}/telemetry`
        );
        if (telemetryResponse.ok) {
          enhancedTelemetry = await telemetryResponse.json();
          console.log(
            `Loaded enhanced telemetry with ${
              enhancedTelemetry.telemetry?.length || 0
            } points`
          );
        }
      } catch (err) {
        console.warn("Enhanced telemetry not available:", err);
      }

      // Load GPS metadata if available (legacy fallback)
      let gpsMetadata = [];
      if (metadataResponse.ok) {
        gpsMetadata = await metadataResponse.json();
        console.log(`Loaded ${gpsMetadata.length} GPS metadata entries`);
      } else {
        console.warn("No GPS metadata available for this session");
      }

      // Convert session to SessionWithMetadata
      const sessionWithMetadata: SessionWithMetadata = {
        ...session,
        metadata: gpsMetadata,
        enhanced_telemetry: enhancedTelemetry,
      };

      setSelectedSession(sessionWithMetadata);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load session data"
      );
    } finally {
      setLoading(false);
    }
  };

  const regenerateDetections = async () => {
    if (!selectedSession) return;

    setIsGeneratingDetections(true);
    setGenerationMessage(null);
    setGenerationError(null);

    try {
      const response = await fetch(
        `/api/sessions/${selectedSession.session_id}/generate-detections`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            videoPath: selectedSession.video_path,
          }),
        }
      );

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        const message =
          (errorBody && errorBody.error) ||
          `Failed to regenerate detections (status ${response.status})`;
        throw new Error(message);
      }

      const data = await response.json();

      // Refresh session data to reflect the new detections
      await loadSessionData(selectedSession);

      if (data?.message) {
        const detectionInfo =
          typeof data.detections === "number"
            ? `${data.message} (${data.detections} detections)`
            : data.message;
        setGenerationMessage(detectionInfo);
      } else {
        setGenerationMessage("Detections regenerated successfully.");
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to regenerate detections";
      setGenerationError(message);
      console.error(err);
    } finally {
      setIsGeneratingDetections(false);
    }
  };

  const getVideoUrl = (session: Session | SessionWithMetadata): string => {
    const url = `/api/video/${encodeURIComponent(session.video_path)}`;
    console.log("Generated video URL:", url, "for path:", session.video_path);
    return url;
  };

  const totalDetections = trackingData.reduce(
    (acc, frame) => acc + frame.objects.length,
    0
  );
  const framesWithDetections = trackingData.filter(
    (frame) => frame.objects.length > 0
  ).length;

  return (
    <div className="min-h-screen bg-tactical-bg">
      {/* Header Navigation */}
      <header className="tactical-panel border-b border-tactical-border shadow-tactical">
        <div className="w-full px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-gradient-to-br from-primary-600 via-primary-500 to-tactical-glow rounded-xl flex items-center justify-center shadow-glow">
                  <HiLocationMarker className="text-white text-2xl" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-tactical-text">
                    EDTH Object Tracker
                  </h1>
                  <p className="text-sm text-tactical-muted">
                    Advanced object detection and tracking
                  </p>
                </div>
              </div>
            </div>

            {/* Mode Selector */}
            <div className="flex items-center gap-6">
              <div className="bg-tactical-surface/50 rounded-xl p-1 border border-tactical-border shadow-inner-glow">
                <button
                  onClick={() => setViewMode("recorded")}
                  className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all flex items-center gap-2 ${
                    viewMode === "recorded"
                      ? "bg-primary-600 text-white shadow-glow border border-primary-500/50"
                      : "text-tactical-muted hover:bg-tactical-surface hover:text-tactical-text"
                  }`}
                >
                  <HiFolder className="w-4 h-4" />
                  <span className="hidden sm:inline">Recorded</span>
                </button>
                <button
                  onClick={() => setViewMode("realtime")}
                  className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all flex items-center gap-2 ${
                    viewMode === "realtime"
                      ? "bg-accent-600 text-white shadow-glow border border-accent-500/50"
                      : "text-tactical-muted hover:bg-tactical-surface hover:text-tactical-text"
                  }`}
                >
                  <RiLiveLine className="w-4 h-4" />
                  <span className="hidden sm:inline">Live</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="w-full min-h-0 flex-1">
        {viewMode === "realtime" ? (
          /* Real-time Detection Mode */
          <div className="h-[calc(100vh-120px)] p-4 animate-fade-in">
            {/* Realtime Header */}
            <div className="card">
              <div className="card-header">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 bg-gradient-to-br from-accent-600 to-accent-700 rounded-lg flex items-center justify-center shadow-glow">
                      <RiLiveLine className="text-white text-xl" />
                    </div>
                    <div>
                      <h2 className="text-xl font-bold text-tactical-text">
                        Live Detection & Tracking
                      </h2>
                      <p className="text-sm text-tactical-muted">
                        Real-time object detection with AI-powered tracking
                      </p>
                    </div>
                  </div>
                  <div className="status-connected">
                    <div className="w-2 h-2 bg-success-400 rounded-full animate-pulse"></div>
                    OPERATIONAL
                  </div>
                </div>
              </div>
              <div className="card-body">
                <RealtimeVideoCanvas
                  onDetectionData={(data) => {
                    console.log("Real-time detection data:", data);
                  }}
                />
              </div>
            </div>
          </div>
        ) : (
          /* Recorded Sessions Mode */
          <div className="flex h-[calc(100vh-120px)] gap-4 p-4 animate-fade-in">
            {/* Session Selection Sidebar */}
            <div className="w-80 flex-shrink-0">
              <div className="card sticky top-6">
                <div className="card-header">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-gradient-to-br from-primary-600 to-primary-700 rounded-lg flex items-center justify-center shadow-glow">
                      <RiFileListLine className="text-white text-lg" />
                    </div>
                    <div>
                      <h2 className="text-lg font-bold text-tactical-text">
                        Recorded Sessions
                      </h2>
                      <p className="text-xs text-tactical-muted">
                        {sessions.length} sessions available
                      </p>
                    </div>
                  </div>
                </div>

                <div className="p-4">
                  {error && (
                    <div className="mb-4 p-3 bg-accent-900/50 border border-accent-600/50 text-accent-200 rounded-lg text-sm shadow-inner-glow">
                      <div className="font-semibold">System Error</div>
                      <div className="text-xs mt-1 font-mono">{error}</div>
                    </div>
                  )}

                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {sessions.map((session) => (
                      <div
                        key={session.session_id}
                        className={`p-3 rounded-xl border cursor-pointer transition-all duration-300 ${
                          selectedSession?.session_id === session.session_id
                            ? "border-primary-500/50 bg-primary-900/30 shadow-glow"
                            : "border-tactical-border hover:border-primary-600/30 hover:bg-tactical-surface/50 hover:shadow-inner-glow"
                        }`}
                        onClick={() => loadSessionData(session)}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="font-semibold text-sm text-tactical-text truncate">
                              Session #{session.session_id}
                            </div>
                            <div className="text-xs text-tactical-muted mt-1 truncate font-mono">
                              {session.video_path.split("/").pop()}
                            </div>
                          </div>
                          {selectedSession?.session_id ===
                            session.session_id && (
                            <div className="w-2 h-2 bg-primary-400 rounded-full flex-shrink-0 shadow-glow"></div>
                          )}
                        </div>

                        <div className="flex items-center gap-2 mt-2">
                          <span className="px-2 py-1 bg-tactical-surface/50 text-tactical-text rounded text-xs font-mono border border-tactical-border">
                            {session.total_frames
                              ? `${session.total_frames} FRAMES`
                              : "PROCESSING"}
                          </span>
                          {session.fps && (
                            <span className="px-2 py-1 bg-primary-900/50 text-primary-300 rounded text-xs font-mono border border-primary-600/50">
                              {session.fps} FPS
                            </span>
                          )}
                        </div>

                        <div className="text-xs text-tactical-muted mt-2 font-mono">
                          {new Date(session.start_time).toLocaleDateString()}{" "}
                          {new Date(session.start_time).toLocaleTimeString()}
                        </div>
                      </div>
                    ))}

                    {sessions.length === 0 && !error && (
                      <div className="text-center py-8">
                        <HiFolder className="text-neutral-400 text-2xl mb-2 mx-auto" />
                        <div className="text-neutral-500 text-sm">
                          No sessions found
                        </div>
                        <div className="text-neutral-400 text-xs mt-1">
                          Start recording to see sessions here
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="mt-4 pt-4 border-t border-neutral-200">
                    <button
                      onClick={fetchSessions}
                      className="btn-secondary w-full text-sm"
                    >
                      <HiRefresh className="w-4 h-4" />
                      Refresh
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Main Content Area - Map Hero */}
            <div className="flex-1 flex flex-col min-h-0">
              {loading && (
                <div className="card">
                  <div className="card-body">
                    <div className="flex items-center justify-center py-12">
                      <div className="flex flex-col items-center gap-4">
                        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary-600"></div>
                        <div className="text-center">
                          <div className="font-medium text-tactical-text">
                            Loading Session Data
                          </div>
                          <div className="text-sm text-tactical-muted mt-1">
                            Processing tracking data and metadata...
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {selectedSession && !loading && (
                <div className="space-y-6">
                  {/* Session Summary Dashboard */}
                  <div className="card">
                    <div className="card-header">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                            <RiDashboardFill className="text-white text-lg" />
                          </div>
                          <div>
                            <h3 className="text-lg font-semibold text-tactical-text">
                              Session #{selectedSession.session_id}
                            </h3>
                            <p className="text-sm text-tactical-muted">
                              {selectedSession.video_path.split("/").pop()}
                            </p>
                          </div>
                        </div>
                        <button
                          onClick={regenerateDetections}
                          disabled={isGeneratingDetections}
                          className={`btn ${
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
                              Regenerate
                            </>
                          )}
                        </button>
                      </div>
                    </div>

                    <div className="card-body">
                      {/* Status Messages */}
                      {generationMessage && (
                        <div className="mb-4 p-3 bg-success-50 border border-success-200 text-success-800 rounded-lg text-sm">
                          <div className="font-medium">Success</div>
                          <div className="text-xs mt-1">
                            {generationMessage}
                          </div>
                        </div>
                      )}
                      {generationError && (
                        <div className="mb-4 p-3 bg-accent-50 border border-accent-200 text-accent-800 rounded-lg text-sm">
                          <div className="font-medium">Error</div>
                          <div className="text-xs mt-1">{generationError}</div>
                        </div>
                      )}

                      {/* Metrics Grid */}
                      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                        <div className="metric-display">
                          <div className="text-2xl font-bold text-primary-600">
                            {trackingData.length}
                          </div>
                          <div className="text-sm text-neutral-600">
                            Total Frames
                          </div>
                        </div>
                        <div className="metric-display">
                          <div className="text-2xl font-bold text-success-600">
                            {framesWithDetections}
                          </div>
                          <div className="text-sm text-neutral-600">
                            With Detections
                          </div>
                        </div>
                        <div className="metric-display">
                          <div className="text-2xl font-bold text-secondary-600">
                            {totalDetections}
                          </div>
                          <div className="text-sm text-neutral-600">
                            Objects Tracked
                          </div>
                        </div>
                        <div className="metric-display">
                          <div className="text-2xl font-bold text-warning-600">
                            {selectedSession.fps || "N/A"}
                          </div>
                          <div className="text-sm text-neutral-600">FPS</div>
                        </div>
                      </div>

                      {trackingData.length === 0 && (
                        <div className="text-center py-8 text-neutral-500">
                          <HiChartBar className="text-neutral-400 text-4xl mb-2 mx-auto" />
                          <div className="font-medium">
                            No detection data available
                          </div>
                          <div className="text-sm mt-1">
                            Click "Regenerate" to process this session
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Video and Map Viewer */}
                  <VideoMapViewer
                    session={selectedSession}
                    trackingData={trackingData}
                    videoSrc={getVideoUrl(selectedSession)}
                  />
                </div>
              )}

              {!selectedSession && !loading && (
                <div className="card">
                  <div className="card-body">
                    <div className="text-center py-16">
                      <div className="w-16 h-16 bg-gradient-to-br from-primary-100 to-secondary-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
                        <HiLocationMarker className="text-primary-600 text-2xl" />
                      </div>
                      <h3 className="text-xl font-semibold text-tactical-text mb-2">
                        Select a Session
                      </h3>
                      <p className="text-tactical-muted max-w-md mx-auto">
                        Choose a tracking session from the sidebar to view the
                        video with object detections and synchronized drone
                        telemetry
                      </p>
                      <div className="mt-6 text-sm text-tactical-muted flex items-center justify-center gap-2">
                        <HiServer className="w-4 h-4" />
                        Sessions with GPS data will show flight paths and camera
                        footprints
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
