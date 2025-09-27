import { useState, useEffect } from "react";
import {
  HiFolder,
  HiRefresh,
  HiChartBar,
  HiLocationMarker,
  HiPlay,
  HiBookOpen,
  HiLink,
  HiMap,
  HiAdjustments,
} from "react-icons/hi";
import { MdGpsFixed, MdAnalytics } from "react-icons/md";
import { RiLiveLine, RiFileListLine } from "react-icons/ri";
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
    frames.forEach((frame) => {
      frame.timestamp = Math.max(0, frame.timestamp - minTimestamp);
    });
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
      console.log("🔄 Loading session data for session:", session.session_id);

      // Load both detections and metadata in parallel
      const [detectionsResponse, metadataResponse] = await Promise.all([
        fetch(`/api/sessions/${session.session_id}/detections`),
        fetch(`/api/sessions/${session.session_id}/metadata`),
      ]);

      if (!detectionsResponse.ok)
        throw new Error("Failed to fetch tracking data");

      const detections: DetectionData[] = await detectionsResponse.json();
      console.log("📊 Loaded detections:", detections.length, "raw detections");

      const frames = convertDetectionsToFrames(detections, session.fps);
      console.log(
        "🎬 Converted to frames:",
        frames.length,
        "frames with",
        frames.reduce((acc, f) => acc + f.objects.length, 0),
        "total objects"
      );

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
            } GPS points`
          );
        }
      } catch (telemetryErr) {
        console.log("No enhanced telemetry available");
      }

      // Load basic GPS metadata if available
      let basicMetadata = null;
      if (metadataResponse.ok) {
        try {
          basicMetadata = await metadataResponse.json();
          console.log(
            `Loaded basic GPS metadata with ${
              basicMetadata?.length || 0
            } GPS points`
          );
        } catch (metadataErr) {
          console.log("No basic GPS metadata available");
        }
      }

      const enrichedSession: SessionWithMetadata = {
        ...session,
        enhanced_telemetry: enhancedTelemetry,
        metadata: basicMetadata,
      };

      setSelectedSession(enrichedSession);
      console.log("✅ Session data loaded successfully");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to load session data";
      setError(message);
      console.error("❌ Failed to load session data:", err);
    } finally {
      setLoading(false);
    }
  };

  const regenerateDetections = async () => {
    if (!selectedSession) return;

    console.log(
      "🚀 Starting regenerate detections for session:",
      selectedSession.session_id
    );

    setIsGeneratingDetections(true);
    setGenerationMessage(null);
    setGenerationError(null);

    try {
      console.log("📡 Calling regenerate API...");
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
      console.log("✅ Regenerate API response:", data);

      // Store old detection count for comparison
      const oldDetectionCount = trackingData.reduce(
        (acc, f) => acc + f.objects.length,
        0
      );
      console.log("📊 Old detection count:", oldDetectionCount);

      // Refresh session data to reflect the new detections
      console.log("🔄 Refreshing session data...");
      await loadSessionData(selectedSession);

      // Check new detection count
      const newDetectionCount = trackingData.reduce(
        (acc, f) => acc + f.objects.length,
        0
      );
      console.log("📊 New detection count:", newDetectionCount);

      if (data?.message) {
        const detectionInfo =
          typeof data.detections === "number"
            ? `${data.message} (${data.detections} detections) - Old: ${oldDetectionCount}, New: ${newDetectionCount}`
            : data.message;
        setGenerationMessage(detectionInfo);
      } else {
        setGenerationMessage(
          `Detections regenerated successfully. Old: ${oldDetectionCount}, New: ${newDetectionCount}`
        );
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to regenerate detections";
      setGenerationError(message);
      console.error("❌ Regenerate failed:", err);
    } finally {
      setIsGeneratingDetections(false);
    }
  };

  if (error) {
    return (
      <div className="app-scaled">
        <div className="min-h-screen bg-cyber-black text-cyber-text flex items-center justify-center p-6">
          <div className="text-center">
            <div className="text-6xl mb-4">⚠️</div>
            <h1 className="text-2xl font-bold mb-2 text-red-400">Error</h1>
            <p className="text-cyber-muted">{error}</p>
            <button
              onClick={() => window.location.reload()}
              className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
            >
              Reload Page
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-scaled">
      <div className="min-h-screen bg-cyber-black text-cyber-text">
        {/* Header */}
        <header className="cyber-panel border-b border-cyber-border">
          <div className="w-full px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-cyber-surface border-2 border-neon-cyan flex items-center justify-center shadow-cyber relative">
                    <HiLocationMarker className="text-neon-cyan text-2xl text-glow" />
                    <div className="absolute inset-0 border border-neon-cyan animate-pulse-neon opacity-50"></div>
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold text-neon-cyan text-glow">
                      &gt; EDTH_TRACKER.EXE
                    </h1>
                    <p className="text-sm text-cyber-muted font-mono">
                      [NEURAL_DETECTION_SYSTEM_v2.1_ONLINE]
                    </p>
                  </div>
                </div>
              </div>

              {/* Mode Selector */}
              <div className="flex items-center gap-6">
                <div className="bg-cyber-surface border border-cyber-border p-1 flex">
                  <button
                    onClick={() => setViewMode("recorded")}
                    className={`px-6 py-3 text-sm font-bold transition-all flex items-center gap-3 font-mono uppercase tracking-wider ${
                      viewMode === "recorded"
                        ? "bg-neon-cyan text-cyber-black shadow-cyber border-glow"
                        : "text-neon-cyan hover:bg-cyber-border hover:text-neon-cyan border border-transparent hover:border-neon-cyan"
                    }`}
                  >
                    <HiFolder className="w-4 h-4" />
                    <span className="hidden sm:inline">ARCHIVE</span>
                  </button>
                  <button
                    onClick={() => setViewMode("realtime")}
                    className={`px-6 py-3 text-sm font-bold transition-all flex items-center gap-3 font-mono uppercase tracking-wider ${
                      viewMode === "realtime"
                        ? "bg-neon-pink text-cyber-black shadow-cyber-pink border-glow"
                        : "text-neon-pink hover:bg-cyber-border hover:text-neon-pink border border-transparent hover:border-neon-pink"
                    }`}
                  >
                    <RiLiveLine className="w-4 h-4" />
                    <span className="hidden sm:inline">REALTIME</span>
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
            <div className="min-h-[calc(100vh-120px)] p-6 cyber-scan overflow-y-auto">
              {/* Realtime Header */}
              <div className="cyber-card">
                <div className="cyber-card-header">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-6">
                      <div className="w-12 h-12 bg-cyber-surface border-2 border-neon-pink flex items-center justify-center shadow-cyber-pink relative">
                        <RiLiveLine className="text-neon-pink text-xl text-glow" />
                        <div className="absolute inset-0 border border-neon-pink animate-flicker opacity-30"></div>
                      </div>
                      <div>
                        <h2 className="text-xl font-bold text-neon-pink text-glow">
                          &gt;&gt; NEURAL_STREAM_ACTIVE
                        </h2>
                        <p className="text-sm text-cyber-muted font-mono">
                          [AI_DETECTION_ENGINE_PROCESSING...]
                        </p>
                      </div>
                    </div>
                    <div className="cyber-status-online">
                      <div className="w-3 h-3 bg-neon-green animate-pulse"></div>
                      SYSTEM_ONLINE
                    </div>
                  </div>
                </div>
                <div className="p-6">
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
            <div
              className={`flex w-full min-h-0 ${
                selectedSession ? "gap-6 p-6" : ""
              }`}
            >
              {/* Sessions Sidebar */}
              <div
                className={`${
                  selectedSession
                    ? "w-80 flex-shrink-0"
                    : "flex-1 flex flex-col min-h-0"
                } cyber-panel`}
              >
                <div className="cyber-card-header">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-cyber-surface border-2 border-neon-cyan flex items-center justify-center shadow-cyber relative">
                        <RiFileListLine className="text-neon-cyan text-sm text-glow" />
                        <div className="absolute inset-0 border border-neon-cyan animate-pulse-neon opacity-30"></div>
                      </div>
                      <div>
                        <h3 className="font-semibold text-neon-cyan text-glow">
                          &gt;&gt; ARCHIVE
                        </h3>
                        <p className="text-xs text-cyber-muted font-mono">
                          [SESSION_DATA_v1.0]
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={fetchSessions}
                      className="p-2 bg-cyber-surface border border-neon-cyan text-neon-cyan hover:bg-cyber-border transition-colors shadow-cyber"
                    >
                      <HiRefresh className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {!selectedSession && (
                  /* Quick Start Guide */
                  <div className="cyber-card mt-6">
                    <div className="cyber-card-header">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 bg-cyber-surface border-2 border-neon-green flex items-center justify-center shadow-cyber-green">
                          <HiBookOpen className="text-neon-green text-sm text-glow" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-neon-green text-glow">
                            &gt;&gt; QUICK START
                          </h3>
                          <p className="text-xs text-cyber-muted font-mono">
                            [INTERFACE_GUIDE_v1.0]
                          </p>
                        </div>
                      </div>
                    </div>
                    <div className="p-4">
                      <div className="space-y-4">
                        <div className="space-y-2">
                          <h4 className="font-medium text-neon-cyan flex items-center gap-2 font-mono text-sm">
                            <span className="w-5 h-5 bg-cyber-surface border border-neon-cyan flex items-center justify-center text-xs font-bold">
                              1
                            </span>
                            VIDEO_CTRL
                          </h4>
                          <ul className="space-y-1 text-xs text-cyber-muted font-mono">
                            <li className="flex items-start gap-2">
                              <HiPlay className="text-neon-cyan w-3 h-3 mt-0.5 flex-shrink-0" />
                              <span>PLAY/PAUSE_TOGGLE</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <HiLocationMarker className="text-neon-cyan w-3 h-3 mt-0.5 flex-shrink-0" />
                              <span>TIMELINE_SEEK</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <MdGpsFixed className="text-neon-cyan w-3 h-3 mt-0.5 flex-shrink-0" />
                              <span>HOVER_DETECT_INFO</span>
                            </li>
                          </ul>
                        </div>
                        <div className="space-y-2">
                          <h4 className="font-medium text-neon-pink flex items-center gap-2 font-mono text-sm">
                            <span className="w-5 h-5 bg-cyber-surface border border-neon-pink flex items-center justify-center text-xs font-bold">
                              2
                            </span>
                            MAP_CTRL
                          </h4>
                          <ul className="space-y-1 text-xs text-cyber-muted font-mono">
                            <li className="flex items-start gap-2">
                              <HiLink className="text-neon-pink w-3 h-3 mt-0.5 flex-shrink-0" />
                              <span>SYNC_TOGGLE</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <HiMap className="text-neon-pink w-3 h-3 mt-0.5 flex-shrink-0" />
                              <span>ZOOM_PAN</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <HiAdjustments className="text-neon-pink w-3 h-3 mt-0.5 flex-shrink-0" />
                              <span>FOOTPRINT_VIEW</span>
                            </li>
                          </ul>
                        </div>
                        <div className="space-y-2">
                          <h4 className="font-medium text-neon-yellow flex items-center gap-2 font-mono text-sm">
                            <span className="w-5 h-5 bg-cyber-surface border border-neon-yellow flex items-center justify-center text-xs font-bold">
                              3
                            </span>
                            ANALYSIS_MOD
                          </h4>
                          <ul className="space-y-1 text-xs text-cyber-muted font-mono">
                            <li className="flex items-start gap-2">
                              <HiChartBar className="text-neon-yellow w-3 h-3 mt-0.5 flex-shrink-0" />
                              <span>TELEMETRY_LIVE</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <HiAdjustments className="text-neon-yellow w-3 h-3 mt-0.5 flex-shrink-0" />
                              <span>FRAME_MANUAL</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <MdAnalytics className="text-neon-yellow w-3 h-3 mt-0.5 flex-shrink-0" />
                              <span>ANALYTICS_VIEW</span>
                            </li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Main Content Area - Map Hero */}
                <div className="flex-1 flex flex-col min-h-0">
                  <div className="p-4 flex-1 overflow-y-auto">
                    {sessions.length === 0 ? (
                      <div className="text-center py-8 text-cyber-muted">
                        <HiFolder className="text-6xl mb-4 mx-auto opacity-50" />
                        <div className="font-medium">No sessions found</div>
                        <div className="text-sm mt-2">
                          Process some videos to see tracking sessions here
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        {sessions.map((session) => (
                          <button
                            key={session.session_id}
                            onClick={() => loadSessionData(session)}
                            className={`w-full text-left p-3 border transition-all ${
                              selectedSession?.session_id === session.session_id
                                ? "bg-cyber-border border-neon-cyan shadow-cyber text-neon-cyan"
                                : "bg-cyber-surface border-cyber-border text-cyber-text hover:border-neon-cyan hover:bg-cyber-border"
                            }`}
                          >
                            <div className="font-mono font-medium">
                              SESSION_LOG #{session.session_id}
                            </div>
                            <div className="text-xs text-cyber-muted mt-1 truncate">
                              {session.video_path.split("/").pop()}
                            </div>
                            {session.session_id.toString() && (
                              <div className="text-xs text-cyber-muted">
                                {new Date(
                                  session.session_id.toString()
                                ).toLocaleDateString()}
                              </div>
                            )}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Main Video Viewer */}
              {selectedSession && (
                <div className="flex-1 min-h-0">
                  {loading ? (
                    <div className="cyber-card h-full flex items-center justify-center">
                      <div className="text-center text-cyber-muted">
                        <div className="animate-spin w-8 h-8 border-2 border-neon-cyan border-t-transparent mx-auto mb-4"></div>
                        <div className="font-medium font-mono">
                          [LOADING_SESSION_DATA...]
                        </div>
                        <div className="text-sm mt-1">
                          Processing tracking information
                        </div>
                      </div>
                    </div>
                  ) : (
                    <VideoMapViewer
                      session={selectedSession}
                      trackingData={trackingData}
                      videoSrc={`/api/sessions/${selectedSession.session_id}/video`}
                      onRegenerateDetections={regenerateDetections}
                      isGeneratingDetections={isGeneratingDetections}
                      generationMessage={generationMessage}
                      generationError={generationError}
                    />
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
