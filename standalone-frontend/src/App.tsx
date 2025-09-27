import { useState, useEffect } from "react";
import VideoMapViewer from "./VideoMapViewer";
import RealtimeVideoCanvas from "./RealtimeVideoCanvas";
import type { Session, FrameDetections, DetectionData, SessionWithMetadata } from "./types";

function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedSession, setSelectedSession] = useState<SessionWithMetadata | null>(null);
  const [trackingData, setTrackingData] = useState<FrameDetections[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"recorded" | "realtime">("recorded");

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

    try {
      // Load both detections and metadata in parallel
      const [detectionsResponse, metadataResponse] = await Promise.all([
        fetch(`/api/sessions/${session.session_id}/detections`),
        fetch(`/api/sessions/${session.session_id}/metadata`)
      ]);

      if (!detectionsResponse.ok) throw new Error("Failed to fetch tracking data");

      const detections: DetectionData[] = await detectionsResponse.json();

      // Convert detection data to FrameDetections format
      const frameMap = new Map<number, FrameDetections>();

      detections.forEach((detection) => {
        if (!frameMap.has(detection.frame_number)) {
          frameMap.set(detection.frame_number, {
            frame_number: detection.frame_number,
            timestamp: detection.timestamp,
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
      });

      const frames = Array.from(frameMap.values()).sort(
        (a, b) => a.frame_number - b.frame_number
      );
      setTrackingData(frames);

      // Load GPS metadata if available
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
        metadata: gpsMetadata
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

  const getVideoUrl = (session: Session | SessionWithMetadata): string => {
    const url = `/api/video/${encodeURIComponent(session.video_path)}`;
    console.log("Generated video URL:", url, "for path:", session.video_path);
    return url;
  };

  return (
    <div className="bg-gradient-to-br from-primary-500 to-secondary-500 min-h-screen">
      <div className="max-w-7xl mx-auto p-6">
        <header className="text-center mb-8 text-white">
          <h1 className="text-4xl font-bold mb-3 drop-shadow-lg">
            EDTH Object Tracker
          </h1>
          <p className="text-lg opacity-90">
            Real-time object tracking with video visualization
          </p>

          {/* Mode Selector */}
          <div className="mt-6 flex justify-center">
            <div className="bg-white/20 rounded-lg p-1 backdrop-blur-sm">
              <button
                onClick={() => setViewMode("recorded")}
                className={`px-6 py-2 rounded-md transition-all ${
                  viewMode === "recorded"
                    ? "bg-white text-primary-600 shadow-md"
                    : "text-white hover:bg-white/10"
                }`}
              >
                📁 Recorded Sessions
              </button>
              <button
                onClick={() => setViewMode("realtime")}
                className={`px-6 py-2 rounded-md transition-all ${
                  viewMode === "realtime"
                    ? "bg-white text-primary-600 shadow-md"
                    : "text-white hover:bg-white/10"
                }`}
              >
                🔴 Live Detection
              </button>
            </div>
          </div>
        </header>

        {viewMode === "realtime" ? (
          /* Real-time Detection Mode */
          <div className="bg-white rounded-xl p-6 shadow-xl">
            <h2 className="text-xl font-semibold text-gray-700 mb-6">
              🔴 Live Object Detection & Tracking
            </h2>
            <RealtimeVideoCanvas
              onDetectionData={(data) => {
                console.log("Real-time detection data:", data);
              }}
            />
          </div>
        ) : (
          /* Recorded Sessions Mode */
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Session Selection */}
            <div className="lg:col-span-1">
              <div className="bg-white rounded-xl p-6 shadow-xl">
                <h2 className="text-xl font-semibold text-gray-700 mb-4">
                  📁 Tracking Sessions
                </h2>

                {error && (
                  <div className="mb-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
                    {error}
                  </div>
                )}

                <div className="space-y-3">
                  {sessions.map((session) => (
                    <div
                      key={session.session_id}
                      className={`p-3 rounded-lg border cursor-pointer transition-all ${
                        selectedSession?.session_id === session.session_id
                          ? "border-primary-500 bg-primary-50"
                          : "border-gray-200 hover:border-primary-300 hover:bg-gray-50"
                      }`}
                      onClick={() => loadSessionData(session)}
                    >
                      <div className="font-medium text-sm text-gray-900">
                        Session #{session.session_id}
                      </div>
                      <div className="text-xs text-gray-600 mt-1">
                        {session.video_path.split("/").pop()}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {session.total_frames
                          ? `${session.total_frames} frames`
                          : "In progress"}
                        {session.fps && ` • ${session.fps} FPS`}
                      </div>
                      <div className="text-xs text-gray-500">
                        {new Date(session.start_time).toLocaleString()}
                      </div>
                    </div>
                  ))}

                  {sessions.length === 0 && !error && (
                    <div className="text-gray-500 text-center py-4">
                      No tracking sessions found
                    </div>
                  )}
                </div>

                <button
                  onClick={fetchSessions}
                  className="mt-4 w-full bg-primary-500 text-white px-4 py-2 rounded-lg hover:bg-primary-600 transition-colors"
                >
                  Refresh Sessions
                </button>
              </div>
            </div>

            {/* Video and Map Viewer */}
            <div className="lg:col-span-2">
              {loading && (
                <div className="bg-white rounded-xl p-6 shadow-xl">
                  <div className="flex items-center justify-center py-12">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
                    <span className="ml-3 text-gray-600">
                      Loading tracking data...
                    </span>
                  </div>
                </div>
              )}

              {selectedSession && !loading && (
                <VideoMapViewer
                  session={selectedSession}
                  trackingData={trackingData}
                  videoSrc={getVideoUrl(selectedSession)}
                />
              )}

              {!selectedSession && !loading && (
                <div className="bg-white rounded-xl p-6 shadow-xl">
                  <h2 className="text-xl font-semibold text-gray-700 mb-4">
                    Video Player with Detections & Map
                  </h2>
                  <div className="text-center py-12 text-gray-500">
                    <div className="text-4xl mb-4">🗺️📹</div>
                    <p>
                      Select a tracking session to view the video with
                      detections and synchronized map
                    </p>
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
