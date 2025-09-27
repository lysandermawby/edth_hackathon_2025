import React, { useState, useEffect } from "react";
import { HiUpload, HiVideoCamera, HiRefresh, HiX, HiPlay, HiFolder } from "react-icons/hi";

interface Video {
  filename: string;
  path: string;
  size: number;
  modified: string;
  sizeFormatted: string;
}

interface VideoImportProps {
  isOpen: boolean;
  onClose: () => void;
  onImportSuccess: () => void;
}

const VideoImport: React.FC<VideoImportProps> = ({ isOpen, onClose, onImportSuccess }) => {
  const [availableVideos, setAvailableVideos] = useState<Video[]>([]);
  const [loadingVideos, setLoadingVideos] = useState(false);
  const [importingVideo, setImportingVideo] = useState<string | null>(null);
  const [importMessage, setImportMessage] = useState<string | null>(null);
  const [importError, setImportError] = useState<string | null>(null);

  const fetchAvailableVideos = async () => {
    setLoadingVideos(true);
    setImportError(null);
    try {
      const response = await fetch("/api/videos/available");
      if (!response.ok) throw new Error("Failed to fetch available videos");
      const data = await response.json();
      setAvailableVideos(data);
    } catch (err) {
      setImportError(err instanceof Error ? err.message : "Failed to fetch videos");
    } finally {
      setLoadingVideos(false);
    }
  };

  const importVideo = async (videoPath: string) => {
    setImportingVideo(videoPath);
    setImportMessage(null);
    setImportError(null);
    
    try {
      console.log("🎬 Importing video:", videoPath);
      const response = await fetch("/api/sessions/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          videoPath: videoPath,
          autoProcess: true
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || "Failed to import video");
      }

      const data = await response.json();
      console.log("✅ Import response:", data);
      
      setImportMessage(`${data.message} (Session #${data.session_id}, ${data.detections} detections)`);
      
      // Notify parent component
      onImportSuccess();
      
      // Close modal after successful import
      setTimeout(() => {
        onClose();
        setImportMessage(null);
      }, 3000);
      
    } catch (err) {
      console.error("❌ Import failed:", err);
      setImportError(err instanceof Error ? err.message : "Failed to import video");
    } finally {
      setImportingVideo(null);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchAvailableVideos();
      setImportMessage(null);
      setImportError(null);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-cyber-black/80 flex items-center justify-center z-50">
      <div className="cyber-card w-full max-w-4xl mx-4 max-h-[80vh] overflow-hidden flex flex-col">
        <div className="cyber-card-header">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-cyber-surface border-2 border-neon-cyan flex items-center justify-center shadow-cyber">
                <HiUpload className="text-neon-cyan text-sm text-glow" />
              </div>
              <div>
                <h3 className="font-semibold text-neon-cyan text-glow">
                  &gt;&gt; IMPORT VIDEO
                </h3>
                <p className="text-xs text-cyber-muted font-mono">
                  [VIDEO_PROCESSING_PIPELINE_v1.0]
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 bg-cyber-surface border border-cyber-border text-cyber-text hover:border-neon-cyan hover:text-neon-cyan transition-colors"
            >
              <HiX className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="p-6 flex-1 overflow-hidden flex flex-col">
          {/* Status Messages */}
          {importMessage && (
            <div className="mb-4 p-3 bg-neon-green/10 border border-neon-green text-neon-green text-sm">
              <div className="font-medium">SUCCESS</div>
              <div className="text-xs mt-1 font-mono">{importMessage}</div>
            </div>
          )}
          
          {importError && (
            <div className="mb-4 p-3 bg-red-500/10 border border-red-500 text-red-400 text-sm">
              <div className="font-medium">ERROR</div>
              <div className="text-xs mt-1 font-mono">{importError}</div>
            </div>
          )}

          {/* Controls */}
          <div className="flex items-center justify-between mb-6">
            <div className="text-sm text-cyber-muted font-mono">
              Found {availableVideos.length} video{availableVideos.length !== 1 ? 's' : ''} in data directory
            </div>
            <button
              onClick={fetchAvailableVideos}
              disabled={loadingVideos}
              className="px-3 py-2 bg-cyber-surface border border-neon-cyan text-neon-cyan hover:bg-cyber-border transition-colors text-sm font-mono disabled:opacity-50"
            >
              <HiRefresh className="w-4 h-4 mr-2 inline" />
              {loadingVideos ? 'SCANNING...' : 'REFRESH'}
            </button>
          </div>

          {/* Video List */}
          <div className="flex-1 overflow-y-auto">
            {loadingVideos ? (
              <div className="text-center py-8 text-cyber-muted">
                <div className="animate-spin w-8 h-8 border-2 border-neon-cyan border-t-transparent mx-auto mb-4"></div>
                <div className="font-medium font-mono">[SCANNING_DATA_DIRECTORY...]</div>
              </div>
            ) : availableVideos.length === 0 ? (
              <div className="text-center py-8 text-cyber-muted">
                <HiFolder className="text-6xl mb-4 mx-auto opacity-50" />
                <div className="font-medium">No videos found</div>
                <div className="text-sm mt-2 font-mono">
                  Add .mp4, .avi, .mov, or other video files to the data/ directory
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                {availableVideos.map((video) => (
                  <div
                    key={video.path}
                    className="cyber-card border border-cyber-border hover:border-neon-cyan transition-colors"
                  >
                    <div className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4 flex-1 min-w-0">
                          <div className="w-10 h-10 bg-cyber-surface border border-neon-cyan flex items-center justify-center">
                            <HiVideoCamera className="text-neon-cyan text-lg" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="font-medium text-cyber-text truncate font-mono">
                              {video.filename}
                            </div>
                            <div className="text-xs text-cyber-muted font-mono">
                              {video.path} • {video.sizeFormatted}
                            </div>
                            <div className="text-xs text-cyber-muted font-mono">
                              Modified: {new Date(video.modified).toLocaleDateString()}
                            </div>
                          </div>
                        </div>
                        <button
                          onClick={() => importVideo(video.path)}
                          disabled={importingVideo === video.path}
                          className={`px-4 py-2 text-sm font-bold font-mono uppercase tracking-wider transition-all ${
                            importingVideo === video.path
                              ? "bg-cyber-surface text-cyber-muted cursor-not-allowed border border-cyber-border"
                              : "bg-neon-cyan text-cyber-black hover:bg-neon-cyan/80 border border-neon-cyan shadow-cyber"
                          }`}
                        >
                          {importingVideo === video.path ? (
                            <>
                              <div className="animate-spin w-4 h-4 border-2 border-cyber-muted border-t-transparent inline-block mr-2"></div>
                              PROCESSING...
                            </>
                          ) : (
                            <>
                              <HiPlay className="w-4 h-4 mr-2 inline" />
                              IMPORT & PROCESS
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Instructions */}
          <div className="mt-6 p-4 bg-cyber-surface/50 border border-cyber-border">
            <div className="text-sm text-cyber-muted font-mono space-y-1">
              <div className="font-medium text-neon-yellow">INSTRUCTIONS:</div>
              <div>• Copy video files to the data/ directory</div>
              <div>• Click "IMPORT & PROCESS" to create a new session</div>
              <div>• Processing will automatically generate detections using improved AI</div>
              <div>• New sessions will appear in the sessions list after import</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoImport;
