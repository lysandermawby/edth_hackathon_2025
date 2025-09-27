import React, { useState, useEffect } from "react";
import { createPortal } from "react-dom";
import {
  HiUpload,
  HiVideoCamera,
  HiRefresh,
  HiX,
  HiPlay,
  HiFolder,
  HiDocumentText,
  HiCheck,
} from "react-icons/hi";

interface FileItem {
  filename: string;
  path: string;
  directory: string;
  size: number;
  modified: string;
  sizeFormatted: string;
}

interface AvailableFiles {
  videos: FileItem[];
  csvFiles: FileItem[];
}

interface VideoImportProps {
  isOpen: boolean;
  onClose: () => void;
  onImportSuccess: () => void;
}

const VideoImport: React.FC<VideoImportProps> = ({
  isOpen,
  onClose,
  onImportSuccess,
}) => {
  const [availableFiles, setAvailableFiles] = useState<AvailableFiles>({
    videos: [],
    csvFiles: [],
  });
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
  const [selectedCsv, setSelectedCsv] = useState<string | null>(null);
  const [importingFiles, setImportingFiles] = useState(false);
  const [importMessage, setImportMessage] = useState<string | null>(null);
  const [importError, setImportError] = useState<string | null>(null);

  const fetchAvailableFiles = async () => {
    setLoadingFiles(true);
    setImportError(null);
    try {
      const response = await fetch("/api/files/available");
      if (!response.ok) throw new Error("Failed to fetch available files");
      const data = await response.json();
      console.log("📁 Fetched files:", data);
      setAvailableFiles(data);
    } catch (err) {
      setImportError(
        err instanceof Error ? err.message : "Failed to fetch files"
      );
    } finally {
      setLoadingFiles(false);
    }
  };

  const importFiles = async () => {
    if (!selectedVideo) {
      setImportError("Please select a video file");
      return;
    }

    setImportingFiles(true);
    setImportMessage(null);
    setImportError(null);

    try {
      console.log(
        "🎬 Importing video:",
        selectedVideo,
        "with CSV:",
        selectedCsv
      );
      const response = await fetch("/api/sessions/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          videoPath: selectedVideo,
          csvPath: selectedCsv,
          autoProcess: true,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || "Failed to import files");
      }

      const data = await response.json();
      console.log("✅ Import response:", data);

      const csvInfo = data.csv_processed ? " with telemetry data" : "";

      if (data.realtime_processing) {
        setImportMessage(
          `${data.message}${csvInfo} (Session #${data.session_id}) - Detections are being generated in real-time and stored in database. You can view progress in the sessions list.`
        );

        // Notify parent component immediately
        onImportSuccess();

        // Close modal after real-time processing starts
        setTimeout(() => {
          onClose();
          setImportMessage(null);
          setSelectedVideo(null);
          setSelectedCsv(null);
        }, 4000);
      } else {
        setImportMessage(
          `${data.message}${csvInfo} (Session #${data.session_id}, ${data.detections} detections)`
        );

        // Notify parent component
        onImportSuccess();

        // Close modal after successful import
        setTimeout(() => {
          onClose();
          setImportMessage(null);
          setSelectedVideo(null);
          setSelectedCsv(null);
        }, 3000);
      }
    } catch (err) {
      console.error("❌ Import failed:", err);
      setImportError(
        err instanceof Error ? err.message : "Failed to import files"
      );
    } finally {
      setImportingFiles(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchAvailableFiles();
      setImportMessage(null);
      setImportError(null);
      setSelectedVideo(null);
      setSelectedCsv(null);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const canImport = selectedVideo && !importingFiles;

  const modalContent = (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        width: "100vw",
        height: "100vh",
        backgroundColor: "rgba(0, 0, 0, 0.8)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 9999,
        margin: 0,
        padding: 0,
      }}
    >
      <div className="cyber-card w-full max-w-6xl mx-4 h-[80vh] flex flex-col">
        <div className="cyber-card-header">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-cyber-surface border-2 border-neon-cyan flex items-center justify-center shadow-cyber">
                <HiUpload className="text-neon-cyan text-sm text-glow" />
              </div>
              <div>
                <h3 className="font-semibold text-neon-cyan text-glow">
                  &gt;&gt; IMPORT VIDEO + TELEMETRY
                </h3>
                <p className="text-xs text-cyber-muted font-mono">
                  [DUAL_FILE_PROCESSING_PIPELINE_v2.0]
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

        <div
          className="p-6 flex-1 flex flex-col"
          style={{ height: "calc(100% - 80px)" }}
        >
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
              Found {availableFiles.videos.length} video
              {availableFiles.videos.length !== 1 ? "s" : ""} and{" "}
              {availableFiles.csvFiles.length} CSV file
              {availableFiles.csvFiles.length !== 1 ? "s" : ""}
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={fetchAvailableFiles}
                disabled={loadingFiles}
                className="px-3 py-2 bg-cyber-surface border border-neon-cyan text-neon-cyan hover:bg-cyber-border transition-colors text-sm font-mono disabled:opacity-50"
              >
                <HiRefresh className="w-4 h-4 mr-2 inline" />
                {loadingFiles ? "SCANNING..." : "REFRESH"}
              </button>
              <button
                onClick={importFiles}
                disabled={!canImport}
                className={`px-6 py-2 text-sm font-bold font-mono uppercase tracking-wider transition-all ${
                  canImport
                    ? "bg-neon-cyan text-cyber-black hover:bg-neon-cyan/80 border border-neon-cyan shadow-cyber"
                    : "bg-cyber-surface text-cyber-muted cursor-not-allowed border border-cyber-border"
                }`}
              >
                {importingFiles ? (
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

          {/* Selection Summary */}
          <div className="mb-4 p-3 bg-cyber-surface/50 border border-cyber-border">
            <div className="text-sm text-cyber-muted font-mono">
              <div className="font-medium text-neon-yellow mb-2">
                SELECTED FILES:
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-neon-cyan">VIDEO:</span>{" "}
                  {selectedVideo
                    ? selectedVideo.split("/").pop()
                    : "None selected"}
                </div>
                <div>
                  <span className="text-neon-pink">CSV:</span>{" "}
                  {selectedCsv
                    ? selectedCsv.split("/").pop()
                    : "None selected (optional)"}
                </div>
              </div>
            </div>
          </div>

          {/* File Lists */}
          <div style={{ height: "400px" }}>
            {loadingFiles ? (
              <div className="text-center py-8 text-cyber-muted">
                <div className="animate-spin w-8 h-8 border-2 border-neon-cyan border-t-transparent mx-auto mb-4"></div>
                <div className="font-medium font-mono">
                  [SCANNING_DATA_DIRECTORY...]
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-6 h-full">
                {/* Video Files Column */}
                <div className="flex flex-col h-full">
                  <div className="flex items-center gap-2 mb-4 flex-shrink-0">
                    <HiVideoCamera className="text-neon-cyan text-lg" />
                    <h4 className="font-medium text-neon-cyan font-mono">
                      VIDEO FILES
                    </h4>
                    <span className="text-xs text-cyber-muted font-mono">
                      ({availableFiles.videos.length})
                    </span>
                  </div>
                  <div
                    className="overflow-y-auto space-y-2"
                    style={{ height: "350px" }}
                  >
                    {availableFiles.videos.length === 0 ? (
                      <div className="text-center py-8 text-cyber-muted">
                        <HiFolder className="text-4xl mb-2 mx-auto opacity-50" />
                        <div className="text-sm font-mono">No videos found</div>
                      </div>
                    ) : (
                      availableFiles.videos.map((video) => (
                        <div
                          key={video.path}
                          className={`cyber-card border cursor-pointer transition-colors ${
                            selectedVideo === video.path
                              ? "border-neon-cyan bg-neon-cyan/10"
                              : "border-cyber-border hover:border-neon-cyan/50"
                          }`}
                          onClick={() => setSelectedVideo(video.path)}
                        >
                          <div className="p-3">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-3 flex-1 min-w-0">
                                <div className="w-8 h-8 bg-cyber-surface border border-neon-cyan flex items-center justify-center">
                                  <HiVideoCamera className="text-neon-cyan text-sm" />
                                </div>
                                <div className="flex-1 min-w-0">
                                  <div className="font-medium text-cyber-text truncate font-mono text-sm">
                                    {video.filename}
                                  </div>
                                  {video.directory && (
                                    <div className="text-xs text-neon-cyan/70 font-mono truncate">
                                      📁 {video.directory}
                                    </div>
                                  )}
                                  <div className="text-xs text-cyber-muted font-mono">
                                    {video.sizeFormatted} •{" "}
                                    {new Date(
                                      video.modified
                                    ).toLocaleDateString()}
                                  </div>
                                </div>
                              </div>
                              {selectedVideo === video.path && (
                                <HiCheck className="text-neon-cyan text-lg" />
                              )}
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>

                {/* CSV Files Column */}
                <div className="flex flex-col h-full">
                  <div className="flex items-center gap-2 mb-4 flex-shrink-0">
                    <HiDocumentText className="text-neon-pink text-lg" />
                    <h4 className="font-medium text-neon-pink font-mono">
                      TELEMETRY (CSV)
                    </h4>
                    <span className="text-xs text-cyber-muted font-mono">
                      ({availableFiles.csvFiles.length}) OPTIONAL
                    </span>
                  </div>
                  <div
                    className="overflow-y-auto space-y-2"
                    style={{ height: "350px" }}
                  >
                    {/* None Selected Option */}
                    <div
                      className={`cyber-card border cursor-pointer transition-colors ${
                        selectedCsv === null
                          ? "border-neon-pink bg-neon-pink/10"
                          : "border-cyber-border hover:border-neon-pink/50"
                      }`}
                      onClick={() => setSelectedCsv(null)}
                    >
                      <div className="p-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 bg-cyber-surface border border-neon-pink flex items-center justify-center">
                              <HiX className="text-neon-pink text-sm" />
                            </div>
                            <div>
                              <div className="font-medium text-cyber-text font-mono text-sm">
                                No CSV File
                              </div>
                              <div className="text-xs text-cyber-muted font-mono">
                                Import video only (no telemetry)
                              </div>
                            </div>
                          </div>
                          {selectedCsv === null && (
                            <HiCheck className="text-neon-pink text-lg" />
                          )}
                        </div>
                      </div>
                    </div>

                    {availableFiles.csvFiles.length === 0 ? (
                      <div className="text-center py-8 text-cyber-muted">
                        <HiFolder className="text-4xl mb-2 mx-auto opacity-50" />
                        <div className="text-sm font-mono">
                          No CSV files found
                        </div>
                      </div>
                    ) : (
                      availableFiles.csvFiles.map((csv) => (
                        <div
                          key={csv.path}
                          className={`cyber-card border cursor-pointer transition-colors ${
                            selectedCsv === csv.path
                              ? "border-neon-pink bg-neon-pink/10"
                              : "border-cyber-border hover:border-neon-pink/50"
                          }`}
                          onClick={() => setSelectedCsv(csv.path)}
                        >
                          <div className="p-3">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-3 flex-1 min-w-0">
                                <div className="w-8 h-8 bg-cyber-surface border border-neon-pink flex items-center justify-center">
                                  <HiDocumentText className="text-neon-pink text-sm" />
                                </div>
                                <div className="flex-1 min-w-0">
                                  <div className="font-medium text-cyber-text truncate font-mono text-sm">
                                    {csv.filename}
                                  </div>
                                  {csv.directory && (
                                    <div className="text-xs text-neon-pink/70 font-mono truncate">
                                      📁 {csv.directory}
                                    </div>
                                  )}
                                  <div className="text-xs text-cyber-muted font-mono">
                                    {csv.sizeFormatted} •{" "}
                                    {new Date(
                                      csv.modified
                                    ).toLocaleDateString()}
                                  </div>
                                </div>
                              </div>
                              {selectedCsv === csv.path && (
                                <HiCheck className="text-neon-pink text-lg" />
                              )}
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Instructions */}
          <div className="mt-6 p-4 bg-cyber-surface/50 border border-cyber-border">
            <div className="text-sm text-cyber-muted font-mono space-y-1">
              <div className="font-medium text-neon-yellow">INSTRUCTIONS:</div>
              <div>
                • <span className="text-neon-cyan">STEP 1:</span> Select a video
                file from the left column
              </div>
              <div>
                • <span className="text-neon-pink">STEP 2:</span> Optionally
                select a CSV file for telemetry data
              </div>
              <div>
                • <span className="text-neon-green">STEP 3:</span> Click "IMPORT
                & PROCESS" for real-time detection generation
              </div>
              <div>
                • <span className="text-neon-yellow">REAL-TIME:</span>{" "}
                Detections are generated and stored in SQLite as they're
                processed
              </div>
              <div>
                • CSV files contain drone telemetry (GPS, attitude, gimbal data)
              </div>
              <div>
                • View progress in sessions list - detection count updates live
                during processing
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return createPortal(modalContent, document.body);
};

export default VideoImport;
