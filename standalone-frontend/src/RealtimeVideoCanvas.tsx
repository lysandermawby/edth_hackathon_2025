import React, { useEffect, useRef, useState, useCallback } from "react";
import { HiStop, HiRefresh, HiVideoCamera } from "react-icons/hi";
import { BiWifi } from "react-icons/bi";
import { MdVideocam, MdVideoFile } from "react-icons/md";
import type { TrackingObject } from "./types";

interface RealtimeVideoCanvasProps {
  onDetectionData?: (data: DetectionFrame) => void;
}

interface DetectionFrame {
  frame_number: number;
  timestamp: number;
  detections: TrackingObject[];
  fps: number;
  session_id: number;
}

interface ServerMessage {
  type: string;
  [key: string]: any;
}

const DETECTION_COLOURS = [
  "#FF6B6B",
  "#4ECDC4",
  "#A06CD5",
  "#FFD166",
  "#06D6A0",
  "#118AB2",
  "#EF476F",
  "#073B4C",
  "#F78104",
  "#7B2CBF",
];

const colourForObject = (obj: TrackingObject): string => {
  const base = obj.tracker_id ?? obj.class_id ?? 0;
  const index = Math.abs(base) % DETECTION_COLOURS.length;
  return DETECTION_COLOURS[index];
};

const isPointInBoundingBox = (
  x: number,
  y: number,
  bbox: { x1: number; y1: number; x2: number; y2: number }
): boolean => {
  return x >= bbox.x1 && x <= bbox.x2 && y >= bbox.y1 && y <= bbox.y2;
};

const drawDetections = (
  ctx: CanvasRenderingContext2D,
  objects: TrackingObject[]
): void => {
  objects.forEach((obj) => {
    if (!obj.bbox) return;

    const colour = colourForObject(obj);
    const x = obj.bbox.x1;
    const y = obj.bbox.y1;
    const width = obj.bbox.x2 - obj.bbox.x1;
    const height = obj.bbox.y2 - obj.bbox.y1;

    ctx.strokeStyle = colour;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.rect(x, y, width, height);
    ctx.stroke();

    const labelParts: string[] = [];
    if (obj.class_name) {
      labelParts.push(obj.class_name);
    }
    if (Number.isFinite(obj.tracker_id)) {
      labelParts.push(`#${obj.tracker_id}`);
    }
    if (typeof obj.confidence === "number") {
      labelParts.push(`${Math.round(obj.confidence * 100)}%`);
    }

    if (labelParts.length > 0) {
      const label = labelParts.join(" ");
      ctx.font = "16px Inter, system-ui, sans-serif";
      const metrics = ctx.measureText(label);
      const padding = 8;
      const textHeight = 22;
      const labelWidth = metrics.width + padding * 2;
      const labelX = x;
      const labelY = y - textHeight - 4;
      const drawAbove = labelY > 0;
      const boxY = drawAbove ? labelY : y + height + 4;

      ctx.fillStyle = `${colour}DD`;
      ctx.fillRect(labelX, boxY, labelWidth, textHeight);

      ctx.fillStyle = "#FFFFFF";
      ctx.fillText(label, labelX + padding, boxY + textHeight - 6);
    }
  });
};

const RealtimeVideoCanvas: React.FC<RealtimeVideoCanvasProps> = ({
  onDetectionData,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<string>("Disconnected");
  const [fps, setFps] = useState<number>(0);
  const [objectCount, setObjectCount] = useState<number>(0);
  const [frameNumber, setFrameNumber] = useState<number>(0);
  const [currentDetections, setCurrentDetections] = useState<TrackingObject[]>(
    []
  );
  const [hoveredObject, setHoveredObject] = useState<TrackingObject | null>(
    null
  );
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [videoInfo, setVideoInfo] = useState<{
    width: number;
    height: number;
    source: string;
  } | null>(null);
  const [selectedCameraId, setSelectedCameraId] = useState<number>(0);
  const [availableCameras, setAvailableCameras] = useState<number[]>([
    0, 1, 2, 3,
  ]); // Default camera options

  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket("ws://localhost:8765");
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("Connected to real-time detection server");
        setConnected(true);
        setStatus("Connected");
        // Request available cameras
        ws.send(JSON.stringify({ command: "list_cameras" }));
      };

      ws.onmessage = (event) => {
        try {
          const message: ServerMessage = JSON.parse(event.data);

          switch (message.type) {
            case "frame":
              handleFrameMessage(message);
              break;

            case "video_info":
              setVideoInfo({
                width: message.width,
                height: message.height,
                source: message.source,
              });
              setStatus(
                `Video: ${message.source} (${message.width}x${message.height})`
              );
              break;

            case "status":
              setStatus(message.message);
              break;

            case "error":
              setStatus(`Error: ${message.message}`);
              console.error("Server error:", message.message);
              break;

            case "video_ended":
              setStatus("Video ended");
              break;

            case "camera_list":
              if (message.cameras && Array.isArray(message.cameras)) {
                setAvailableCameras(message.cameras);
                // If current selected camera is not available, switch to first available
                if (
                  !message.cameras.includes(selectedCameraId) &&
                  message.cameras.length > 0
                ) {
                  setSelectedCameraId(message.cameras[0]);
                }
              }
              break;

            case "pong":
              // Handle ping response if needed
              break;

            default:
              console.log("Unknown message type:", message.type);
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
        }
      };

      ws.onclose = () => {
        console.log("Disconnected from real-time detection server");
        setConnected(false);
        setStatus("Disconnected");
        setCurrentDetections([]);
        setHoveredObject(null);
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setStatus("Connection error");
      };
    } catch (error) {
      console.error("Failed to connect to WebSocket:", error);
      setStatus("Failed to connect");
    }
  }, []);

  const handleFrameMessage = useCallback(
    (message: any) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      // Update state
      setFps(message.fps || 0);
      setObjectCount(message.detections?.length || 0);
      setFrameNumber(message.frame_number || 0);
      setCurrentDetections(message.detections || []);

      // Call callback with detection data
      if (onDetectionData) {
        onDetectionData({
          frame_number: message.frame_number,
          timestamp: message.timestamp,
          detections: message.detections || [],
          fps: message.fps || 0,
          session_id: message.session_id,
        });
      }

      // Create image from base64 data
      if (message.frame_data) {
        const img = new Image();
        img.onload = () => {
          const ctx = canvas.getContext("2d");
          if (!ctx) return;

          // Set canvas size to match image
          canvas.width = img.width;
          canvas.height = img.height;

          // Draw the frame
          ctx.drawImage(img, 0, 0);

          // Draw additional detection overlays if needed
          if (message.detections && message.detections.length > 0) {
            drawDetections(ctx, message.detections);
          }
        };
        img.src = `data:image/jpeg;base64,${message.frame_data}`;
        imageRef.current = img;
      }
    },
    [onDetectionData]
  );

  const sendCommand = useCallback((command: string, data: any = {}) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command, ...data }));
    }
  }, []);

  const startCamera = useCallback(
    (cameraId?: number) => {
      const cameraToUse = cameraId !== undefined ? cameraId : selectedCameraId;
      sendCommand("start_camera", { camera_id: cameraToUse });
      setStatus(`Starting camera ${cameraToUse}...`);
    },
    [sendCommand, selectedCameraId]
  );

  const startVideo = useCallback(
    (videoPath: string) => {
      sendCommand("start_video", { video_path: videoPath });
      setStatus("Starting video...");
    },
    [sendCommand]
  );

  const stopProcessing = useCallback(() => {
    sendCommand("stop");
    setStatus("Stopping...");
  }, [sendCommand]);

  const refreshCameras = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      sendCommand("list_cameras");
      setStatus("Detecting cameras...");
    }
  }, [sendCommand]);

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    // Update mouse position for tooltip
    setMousePosition({ x: event.clientX, y: event.clientY });

    // Find hovered object
    const hoveredObj = currentDetections
      .slice()
      .reverse()
      .find((obj) => obj.bbox && isPointInBoundingBox(x, y, obj.bbox));

    setHoveredObject(hoveredObj || null);
  };

  const handleMouseLeave = () => {
    setHoveredObject(null);
  };

  useEffect(() => {
    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);

  return (
    <div className="space-y-6">
      {/* Connection Status Panel */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div
                className={`status-indicator ${
                  connected ? "status-connected" : "status-disconnected"
                }`}
              >
                <div
                  className={`w-2 h-2 rounded-full ${
                    connected ? "bg-success-500 animate-pulse" : "bg-accent-500"
                  }`}
                />
                {connected ? "Connected" : "Disconnected"}
              </div>
              <div className="text-sm text-neutral-600">{status}</div>
            </div>

            <div className="flex gap-2">
              <button
                onClick={connectWebSocket}
                disabled={connected}
                className="btn btn-primary btn-sm"
              >
                <BiWifi className="w-4 h-4" />
                Connect
              </button>
              <button
                onClick={stopProcessing}
                disabled={!connected}
                className="btn btn-accent btn-sm"
              >
                <HiStop className="w-4 h-4" />
                Stop
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-3 gap-4">
        <div className="stats-card bg-gradient-to-br from-primary-50 to-primary-100 border-primary-200">
          <div className="text-3xl font-bold text-primary-600">
            {fps.toFixed(1)}
          </div>
          <div className="text-sm text-primary-800 font-medium">FPS</div>
          <div className="text-xs text-primary-600 mt-1">Frames per second</div>
        </div>
        <div className="stats-card bg-gradient-to-br from-success-50 to-success-100 border-success-200">
          <div className="text-3xl font-bold text-success-600">
            {objectCount}
          </div>
          <div className="text-sm text-success-800 font-medium">Objects</div>
          <div className="text-xs text-success-600 mt-1">
            Currently detected
          </div>
        </div>
        <div className="stats-card bg-gradient-to-br from-secondary-50 to-secondary-100 border-secondary-200">
          <div className="text-3xl font-bold text-secondary-600">
            {frameNumber}
          </div>
          <div className="text-sm text-secondary-800 font-medium">Frame</div>
          <div className="text-xs text-secondary-600 mt-1">Current frame</div>
        </div>
      </div>

      {/* Controls Panel */}
      <div className="card">
        <div className="card-header">
          <h4 className="font-semibold text-neutral-900">Input Sources</h4>
        </div>
        <div className="card-body">
          {/* Quick Start Controls */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Camera Feed
              </label>
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <label
                    htmlFor="camera-select"
                    className="text-xs text-gray-600 whitespace-nowrap"
                  >
                    Camera:
                  </label>
                  <select
                    id="camera-select"
                    value={selectedCameraId}
                    onChange={(e) =>
                      setSelectedCameraId(Number(e.target.value))
                    }
                    disabled={!connected}
                    className="flex-1 px-2 py-1 text-sm border border-gray-300 rounded disabled:bg-gray-100"
                  >
                    {availableCameras.map((cameraId) => (
                      <option key={cameraId} value={cameraId}>
                        Camera {cameraId}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={refreshCameras}
                    disabled={!connected}
                    className="px-2 py-1 text-xs rounded border border-gray-300 hover:bg-gray-50 disabled:bg-gray-100"
                    title="Detect available cameras"
                  >
                    <HiRefresh className="w-3 h-3" />
                  </button>
                </div>
                <div className="text-xs text-gray-500">
                  {availableCameras.length === 1
                    ? "1 camera detected"
                    : `${availableCameras.length} cameras detected`}
                </div>
                <button
                  onClick={() => startCamera()}
                  disabled={!connected}
                  className="w-full px-4 py-2 rounded bg-green-500 text-white disabled:bg-gray-400 flex items-center justify-center gap-2"
                >
                  <MdVideocam className="w-4 h-4" />
                  Start Camera {selectedCameraId}
                </button>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Video File
              </label>
              <button
                onClick={() => startVideo("data/Individual_2.mp4")}
                disabled={!connected}
                className="w-full px-4 py-2 rounded bg-purple-500 text-white disabled:bg-gray-400 flex items-center justify-center gap-2"
              >
                <MdVideoFile className="w-4 h-4" />
                Start Sample Video
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 text-center">
        <div className="bg-blue-100 rounded-lg p-3">
          <div className="text-2xl font-bold text-blue-600">
            {fps.toFixed(1)}
          </div>
          <div className="text-sm text-blue-800">FPS</div>
        </div>
        <div className="bg-green-100 rounded-lg p-3">
          <div className="text-2xl font-bold text-green-600">{objectCount}</div>
          <div className="text-sm text-green-800">Objects</div>
        </div>
        <div className="bg-purple-100 rounded-lg p-3">
          <div className="text-2xl font-bold text-purple-600">
            {frameNumber}
          </div>
          <div className="text-sm text-purple-800">Frame #</div>
        </div>
      </div>

      {/* Video Canvas */}
      <div className="relative">
        <canvas
          ref={canvasRef}
          className="w-full h-auto rounded-xl border border-gray-200 shadow-lg bg-black cursor-crosshair"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        />

        {!connected && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900/50 text-white text-lg">
            Connect to start real-time detection
          </div>
        )}
      </div>

      {/* Detection Tooltip */}
      {hoveredObject && (
        <div
          className="fixed z-50 pointer-events-none"
          style={{
            left: mousePosition.x + 10,
            top: mousePosition.y - 10,
            transform:
              mousePosition.x > window.innerWidth - 200
                ? "translateX(-100%)"
                : "none",
          }}
        >
          <div className="bg-gray-900 text-white rounded-lg shadow-xl border border-gray-600 p-3 max-w-xs">
            <div className="space-y-1 text-sm">
              <div className="font-semibold text-blue-300">
                🎯 {hoveredObject.class_name}
              </div>
              {hoveredObject.tracker_id && (
                <div className="text-gray-300">
                  <span className="text-green-400">ID:</span> #
                  {hoveredObject.tracker_id}
                </div>
              )}
              <div className="text-gray-300">
                <span className="text-yellow-400">Confidence:</span>{" "}
                {Math.round(hoveredObject.confidence * 100)}%
              </div>
              {hoveredObject.bbox && (
                <>
                  <div className="text-gray-300">
                    <span className="text-purple-400">Position:</span> (
                    {Math.round(hoveredObject.bbox.x1)},{" "}
                    {Math.round(hoveredObject.bbox.y1)})
                  </div>
                  <div className="text-gray-300">
                    <span className="text-orange-400">Size:</span>{" "}
                    {Math.round(hoveredObject.bbox.x2 - hoveredObject.bbox.x1)}{" "}
                    ×{" "}
                    {Math.round(hoveredObject.bbox.y2 - hoveredObject.bbox.y1)}
                  </div>
                </>
              )}
            </div>
            <div
              className="absolute w-0 h-0 border-l-[6px] border-r-[6px] border-t-[6px] border-l-transparent border-r-transparent border-t-gray-900"
              style={{
                left: "20px",
                bottom: "-6px",
              }}
            />
          </div>
        </div>
      )}

      {/* Video Info */}
      {videoInfo && (
        <div className="text-sm text-tactical-muted flex items-center gap-2">
          <HiVideoCamera className="w-4 h-4" />
          <span>
            Source: {videoInfo.source} | Resolution: {videoInfo.width}x
            {videoInfo.height}
          </span>
        </div>
      )}
    </div>
  );
};

export default RealtimeVideoCanvas;
