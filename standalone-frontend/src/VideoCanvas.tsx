import React, { useEffect, useMemo, useRef, useState } from "react";
import type { FrameDetections, TrackingObject } from "./types";

interface VideoCanvasProps {
  videoSrc: string;
  trackingData: FrameDetections[];
  onTimeUpdate?: (currentTime: number) => void;
  onDurationLoad?: (duration: number) => void;
}

interface PreparedTrackingData {
  frames: FrameDetections[];
  fps: number;
  hasDetections: boolean;
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

const formatTime = (time: number): string => {
  if (!Number.isFinite(time)) return "0:00";
  const minutes = Math.floor(time / 60);
  const seconds = Math.floor(time % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
};

const colourForObject = (obj: TrackingObject): string => {
  const base = obj.tracker_id ?? obj.class_id ?? 0;
  const index = Math.abs(base) % DETECTION_COLOURS.length;
  return DETECTION_COLOURS[index];
};

const prepareTrackingData = (data: FrameDetections[]): PreparedTrackingData => {
  if (!data || data.length === 0) {
    return {
      frames: [],
      fps: 30,
      hasDetections: false,
    };
  }

  const frames = [...data].sort((a, b) => a.timestamp - b.timestamp);

  let fps = 30;
  for (let i = 1; i < frames.length; i += 1) {
    const delta = frames[i].timestamp - frames[i - 1].timestamp;
    if (delta > 0.0001) {
      fps = 1 / delta;
      break;
    }
  }

  const hasDetections = frames.some((frame) => frame.objects.length > 0);

  return { frames, fps, hasDetections };
};

const isPointInBoundingBox = (
  x: number,
  y: number,
  bbox: { x1: number; y1: number; x2: number; y2: number }
): boolean => {
  return x >= bbox.x1 && x <= bbox.x2 && y >= bbox.y1 && y <= bbox.y2;
};

const findFrameForTime = (
  time: number,
  frames: FrameDetections[],
  fps: number,
  lastIndex: React.MutableRefObject<number>
): FrameDetections | null => {
  if (frames.length === 0) return null;

  const tolerance = 1 / Math.max(fps, 1);

  let index = lastIndex.current;
  if (index >= frames.length) {
    index = frames.length - 1;
  }

  if (time + tolerance < frames[index].timestamp) {
    index = 0;
  }

  while (
    index < frames.length - 1 &&
    frames[index + 1].timestamp <= time + tolerance
  ) {
    index += 1;
  }

  lastIndex.current = index;
  return frames[index];
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
    ctx.lineWidth = 2;
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
      const padding = 6;
      const textHeight = 20;
      const labelWidth = metrics.width + padding * 2;
      const labelX = x;
      const labelY = y - textHeight - 4;
      const drawAbove = labelY > 0;
      const boxY = drawAbove ? labelY : y + height + 4;

      ctx.fillStyle = `${colour}CC`;
      ctx.fillRect(labelX, boxY, labelWidth, textHeight);

      ctx.fillStyle = "#0A0A0A";
      ctx.fillText(label, labelX + padding, boxY + textHeight - 6);
    }
  });
};

const VideoCanvas: React.FC<VideoCanvasProps> = ({
  videoSrc,
  trackingData,
  onTimeUpdate,
  onDurationLoad,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationRef = useRef<number | null>(null);
  const lastFrameIndexRef = useRef(0);
  const preparedDataRef = useRef<PreparedTrackingData>({
    frames: [],
    fps: 30,
    hasDetections: false,
  });

  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [hoveredObject, setHoveredObject] = useState<TrackingObject | null>(
    null
  );
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isSeeking, setIsSeeking] = useState(false);

  const preparedData = useMemo(
    () => prepareTrackingData(trackingData),
    [trackingData]
  );

  // Update the ref whenever preparedData changes
  useEffect(() => {
    preparedDataRef.current = preparedData;
  }, [preparedData]);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas) return;

    const handleLoadedMetadata = (): void => {
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const videoDuration = video.duration || 0;
      setDuration(videoDuration);
      setCurrentTime(0);
      lastFrameIndexRef.current = 0;

      // Notify parent component of duration
      if (onDurationLoad) {
        onDurationLoad(videoDuration);
      }
    };

    const handleError = (e: Event): void => {
      console.error("Video error:", e);
    };

    const handleLoadStart = (): void => {};

    const handleCanPlay = (): void => {};

    const handleLoadedData = (): void => {};

    const handleTimeUpdate = (): void => {
      if (!isSeeking) {
        const time = video.currentTime;
        setCurrentTime(time);

        // Notify parent component of time updates
        if (onTimeUpdate) {
          onTimeUpdate(time);
        }
      }
    };

    const handlePlay = (): void => setIsPlaying(true);
    const handlePause = (): void => setIsPlaying(false);
    const handleEnded = (): void => setIsPlaying(false);

    video.addEventListener("loadedmetadata", handleLoadedMetadata);
    video.addEventListener("timeupdate", handleTimeUpdate);
    video.addEventListener("play", handlePlay);
    video.addEventListener("pause", handlePause);
    video.addEventListener("ended", handleEnded);
    video.addEventListener("error", handleError);
    video.addEventListener("loadstart", handleLoadStart);
    video.addEventListener("canplay", handleCanPlay);
    video.addEventListener("loadeddata", handleLoadedData);

    return () => {
      video.removeEventListener("loadedmetadata", handleLoadedMetadata);
      video.removeEventListener("timeupdate", handleTimeUpdate);
      video.removeEventListener("play", handlePlay);
      video.removeEventListener("pause", handlePause);
      video.removeEventListener("ended", handleEnded);
      video.removeEventListener("error", handleError);
      video.removeEventListener("loadstart", handleLoadStart);
      video.removeEventListener("canplay", handleCanPlay);
      video.removeEventListener("loadeddata", handleLoadedData);
    };
  }, [videoSrc]);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const renderFrame = (): void => {
      if (video.readyState >= 2) {
        if (
          canvas.width !== video.videoWidth ||
          canvas.height !== video.videoHeight
        ) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Only draw detections if we have tracking data
        if (
          preparedDataRef.current.frames &&
          preparedDataRef.current.frames.length > 0
        ) {
          const frame = findFrameForTime(
            video.currentTime,
            preparedDataRef.current.frames,
            preparedDataRef.current.fps,
            lastFrameIndexRef
          );
          if (frame && frame.objects.length > 0) {
            drawDetections(ctx, frame.objects);
          }
        }
      }

      // Only continue animation if video is playing
      if (!video.paused && !video.ended && video.readyState >= 2) {
        animationRef.current = requestAnimationFrame(renderFrame);
      } else {
        animationRef.current = null;
      }
    };

    // Always render initial frame when video is ready
    if (video.readyState >= 2) {
      renderFrame();
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    };
  }, [videoSrc]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    // Only reset when video source actually changes
    video.pause();
    video.currentTime = 0;
    setIsPlaying(false);
    setCurrentTime(0);
    lastFrameIndexRef.current = 0;
  }, [videoSrc]);

  // Separate effect for handling seek updates without restarting animation
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || !isSeeking) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Render single frame when seeking
    if (video.readyState >= 2) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Only draw detections if we have tracking data
      if (
        preparedDataRef.current.frames &&
        preparedDataRef.current.frames.length > 0
      ) {
        const frame = findFrameForTime(
          video.currentTime,
          preparedDataRef.current.frames,
          preparedDataRef.current.fps,
          lastFrameIndexRef
        );
        if (frame && frame.objects.length > 0) {
          drawDetections(ctx, frame.objects);
        }
      }
    }
  }, [currentTime, isSeeking]);

  const togglePlayback = async (): Promise<void> => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    if (video.paused || video.ended) {
      try {
        await video.play();

        // Start animation loop for playing video
        if (!animationRef.current && video.readyState >= 2) {
          const ctx = canvas.getContext("2d");
          if (ctx) {
            const renderFrame = (): void => {
              if (video.readyState >= 2) {
                if (
                  canvas.width !== video.videoWidth ||
                  canvas.height !== video.videoHeight
                ) {
                  canvas.width = video.videoWidth;
                  canvas.height = video.videoHeight;
                }

                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                // Only draw detections if we have tracking data
                if (
                  preparedDataRef.current.frames &&
                  preparedDataRef.current.frames.length > 0
                ) {
                  const frame = findFrameForTime(
                    video.currentTime,
                    preparedDataRef.current.frames,
                    preparedDataRef.current.fps,
                    lastFrameIndexRef
                  );
                  if (frame && frame.objects.length > 0) {
                    drawDetections(ctx, frame.objects);
                  }
                }
              }

              // Only continue animation if video is playing
              if (!video.paused && !video.ended && video.readyState >= 2) {
                animationRef.current = requestAnimationFrame(renderFrame);
              } else {
                animationRef.current = null;
              }
            };

            animationRef.current = requestAnimationFrame(renderFrame);
          }
        }
      } catch (error) {
        console.error("Failed to play video", error);
      }
    } else {
      video.pause();
    }
  };

  const handleSeek = (event: React.ChangeEvent<HTMLInputElement>): void => {
    const video = videoRef.current;
    if (!video) return;

    const nextTime = Number(event.target.value);
    setIsSeeking(true);
    setCurrentTime(nextTime);
    video.currentTime = nextTime;
    lastFrameIndexRef.current = 0;

    // Clear seeking flag after a short delay to allow video to update
    setTimeout(() => {
      setIsSeeking(false);
    }, 100);
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    // Update mouse position for tooltip
    setMousePosition({ x: event.clientX, y: event.clientY });

    // Find the current frame and check for object intersections
    const frame = findFrameForTime(
      video.currentTime,
      preparedDataRef.current.frames,
      preparedDataRef.current.fps,
      lastFrameIndexRef
    );

    if (frame && frame.objects.length > 0) {
      // Find the topmost object that contains the mouse point
      // (iterate in reverse to get the last drawn object, which appears on top)
      const hoveredObj = frame.objects
        .slice()
        .reverse()
        .find((obj) => obj.bbox && isPointInBoundingBox(x, y, obj.bbox));

      setHoveredObject(hoveredObj || null);
    } else {
      setHoveredObject(null);
    }
  };

  const handleMouseLeave = () => {
    setHoveredObject(null);
  };

  return (
    <div className="space-y-4">
      <div className="relative w-full max-w-4xl">
        {preparedData.frames.length === 0 && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-gray-900/50 text-white text-sm">
            No tracking data found for this video. Showing raw frames.
          </div>
        )}
        <canvas
          ref={canvasRef}
          className="w-full h-auto rounded-xl border border-gray-200 shadow-lg bg-black cursor-crosshair"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        />
        <video
          ref={videoRef}
          src={videoSrc}
          className="hidden"
          preload="metadata"
          playsInline
          controls={false}
        />
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <button
          onClick={togglePlayback}
          className="px-4 py-2 bg-neon-cyan text-cyber-black font-bold font-mono uppercase tracking-wider transition-all duration-200 border border-neon-cyan hover:bg-neon-cyan/80 shadow-cyber"
        >
          {isPlaying ? "PAUSE" : "PLAY"}
        </button>
        <input
          type="range"
          min={0}
          max={duration || 0}
          step={0.033}
          value={currentTime}
          onChange={handleSeek}
          className="flex-1 h-2 bg-cyber-surface border border-cyber-border appearance-none cursor-pointer cyber-slider"
        />
        <span className="text-sm text-neon-cyan font-mono min-w-[120px] text-right">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
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
              {hoveredObject.center && (
                <div className="text-gray-300">
                  <span className="text-cyan-400">Center:</span> (
                  {Math.round(hoveredObject.center.x)},{" "}
                  {Math.round(hoveredObject.center.y)})
                </div>
              )}
            </div>
            {/* Arrow pointing to the object */}
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
    </div>
  );
};

export default VideoCanvas;
