import React, { useEffect, useMemo, useRef, useState } from "react";
import type { FrameDetections, TrackingObject } from "./types";

interface VideoCanvasProps {
  videoSrc: string;
  trackingData: FrameDetections[];
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
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationRef = useRef<number | null>(null);
  const lastFrameIndexRef = useRef(0);

  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  const preparedData = useMemo(
    () => prepareTrackingData(trackingData),
    [trackingData]
  );

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas) return;

    const handleLoadedMetadata = (): void => {
      console.log("Video loaded:", video.videoWidth, "x", video.videoHeight);
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      setDuration(video.duration || 0);
      setCurrentTime(0);
      lastFrameIndexRef.current = 0;
    };

    const handleError = (e: Event): void => {
      console.error("Video error:", e);
    };

    const handleLoadStart = (): void => {
      console.log("Video load started:", videoSrc);
    };

    const handleTimeUpdate = (): void => {
      setCurrentTime(video.currentTime);
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

    return () => {
      video.removeEventListener("loadedmetadata", handleLoadedMetadata);
      video.removeEventListener("timeupdate", handleTimeUpdate);
      video.removeEventListener("play", handlePlay);
      video.removeEventListener("pause", handlePause);
      video.removeEventListener("ended", handleEnded);
      video.removeEventListener("error", handleError);
      video.removeEventListener("loadstart", handleLoadStart);
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

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const frame = findFrameForTime(
          video.currentTime,
          preparedData.frames,
          preparedData.fps,
          lastFrameIndexRef
        );
        if (frame && frame.objects.length > 0) {
          drawDetections(ctx, frame.objects);
        }
      }

      animationRef.current = requestAnimationFrame(renderFrame);
    };

    renderFrame();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    };
  }, [videoSrc, preparedData.frames, preparedData.fps]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    video.pause();
    video.currentTime = 0;
    setIsPlaying(false);
    setCurrentTime(0);
  }, [videoSrc]);

  const togglePlayback = async (): Promise<void> => {
    const video = videoRef.current;
    if (!video) return;

    if (video.paused || video.ended) {
      try {
        await video.play();
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
    video.currentTime = nextTime;
    setCurrentTime(nextTime);
    lastFrameIndexRef.current = 0;
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
          className="w-full h-auto rounded-xl border border-gray-200 shadow-lg bg-black"
        />
        <video
          ref={videoRef}
          src={videoSrc}
          className="hidden"
          preload="metadata"
          playsInline
          crossOrigin="anonymous"
          controls={false}
        />
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <button
          onClick={togglePlayback}
          className="px-4 py-2 rounded-lg bg-gradient-to-r from-primary-500 to-primary-600 text-white shadow-md hover:from-primary-600 hover:to-primary-700 transition-all duration-200"
        >
          {isPlaying ? "Pause" : "Play"}
        </button>
        <input
          type="range"
          min={0}
          max={duration || 0}
          step={0.033}
          value={currentTime}
          onChange={handleSeek}
          className="flex-1"
        />
        <span className="text-sm text-gray-600 min-w-[120px] text-right">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>

      {preparedData.hasDetections && (
        <div className="text-sm text-gray-600">
          <p>
            📊 Total frames with detections:{" "}
            {preparedData.frames.filter((f) => f.objects.length > 0).length}
          </p>
          <p>
            🎯 Total objects detected:{" "}
            {preparedData.frames.reduce((acc, f) => acc + f.objects.length, 0)}
          </p>
        </div>
      )}
    </div>
  );
};

export default VideoCanvas;
