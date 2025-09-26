export interface OutputProps {
  addToOutput: (message: string, isError?: boolean) => void;
}

export interface OutputSectionProps {
  output: string;
  onClear: () => void;
}

export interface TrackingBoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface TrackingObject {
  tracker_id: number | null;
  class_id: number | null;
  class_name: string;
  confidence?: number | null;
  bbox: TrackingBoundingBox;
  center?: {
    x: number;
    y: number;
  };
}

export interface FrameDetections {
  frame_number: number;
  timestamp: number;
  objects: TrackingObject[];
}
