import React from 'react';

interface VideoDisplayProps {
  onTargetClick: (targetId: string) => void;
}

const VideoDisplay: React.FC<VideoDisplayProps> = ({ onTargetClick }) => {
  // Mock target positions for demo
  const mockTargets = [
    { id: 'TGT-001', x: 20, y: 30 },
    { id: 'TGT-002', x: 60, y: 45 },
    { id: 'TGT-003', x: 35, y: 70 },
  ];

  return (
    <div className="relative w-full h-full bg-card border border-border tactical-grid">
      {/* Video overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-background/20 to-background/40" />
      
      {/* Crosshairs */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="relative">
          <div className="w-8 h-px bg-tactical-active opacity-60"></div>
          <div className="w-px h-8 bg-tactical-active opacity-60 absolute top-0 left-4 -translate-y-4"></div>
        </div>
      </div>

      {/* Mock target markers */}
      {mockTargets.map((target) => (
        <button
          key={target.id}
          onClick={() => onTargetClick(target.id)}
          className="absolute w-6 h-6 border-2 border-tactical-active bg-tactical-active/20 hover:bg-tactical-active/40 transition-all duration-200 pulse-glow"
          style={{
            left: `${target.x}%`,
            top: `${target.y}%`,
            transform: 'translate(-50%, -50%)'
          }}
        >
          <span className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs hud-text status-active whitespace-nowrap">
            {target.id}
          </span>
        </button>
      ))}

      {/* HUD overlay */}
      <div className="absolute inset-0 pointer-events-none">
        {/* Status bar */}
        <div className="absolute top-4 left-4 hud-text text-sm">
          <div className="status-active">● REC</div>
          <div className="text-muted-foreground mt-1">
            FPS: 30 | RES: 1920x1080
          </div>
        </div>

        {/* Timestamp */}
        <div className="absolute top-4 right-4 hud-text text-sm text-muted-foreground">
          {new Date().toLocaleTimeString('en-US', { hour12: false })}
        </div>

        {/* Scan line effect */}
        <div className="scan-line absolute inset-0" />
      </div>
    </div>
  );
};

export default VideoDisplay;