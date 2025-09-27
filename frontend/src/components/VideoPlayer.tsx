import React, { useRef, useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Play, Pause } from 'lucide-react';

interface VideoPlayerProps {
  onTargetClick: (targetId: string) => void;
  onVideoTimeUpdate?: (time: number) => void;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ onTargetClick, onVideoTimeUpdate }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [videoUrl, setVideoUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Mock target positions for demo - will be replaced by real-time detection from backend
  const mockTargets = [
    { id: 'TGT-001', x: 20, y: 30 },
    { id: 'TGT-002', x: 60, y: 45 },
    { id: 'TGT-003', x: 35, y: 70 },
  ];

  // Function to load the hardcoded video file
  const loadVideoFile = async () => {
    setIsLoading(true);
    try {
      // Video file is now in the public directory
      const videoPath = '/2025_09_17-15_02_07_MovingObjects_44_segment_1_55.0s-80.0s.mp4';
      setVideoUrl(videoPath);
      console.log('Video path set:', videoPath);
    } catch (error) {
      console.error('Error loading video:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Load video on component mount
  useEffect(() => {
    loadVideoFile();
  }, []);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && (file.type === 'video/mp4' || file.name.toLowerCase().endsWith('.mp4'))) {
      const url = URL.createObjectURL(file);
      setVideoUrl(url);
    }
  };

  const togglePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Video Controls */}
      <Card className="p-4 bg-card border-border mb-4">
        <h3 className="hud-text font-semibold mb-3 text-accent">VIDEO FEED</h3>
        <div className="flex gap-2">
          <div className="flex-1 hud-text text-sm text-muted-foreground">
            {isLoading ? 'Loading video...' : 'Video: 2025_09_17-15_02_07_MovingObjects_44_segment_1_55.0s-80.0s.mp4'}
          </div>
          {videoUrl && !isLoading && (
            <Button
              onClick={togglePlayPause}
              variant="outline"
              size="sm"
              className="hud-text"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </Button>
          )}
        </div>
      </Card>

      {/* Video Display */}
      <div className="relative flex-1 bg-card border border-border tactical-grid">
        {videoUrl ? (
          <>
            <video
              ref={videoRef}
              src={videoUrl}
              className="w-full h-full object-contain"
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onTimeUpdate={() => {
                if (videoRef.current && onVideoTimeUpdate) {
                  onVideoTimeUpdate(videoRef.current.currentTime);
                }
              }}
              onError={(e) => {
                console.error('Video error:', e);
                console.error('Video src:', videoUrl);
              }}
              onLoadStart={() => console.log('Video load started')}
              onLoadedData={() => console.log('Video data loaded')}
              onCanPlay={() => console.log('Video can play')}
              controls
              preload="metadata"
            />
            
            {/* Target overlays */}
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
          </>
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="w-16 h-16 border-2 border-tactical-active/30 rounded-full flex items-center justify-center mb-4 mx-auto">
                <Play className="w-8 h-8 text-tactical-active/60" />
              </div>
              <p className="hud-text text-muted-foreground">Upload video file to begin analysis</p>
            </div>
          </div>
        )}

        {/* HUD overlay */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Status bar */}
          <div className="absolute top-4 left-4 hud-text text-sm">
            <div className="status-active">● LIVE</div>
            <div className="text-muted-foreground mt-1">
              VIDEO ANALYSIS ACTIVE
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
    </div>
  );
};

export default VideoPlayer;