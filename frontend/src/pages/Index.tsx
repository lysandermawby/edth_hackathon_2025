import React, { useState } from 'react';
import VideoPlayer from '@/components/VideoPlayer';
import TargetList from '@/components/TargetList';
import TargetDetailsModal from '@/components/TargetDetailsModal';
import MiniMap from '@/components/MiniMap';
import { Target, Asset } from '@/types/target';

// Mock data
const mockTargets: Target[] = [
  {
    id: 'TGT-001',
    status: 'active',
    position: { x: 123.45, y: 456.78, z: 15.2 },
    velocity: { speed: 12.5, direction: 45, vertical: -2.1 },
    dimensions: { width: 2.5, height: 1.8, length: 4.2 },
    scale: 1.25,
    confidence: 92,
    lastSeen: new Date(),
    classification: 'Vehicle',
    trackingDuration: 145,
    occlusionCount: 3,
    reidentificationScore: 87
  },
  {
    id: 'TGT-002',
    status: 'tracking',
    position: { x: 67.89, y: 234.56 },
    velocity: { speed: 8.3, direction: 120 },
    dimensions: { width: 1.2, height: 1.7 },
    scale: 0.95,
    confidence: 78,
    lastSeen: new Date(Date.now() - 5000),
    classification: 'Personnel',
    trackingDuration: 89,
    occlusionCount: 1,
    reidentificationScore: 94
  },
  {
    id: 'TGT-003',
    status: 'identified',
    position: { x: 345.67, y: 123.45, z: 8.5 },
    velocity: { speed: 15.7, direction: 280, vertical: 1.2 },
    dimensions: { width: 3.8, height: 2.1, length: 6.5 },
    scale: 1.45,
    confidence: 96,
    lastSeen: new Date(Date.now() - 2000),
    classification: 'Aircraft',
    trackingDuration: 234,
    occlusionCount: 0,
    reidentificationScore: 98
  },
  {
    id: 'TGT-004',
    status: 'lost',
    position: { x: 789.12, y: 567.89 },
    velocity: { speed: 5.2, direction: 195 },
    dimensions: { width: 1.5, height: 1.6 },
    scale: 0.78,
    confidence: 45,
    lastSeen: new Date(Date.now() - 30000),
    classification: 'Unknown',
    trackingDuration: 56,
    occlusionCount: 8,
    reidentificationScore: 23
  }
];

const mockAsset: Asset = {
  id: 'UAV-ALPHA-01',
  type: 'drone',
  position: {
    latitude: 34.0522,
    longitude: -118.2437,
    altitude: 1200
  },
  heading: 135,
  status: 'operational'
};

const Index = () => {
  const [selectedTarget, setSelectedTarget] = useState<Target | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [hasTargets, setHasTargets] = useState(false); // Track if targets detected
  const [videoTime, setVideoTime] = useState(0); // Current video time in seconds

  const handleTargetClick = (targetId: string) => {
    // Simulate target detection
    setHasTargets(true);
    const target = mockTargets.find(t => t.id === targetId);
    if (target) {
      setSelectedTarget(target);
      setIsModalOpen(true);
    }
  };

  const handleTargetSelect = (target: Target) => {
    setSelectedTarget(target);
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setSelectedTarget(null);
  };

  return (
    <div className="min-h-screen bg-background text-foreground" style={{backgroundColor: '#0a0a0a', color: '#ffffff'}}>
      {/* Header */}
      <header className="h-16 border-b border-border bg-card/50 flex items-center px-6" style={{backgroundColor: '#1a1a1a', borderColor: '#333333'}}>
        <div className="flex items-center gap-4">
          <div className="w-2 h-2 bg-tactical-active rounded-full pulse-glow"></div>
          <h1 className="hud-text text-xl font-bold text-foreground">
            TACTICAL VIDEO ANALYSIS SYSTEM
          </h1>
        </div>
        <div className="ml-auto hud-text text-sm text-muted-foreground">
          CLASSIFICATION: UNCLASSIFIED | SYS-ONLINE
        </div>
      </header>

      {/* Main Layout */}
      <div className="flex h-[calc(100vh-4rem)]" style={{backgroundColor: '#0f0f0f', minHeight: 'calc(100vh - 4rem)'}}>
        {/* Left Sidebar - Target List (conditional) */}
        {hasTargets && (
          <div className="w-80 border-r border-border bg-card/30">
            <TargetList
              targets={mockTargets}
              onTargetSelect={handleTargetSelect}
              selectedTargetId={selectedTarget?.id}
            />
          </div>
        )}

        {/* Main Content - Video Display */}
        <div className="flex-1 p-4" style={{backgroundColor: '#111111', minHeight: '400px'}}>
          <VideoPlayer 
            onTargetClick={handleTargetClick} 
            onVideoTimeUpdate={setVideoTime}
          />
        </div>

        {/* Right Sidebar - Mini Map */}
        <div className="w-96 border-l border-border bg-card/30 p-4" style={{backgroundColor: '#1a1a1a', borderColor: '#333333', minHeight: '400px'}}>
          <MiniMap asset={mockAsset} videoTime={videoTime} />
        </div>
      </div>

      {/* Target Details Modal */}
      <TargetDetailsModal
        target={selectedTarget}
        isOpen={isModalOpen}
        onClose={handleCloseModal}
      />
    </div>
  );
};

export default Index;