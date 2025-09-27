import React from 'react';
import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { Card } from '@/components/ui/card';
import { Target } from '@/types/target';

interface TargetDetailsModalProps {
  target: Target | null;
  isOpen: boolean;
  onClose: () => void;
}

const TargetDetailsModal: React.FC<TargetDetailsModalProps> = ({ target, isOpen, onClose }) => {
  if (!target) return null;

  const getStatusColor = (status: Target['status']) => {
    switch (status) {
      case 'active': return 'status-active';
      case 'tracking': return 'status-active';
      case 'identified': return 'status-warning';
      case 'lost': return 'status-critical';
      default: return 'text-muted-foreground';
    }
  };

  return (
    <Sheet open={isOpen} onOpenChange={onClose}>
      <SheetContent side="right" className="w-[800px] max-w-[90vw] bg-card border-border overflow-auto">
        <SheetHeader>
          <SheetTitle className="hud-text text-xl flex items-center gap-2">
            TARGET ANALYSIS: {target.id}
            <span className={`text-sm ${getStatusColor(target.status)}`}>
              ● {target.status.toUpperCase()}
            </span>
          </SheetTitle>
        </SheetHeader>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          {/* Left Column */}
          <div className="space-y-4">
            <Card className="p-4 bg-secondary/50">
              <h3 className="hud-text font-semibold mb-3 text-accent">TRACKING ALGORITHM</h3>
              <div className="space-y-2 text-sm hud-text">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Re-ID Score:</span>
                  <span className="status-active">{target.reidentificationScore}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Occlusion Count:</span>
                  <span className="text-foreground">{target.occlusionCount}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Confidence:</span>
                  <span className="text-foreground">{target.confidence}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Track Duration:</span>
                  <span className="text-foreground">{target.trackingDuration}s</span>
                </div>
              </div>
            </Card>

            <Card className="p-4 bg-secondary/50">
              <h3 className="hud-text font-semibold mb-3 text-accent">KINEMATICS</h3>
              <div className="space-y-2 text-sm hud-text">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Velocity:</span>
                  <span className="text-foreground">{target.velocity.speed.toFixed(2)} m/s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Direction:</span>
                  <span className="text-foreground">{target.velocity.direction}°</span>
                </div>
                {target.velocity.vertical && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Vertical:</span>
                    <span className="text-foreground">{target.velocity.vertical.toFixed(2)} m/s</span>
                  </div>
                )}
              </div>
            </Card>

            <Card className="p-4 bg-secondary/50">
              <h3 className="hud-text font-semibold mb-3 text-accent">CLASSIFICATION</h3>
              <div className="space-y-2 text-sm hud-text">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Type:</span>
                  <span className="text-foreground">{target.classification}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Last Seen:</span>
                  <span className="text-foreground">{target.lastSeen.toLocaleTimeString()}</span>
                </div>
              </div>
            </Card>
          </div>

          {/* Right Column */}
          <div className="space-y-4">
            <Card className="p-4 bg-secondary/50">
              <h3 className="hud-text font-semibold mb-3 text-accent">DIMENSIONS</h3>
              <div className="space-y-2 text-sm hud-text">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Width:</span>
                  <span className="text-foreground">{target.dimensions.width.toFixed(2)} m</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Height:</span>
                  <span className="text-foreground">{target.dimensions.height.toFixed(2)} m</span>
                </div>
                {target.dimensions.length && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Length:</span>
                    <span className="text-foreground">{target.dimensions.length.toFixed(2)} m</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Scale Factor:</span>
                  <span className="text-foreground">{target.scale.toFixed(3)}x</span>
                </div>
              </div>
            </Card>

            <Card className="p-4 bg-secondary/50">
              <h3 className="hud-text font-semibold mb-3 text-accent">POSITION</h3>
              <div className="space-y-2 text-sm hud-text">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">X:</span>
                  <span className="text-foreground">{target.position.x.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Y:</span>
                  <span className="text-foreground">{target.position.y.toFixed(3)}</span>
                </div>
                {target.position.z && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Z:</span>
                    <span className="text-foreground">{target.position.z.toFixed(3)}</span>
                  </div>
                )}
              </div>
            </Card>

            <Card className="p-4 bg-secondary/50">
              <h3 className="hud-text font-semibold mb-3 text-accent">ALGORITHM STATUS</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground hud-text">Robust Re-ID:</span>
                  <div className="w-20 h-2 bg-muted rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-tactical-active transition-all duration-300"
                      style={{ width: `${target.reidentificationScore}%` }}
                    />
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground hud-text">Occlusion Handle:</span>
                  <div className="w-20 h-2 bg-muted rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-tactical-warning transition-all duration-300"
                      style={{ width: `${Math.max(0, 100 - target.occlusionCount * 10)}%` }}
                    />
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground hud-text">Track Quality:</span>
                  <div className="w-20 h-2 bg-muted rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-tactical-active transition-all duration-300"
                      style={{ width: `${target.confidence}%` }}
                    />
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
};

export default TargetDetailsModal;