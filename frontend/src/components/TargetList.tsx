import React from 'react';
import { Card } from '@/components/ui/card';
import { Target } from '@/types/target';

interface TargetListProps {
  targets: Target[];
  onTargetSelect: (target: Target) => void;
  selectedTargetId?: string;
}

const TargetList: React.FC<TargetListProps> = ({ targets, onTargetSelect, selectedTargetId }) => {
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
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-border">
        <h2 className="hud-text text-lg font-semibold text-foreground">
          TARGET DATABASE
        </h2>
        <p className="text-sm text-muted-foreground hud-text">
          {targets.length} active contacts
        </p>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-2">
        {targets.map((target) => (
          <Card
            key={target.id}
            className={`p-3 cursor-pointer transition-all duration-200 hover:tactical-glow ${
              selectedTargetId === target.id ? 'border-primary tactical-glow' : 'border-border'
            }`}
            onClick={() => onTargetSelect(target)}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="hud-text font-medium text-foreground">
                {target.id}
              </span>
              <span className={`text-xs hud-text ${getStatusColor(target.status)}`}>
                ● {target.status.toUpperCase()}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs hud-text text-muted-foreground">
              <div>
                <span className="text-foreground">VEL:</span> {target.velocity.speed.toFixed(1)}m/s
              </div>
              <div>
                <span className="text-foreground">DIR:</span> {target.velocity.direction}°
              </div>
              <div>
                <span className="text-foreground">CONF:</span> {target.confidence}%
              </div>
              <div>
                <span className="text-foreground">SCALE:</span> {target.scale.toFixed(2)}x
              </div>
            </div>

            <div className="mt-2 text-xs hud-text text-muted-foreground">
              <div className="flex justify-between">
                <span>CLASS: {target.classification}</span>
                <span>T: {target.trackingDuration}s</span>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default TargetList;