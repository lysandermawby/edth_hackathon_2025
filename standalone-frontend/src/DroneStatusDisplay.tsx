import React from 'react';
import { HiCube, HiLocationMarker } from 'react-icons/hi';
import type { DroneMetadata } from './types';

interface DroneStatusDisplayProps {
  metadata?: DroneMetadata;
  className?: string;
}

const DroneStatusDisplay: React.FC<DroneStatusDisplayProps> = ({
  metadata,
  className = '',
}) => {
  const formatAngle = (angle?: number) => (angle || 0).toFixed(1);
  const formatAltitude = (alt?: number) => (alt || 0).toFixed(1);

  // Create visual attitude indicator
  const AttitudeIndicator: React.FC<{ roll?: number; pitch?: number; yaw?: number }> = ({ roll = 0, pitch = 0, yaw = 0 }) => {
    const rollRad = (roll * Math.PI) / 180;
    const pitchOffset = (pitch / 90) * 20; // Scale pitch to pixels
    
    return (
      <div className="relative w-24 h-24 bg-cyber-surface border border-cyber-border mx-auto">
        <div className="absolute inset-2 border border-neon-cyan opacity-50"></div>
        
        {/* Artificial horizon */}
        <div 
          className="absolute inset-0 bg-gradient-to-b from-neon-cyan/20 to-cyber-surface"
          style={{
            transform: `rotate(${roll}deg) translateY(${pitchOffset}px)`,
            transformOrigin: 'center',
          }}
        />
        
        {/* Center cross */}
        <div className="absolute top-1/2 left-1/2 w-4 h-0.5 bg-neon-yellow transform -translate-x-1/2 -translate-y-1/2"></div>
        <div className="absolute top-1/2 left-1/2 w-0.5 h-4 bg-neon-yellow transform -translate-x-1/2 -translate-y-1/2"></div>
        
        {/* Yaw indicator */}
        <div className="absolute -top-6 left-1/2 transform -translate-x-1/2">
          <div 
            className="w-6 h-6 border-2 border-neon-pink flex items-center justify-center"
            style={{ transform: `rotate(${yaw}deg)` }}
          >
            <div className="w-1 h-3 bg-neon-pink"></div>
          </div>
        </div>
      </div>
    );
  };

  // Gimbal position indicator
  const GimbalIndicator: React.FC<{ elevation?: number; azimuth?: number }> = ({ elevation = 0, azimuth = 0 }) => {
    return (
      <div className="relative w-20 h-20 bg-cyber-surface border border-cyber-border mx-auto">
        {/* Gimbal base */}
        <div className="absolute inset-2 border border-neon-orange rounded-full opacity-50"></div>
        
        {/* Camera direction */}
        <div 
          className="absolute top-1/2 left-1/2 w-8 h-0.5 bg-neon-yellow origin-left"
          style={{
            transform: `translate(-50%, -50%) rotate(${azimuth}deg)`,
          }}
        />
        
        {/* Elevation indicator */}
        <div className="absolute top-1 left-1/2 transform -translate-x-1/2">
          <div 
            className="w-2 h-2 bg-neon-orange"
            style={{ transform: `translateY(${(elevation / 90) * 8}px)` }}
          />
        </div>
      </div>
    );
  };

  if (!metadata) {
    return (
      <div className={`${className} cyber-card`}>
        <div className="cyber-card-header">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-cyber-surface border-2 border-cyber-border flex items-center justify-center">
              <HiCube className="text-cyber-muted text-sm" />
            </div>
            <div>
              <h3 className="font-semibold text-cyber-muted">
                &gt;&gt; TELEMETRY.EXE
              </h3>
              <p className="text-xs text-cyber-muted font-mono">
                [OFFLINE]
              </p>
            </div>
          </div>
        </div>
        <div className="p-4">
          <div className="text-center text-cyber-muted font-mono text-sm">
            NO_TELEMETRY_DATA<br />
            WAITING_FOR_SIGNAL...
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`${className} cyber-card`}>
      <div className="cyber-card-header">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-cyber-surface border-2 border-neon-cyan flex items-center justify-center shadow-cyber relative">
            <HiCube className="text-neon-cyan text-sm text-glow" />
            <div className="absolute inset-0 border border-neon-cyan animate-pulse-neon opacity-30"></div>
          </div>
          <div>
            <h3 className="font-semibold text-neon-cyan text-glow">
              &gt;&gt; TELEMETRY.EXE
            </h3>
            <p className="text-xs text-cyber-muted font-mono">
              [ATTITUDE_TRACKER_v2.0]
            </p>
          </div>
        </div>
      </div>
      <div className="p-4">
        <div className="space-y-4">
          {/* Attitude Section */}
          <div>
            <h4 className="font-medium text-neon-cyan flex items-center gap-2 font-mono text-sm mb-2">
              <HiLocationMarker className="text-neon-cyan w-3 h-3" />
              ATTITUDE
            </h4>
            <AttitudeIndicator 
              roll={metadata.roll} 
              pitch={metadata.pitch} 
              yaw={metadata.yaw} 
            />
            <div className="mt-2 space-y-1 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-cyber-muted">YAW:</span>
                <span className="text-neon-cyan">{formatAngle(metadata.yaw)}°</span>
              </div>
              <div className="flex justify-between">
                <span className="text-cyber-muted">PITCH:</span>
                <span className="text-neon-cyan">{formatAngle(metadata.pitch)}°</span>
              </div>
              <div className="flex justify-between">
                <span className="text-cyber-muted">ROLL:</span>
                <span className="text-neon-cyan">{formatAngle(metadata.roll)}°</span>
              </div>
            </div>
          </div>

          {/* Gimbal Section */}
          <div className="border-t border-cyber-border pt-3">
            <h4 className="font-medium text-neon-pink flex items-center gap-2 font-mono text-sm mb-2">
              <div className="w-3 h-3 border border-neon-pink"></div>
              GIMBAL
            </h4>
            <GimbalIndicator 
              elevation={metadata.gimbal_elevation} 
              azimuth={metadata.gimbal_azimuth} 
            />
            <div className="mt-2 space-y-1 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-cyber-muted">ELEV:</span>
                <span className="text-neon-pink">{formatAngle(metadata.gimbal_elevation)}°</span>
              </div>
              <div className="flex justify-between">
                <span className="text-cyber-muted">AZIM:</span>
                <span className="text-neon-pink">{formatAngle(metadata.gimbal_azimuth)}°</span>
              </div>
            </div>
          </div>

          {/* Additional Data */}
          <div className="border-t border-cyber-border pt-3">
            <div className="space-y-1 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-cyber-muted">ALT:</span>
                <span className="text-neon-yellow">{formatAltitude(metadata.altitude)}m</span>
              </div>
              <div className="flex justify-between">
                <span className="text-cyber-muted">HFOV:</span>
                <span className="text-neon-green">{formatAngle(metadata.hfov)}°</span>
              </div>
              <div className="flex justify-between">
                <span className="text-cyber-muted">STATUS:</span>
                <span className="text-neon-green">ONLINE</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DroneStatusDisplay;
