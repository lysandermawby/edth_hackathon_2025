export interface Target {
  id: string;
  status: 'active' | 'lost' | 'tracking' | 'identified';
  position: {
    x: number;
    y: number;
    z?: number;
  };
  velocity: {
    speed: number; // m/s
    direction: number; // degrees
    vertical?: number; // m/s
  };
  dimensions: {
    width: number; // meters
    height: number; // meters
    length?: number; // meters
  };
  scale: number; // relative scale factor
  confidence: number; // 0-100
  lastSeen: Date;
  classification: string;
  trackingDuration: number; // seconds
  occlusionCount: number;
  reidentificationScore: number; // 0-100
}

export interface Asset {
  id: string;
  type: 'drone' | 'aircraft' | 'satellite';
  position: {
    latitude: number;
    longitude: number;
    altitude: number;
  };
  heading: number; // degrees
  status: 'operational' | 'standby' | 'maintenance';
}