import React, { useEffect, useRef, useState, useCallback } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, Polygon, useMap } from 'react-leaflet';
import L, { LatLngExpression } from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for default markers in React Leaflet
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface DroneMetadata {
  timestamp: number;
  latitude: number;
  longitude: number;
  altitude: number;
  roll: number;
  pitch: number;
  yaw: number;
  gimbal_elevation: number;
  gimbal_azimuth: number;
  vfov: number;
  hfov: number;
}

interface EnhancedTelemetryPoint extends DroneMetadata {
  center_latitude: number;
  center_longitude: number;
  slant_range: number;
  footprint?: number[][];  // [lon, lat] polygon coordinates
}

interface DroneMapViewerProps {
  metadata: DroneMetadata[];
  currentFrame: number;
  className?: string;
  enhancedTelemetry?: EnhancedTelemetryPoint[];
  showFootprints?: boolean;
}

// Custom drone icon
const createDroneIcon = (yaw: number) => {
  return L.divIcon({
    className: 'drone-marker',
    html: `
      <div style="
        transform: rotate(${yaw}deg);
        width: 24px;
        height: 24px;
        background: #ff0000;
        border-radius: 50%;
        border: 3px solid white;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        display: flex;
        align-items: center;
        justify-content: center;
      ">
        <div style="
          width: 0;
          height: 0;
          border-left: 4px solid transparent;
          border-right: 4px solid transparent;
          border-bottom: 8px solid white;
          transform: translateY(-2px);
        "></div>
      </div>
    `,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  });
};

// FOV cone component
const FovCone: React.FC<{
  position: [number, number];
  yaw: number;
  hfov: number;
  range: number;
}> = ({ position, yaw, hfov, range }) => {
  const map = useMap();

  useEffect(() => {
    const coneAngle = (hfov / 2) * (Math.PI / 180);
    const yawRad = yaw * (Math.PI / 180);

    // Calculate FOV cone vertices
    const leftAngle = yawRad - coneAngle;
    const rightAngle = yawRad + coneAngle;

    // Convert range to lat/lng offset (rough conversion)
    const latOffset = range * Math.cos(leftAngle) / 111000; // meters to degrees
    const lonOffset = range * Math.sin(leftAngle) / (111000 * Math.cos(position[0] * Math.PI / 180));

    const leftPoint: LatLngExpression = [
      position[0] + latOffset,
      position[1] + lonOffset
    ];

    const latOffset2 = range * Math.cos(rightAngle) / 111000;
    const lonOffset2 = range * Math.sin(rightAngle) / (111000 * Math.cos(position[0] * Math.PI / 180));

    const rightPoint: LatLngExpression = [
      position[0] + latOffset2,
      position[1] + lonOffset2
    ];

    // Create FOV polygon
    const fovPolygon = L.polygon([position, leftPoint, rightPoint], {
      color: '#ffff00',
      fillColor: '#ffff00',
      fillOpacity: 0.2,
      weight: 2,
      dashArray: '5, 5'
    });

    fovPolygon.addTo(map);

    return () => {
      map.removeLayer(fovPolygon);
    };
  }, [map, position, yaw, hfov, range]);

  return null;
};

// Map controller to handle view updates
const MapController: React.FC<{
  center: [number, number];
  zoom: number;
}> = ({ center, zoom }) => {
  const map = useMap();

  useEffect(() => {
    map.setView(center, zoom);
  }, [map, center, zoom]);

  return null;
};

const DroneMapViewer: React.FC<DroneMapViewerProps> = ({
  metadata,
  currentFrame,
  className = '',
  enhancedTelemetry,
  showFootprints = false
}) => {
  const [mapZoom, setMapZoom] = useState(16);
  const [showFlightPath, setShowFlightPath] = useState(true);
  const [showFOV, setShowFOV] = useState(true);
  const [showCameraFootprints, setShowCameraFootprints] = useState(showFootprints);
  const mapRef = useRef<L.Map | null>(null);

  // Use enhanced telemetry if available, otherwise fall back to regular metadata
  const activeData = enhancedTelemetry || metadata;

  // Filter valid GPS coordinates
  const validMetadata = activeData.filter(
    entry => entry.latitude > -90 && entry.latitude < 90 &&
             entry.longitude > -180 && entry.longitude < 180
  );

  // Get current metadata entry
  const currentMetadata = validMetadata[Math.min(currentFrame, validMetadata.length - 1)];

  // Calculate map center and bounds
  const getMapCenter = useCallback((): [number, number] => {
    if (!currentMetadata) {
      return [48.0, 11.0]; // Default center
    }
    return [currentMetadata.latitude, currentMetadata.longitude];
  }, [currentMetadata]);

  // Create flight path coordinates
  const flightPath: LatLngExpression[] = validMetadata.map(entry => [
    entry.latitude,
    entry.longitude
  ]);

  // Create completed path (up to current frame)
  const completedPath: LatLngExpression[] = validMetadata
    .slice(0, Math.min(currentFrame + 1, validMetadata.length))
    .map(entry => [entry.latitude, entry.longitude]);

  if (!currentMetadata) {
    return (
      <div className={`${className} flex items-center justify-center bg-gray-100 rounded-lg`}>
        <div className="text-gray-500 text-center">
          <div className="text-4xl mb-2">🗺️</div>
          <p>No GPS data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`${className} relative`}>
      {/* Map Controls */}
      <div className="absolute top-4 left-4 z-[1000] bg-white rounded-lg shadow-lg p-3 space-y-2">
        <div className="flex items-center space-x-2">
          <label className="text-sm font-medium">Zoom:</label>
          <select
            value={mapZoom}
            onChange={(e) => setMapZoom(Number(e.target.value))}
            className="text-sm border rounded px-2 py-1"
          >
            <option value={13}>13</option>
            <option value={14}>14</option>
            <option value={15}>15</option>
            <option value={16}>16</option>
            <option value={17}>17</option>
            <option value={18}>18</option>
          </select>
        </div>

        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            id="flight-path"
            checked={showFlightPath}
            onChange={(e) => setShowFlightPath(e.target.checked)}
            className="rounded"
          />
          <label htmlFor="flight-path" className="text-sm">Flight Path</label>
        </div>

        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            id="fov"
            checked={showFOV}
            onChange={(e) => setShowFOV(e.target.checked)}
            className="rounded"
          />
          <label htmlFor="fov" className="text-sm">Field of View</label>
        </div>

        {enhancedTelemetry && (
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="camera-footprints"
              checked={showCameraFootprints}
              onChange={(e) => setShowCameraFootprints(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="camera-footprints" className="text-sm">📹 Camera Footprints</label>
          </div>
        )}
      </div>

      {/* Position Info */}
      <div className="absolute top-4 right-4 z-[1000] bg-white rounded-lg shadow-lg p-3">
        <div className="text-sm space-y-1">
          <div><strong>Lat:</strong> {currentMetadata.latitude.toFixed(6)}</div>
          <div><strong>Lon:</strong> {currentMetadata.longitude.toFixed(6)}</div>
          <div><strong>Alt:</strong> {currentMetadata.altitude.toFixed(1)}m</div>
          <div><strong>Yaw:</strong> {currentMetadata.yaw.toFixed(1)}°</div>
          <div><strong>Frame:</strong> {currentFrame}/{metadata.length}</div>
        </div>
      </div>

      {/* Map */}
      <MapContainer
        center={getMapCenter()}
        zoom={mapZoom}
        className="w-full h-full rounded-lg"
        ref={mapRef}
      >
        <MapController center={getMapCenter()} zoom={mapZoom} />

        {/* Satellite Tile Layer */}
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          attribution='&copy; <a href="https://www.esri.com/">Esri</a>'
          maxZoom={18}
        />

        {/* Flight Path */}
        {showFlightPath && flightPath.length > 1 && (
          <>
            {/* Full path in cyan */}
            <Polyline
              positions={flightPath}
              color="#00ffff"
              weight={2}
              opacity={0.7}
            />

            {/* Completed path in lime */}
            {completedPath.length > 1 && (
              <Polyline
                positions={completedPath}
                color="#00ff00"
                weight={3}
                opacity={0.9}
              />
            )}
          </>
        )}

        {/* Current Drone Position */}
        <Marker
          position={[currentMetadata.latitude, currentMetadata.longitude]}
          icon={createDroneIcon(currentMetadata.yaw)}
        />

        {/* Field of View Cone */}
        {showFOV && (
          <FovCone
            position={[currentMetadata.latitude, currentMetadata.longitude]}
            yaw={currentMetadata.yaw}
            hfov={currentMetadata.hfov}
            range={300} // 300 meters range
          />
        )}

        {/* Camera Footprints (Enhanced Telemetry) */}
        {showCameraFootprints && enhancedTelemetry && enhancedTelemetry[currentFrame]?.footprint && (
          <Polygon
            positions={enhancedTelemetry[currentFrame].footprint!.map(coord => [coord[1], coord[0]])} // Convert [lon, lat] to [lat, lon]
            pathOptions={{
              color: '#ff6b35',
              fillColor: '#ff6b35',
              fillOpacity: 0.2,
              weight: 2,
              opacity: 0.8
            }}
          />
        )}

        {/* Center Point (What camera is looking at) */}
        {enhancedTelemetry && enhancedTelemetry[currentFrame] && 'center_latitude' in enhancedTelemetry[currentFrame] && (
          <Marker
            position={[
              (enhancedTelemetry[currentFrame] as EnhancedTelemetryPoint).center_latitude,
              (enhancedTelemetry[currentFrame] as EnhancedTelemetryPoint).center_longitude
            ]}
            icon={L.divIcon({
              className: 'camera-target',
              html: `
                <div style="
                  width: 12px;
                  height: 12px;
                  background: #ff6b35;
                  border: 2px solid white;
                  border-radius: 50%;
                  box-shadow: 0 0 6px rgba(0,0,0,0.5);
                "></div>
              `,
              iconSize: [12, 12],
              iconAnchor: [6, 6]
            })}
          />
        )}
      </MapContainer>
    </div>
  );
};

export default DroneMapViewer;