import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
} from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Polyline,
  Polygon,
  useMap,
} from "react-leaflet";
import L, { LatLngExpression } from "leaflet";
import "leaflet/dist/leaflet.css";

// Fix for default markers in React Leaflet
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
  iconUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
  shadowUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
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
  footprint?: number[][]; // [lon, lat] polygon coordinates
}

interface VelocityState {
  vx: number;
  vy: number;
  speed: number;
  heading: number | null;
}

const EARTH_RADIUS = 6371000; // meters
const VELOCITY_SMOOTHING = 0.6;
const VELOCITY_ARROW_MIN_METERS = 15;
const VELOCITY_ARROW_MAX_METERS = 120;

const toRadians = (degrees: number) => (degrees * Math.PI) / 180;
const toDegrees = (radians: number) => (radians * 180) / Math.PI;

const haversineDistance = (
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
) => {
  const dLat = toRadians(lat2 - lat1);
  const dLon = toRadians(lon2 - lon1);
  const rLat1 = toRadians(lat1);
  const rLat2 = toRadians(lat2);

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.sin(dLon / 2) * Math.sin(dLon / 2) * Math.cos(rLat1) * Math.cos(rLat2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return EARTH_RADIUS * c;
};

const calculateBearing = (
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
) => {
  const rLat1 = toRadians(lat1);
  const rLat2 = toRadians(lat2);
  const dLon = toRadians(lon2 - lon1);

  const y = Math.sin(dLon) * Math.cos(rLat2);
  const x =
    Math.cos(rLat1) * Math.sin(rLat2) -
    Math.sin(rLat1) * Math.cos(rLat2) * Math.cos(dLon);
  const bearing = Math.atan2(y, x);
  return (toDegrees(bearing) + 360) % 360;
};

const projectPoint = (
  lat: number,
  lon: number,
  bearingDeg: number,
  distanceMeters: number
): [number, number] => {
  const angularDistance = distanceMeters / EARTH_RADIUS;
  const bearingRad = toRadians(bearingDeg);
  const latRad = toRadians(lat);
  const lonRad = toRadians(lon);

  const newLat = Math.asin(
    Math.sin(latRad) * Math.cos(angularDistance) +
      Math.cos(latRad) * Math.sin(angularDistance) * Math.cos(bearingRad)
  );
  const newLon =
    lonRad +
    Math.atan2(
      Math.sin(bearingRad) * Math.sin(angularDistance) * Math.cos(latRad),
      Math.cos(angularDistance) - Math.sin(latRad) * Math.sin(newLat)
    );

  return [toDegrees(newLat), ((toDegrees(newLon) + 540) % 360) - 180];
};

const clamp = (value: number, min: number, max: number) =>
  Math.min(Math.max(value, min), max);

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
    className: "drone-marker",
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

const createVelocityIcon = (heading: number) => {
  return L.divIcon({
    className: "velocity-marker",
    html: `
      <div style="
        width: 16px;
        height: 16px;
        transform: rotate(${heading}deg);
        display: flex;
        align-items: center;
        justify-content: center;
      ">
        <div style="
          width: 0;
          height: 0;
          border-left: 6px solid transparent;
          border-right: 6px solid transparent;
          border-bottom: 12px solid #ff2e63;
        "></div>
      </div>
    `,
    iconSize: [16, 16],
    iconAnchor: [8, 8],
  });
};

// FOV cone component
const FovCone: React.FC<{
  position: [number, number];
  yaw: number;
  gimbal_azimuth?: number;
  hfov: number;
  range: number;
}> = ({ position, yaw, gimbal_azimuth = 0, hfov, range }) => {
  const map = useMap();

  useEffect(() => {
    const coneAngle = (hfov / 2) * (Math.PI / 180);

    // Calculate actual camera pointing direction: drone yaw + gimbal azimuth
    // Drone yaw is the direction the drone is facing
    // Gimbal azimuth is the relative rotation of the camera from the drone's forward direction
    const actualCameraHeading = yaw + gimbal_azimuth;
    const cameraHeadingRad = actualCameraHeading * (Math.PI / 180);

    // Calculate FOV cone vertices based on actual camera direction
    const leftAngle = cameraHeadingRad - coneAngle;
    const rightAngle = cameraHeadingRad + coneAngle;

    // Convert range to lat/lng offset (rough conversion)
    const latOffset = (range * Math.cos(leftAngle)) / 111000; // meters to degrees
    const lonOffset =
      (range * Math.sin(leftAngle)) /
      (111000 * Math.cos((position[0] * Math.PI) / 180));

    const leftPoint: LatLngExpression = [
      position[0] + latOffset,
      position[1] + lonOffset,
    ];

    const latOffset2 = (range * Math.cos(rightAngle)) / 111000;
    const lonOffset2 =
      (range * Math.sin(rightAngle)) /
      (111000 * Math.cos((position[0] * Math.PI) / 180));

    const rightPoint: LatLngExpression = [
      position[0] + latOffset2,
      position[1] + lonOffset2,
    ];

    // Create FOV polygon
    const fovPolygon = L.polygon([position, leftPoint, rightPoint], {
      color: "#ffff00",
      fillColor: "#ffff00",
      fillOpacity: 0.2,
      weight: 2,
      dashArray: "5, 5",
    });

    fovPolygon.addTo(map);

    return () => {
      map.removeLayer(fovPolygon);
    };
  }, [map, position, yaw, gimbal_azimuth, hfov, range]);

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
  className = "",
  enhancedTelemetry,
  showFootprints = false,
}) => {
  const [mapZoom, setMapZoom] = useState(14);
  const [showFlightPath, setShowFlightPath] = useState(true);
  const [showFOV, setShowFOV] = useState(true);
  const [showCameraFootprints, setShowCameraFootprints] =
    useState(showFootprints);
  const mapRef = useRef<L.Map | null>(null);
  const prevVelocityRef = useRef<VelocityState>({
    vx: 0,
    vy: 0,
    speed: 0,
    heading: null,
  });
  const [currentVelocity, setCurrentVelocity] = useState<VelocityState | null>(
    null
  );

  // Use enhanced telemetry if available, otherwise fall back to regular metadata
  const activeData = enhancedTelemetry || metadata;

  // Filter valid GPS coordinates
  const validMetadata = useMemo(
    () =>
      activeData.filter(
        (entry) =>
          entry.latitude > -90 &&
          entry.latitude < 90 &&
          entry.longitude > -180 &&
          entry.longitude < 180
      ),
    [activeData]
  );

  // Get current metadata entry
  const currentMetadata =
    validMetadata[Math.min(currentFrame, validMetadata.length - 1)];

  useEffect(() => {
    prevVelocityRef.current = { vx: 0, vy: 0, speed: 0, heading: null };
    setCurrentVelocity(null);
  }, [enhancedTelemetry, metadata]);

  useEffect(() => {
    if (!currentMetadata || currentFrame <= 0 || validMetadata.length < 2) {
      return;
    }

    const prevIndex = Math.max(
      0,
      Math.min(currentFrame - 1, validMetadata.length - 1)
    );
    const prevMetadata = validMetadata[prevIndex];

    if (!prevMetadata || prevMetadata === currentMetadata) {
      return;
    }

    const timeDeltaRaw = currentMetadata.timestamp - prevMetadata.timestamp;
    if (!Number.isFinite(timeDeltaRaw) || timeDeltaRaw <= 0) {
      return;
    }

    const timeDeltaSeconds =
      timeDeltaRaw > 1e3 ? timeDeltaRaw / 1_000_000 : timeDeltaRaw;
    if (timeDeltaSeconds <= 0) {
      return;
    }

    const distanceMeters = haversineDistance(
      prevMetadata.latitude,
      prevMetadata.longitude,
      currentMetadata.latitude,
      currentMetadata.longitude
    );

    if (!Number.isFinite(distanceMeters) || distanceMeters < 0.01) {
      return;
    }

    const instantaneousSpeed = distanceMeters / timeDeltaSeconds;
    const bearing = calculateBearing(
      prevMetadata.latitude,
      prevMetadata.longitude,
      currentMetadata.latitude,
      currentMetadata.longitude
    );

    const bearingRad = toRadians(bearing);
    const vx = instantaneousSpeed * Math.sin(bearingRad);
    const vy = instantaneousSpeed * Math.cos(bearingRad);

    const prev = prevVelocityRef.current;
    const smoothedVx =
      prev.heading === null
        ? vx
        : VELOCITY_SMOOTHING * vx + (1 - VELOCITY_SMOOTHING) * prev.vx;
    const smoothedVy =
      prev.heading === null
        ? vy
        : VELOCITY_SMOOTHING * vy + (1 - VELOCITY_SMOOTHING) * prev.vy;

    const smoothedSpeed = Math.hypot(smoothedVx, smoothedVy);
    const smoothedHeading =
      smoothedSpeed > 0.05
        ? (Math.atan2(smoothedVx, smoothedVy) * 180) / Math.PI
        : null;

    const normalizedHeading =
      smoothedHeading !== null ? (smoothedHeading + 360) % 360 : null;

    const nextVelocity: VelocityState = {
      vx: smoothedVx,
      vy: smoothedVy,
      speed: smoothedSpeed,
      heading: normalizedHeading,
    };

    prevVelocityRef.current = nextVelocity;
    setCurrentVelocity(nextVelocity);
  }, [currentFrame, currentMetadata, validMetadata]);

  // Calculate map center and bounds
  const getMapCenter = useCallback((): [number, number] => {
    if (!currentMetadata) {
      return [48.0, 11.0]; // Default center
    }
    return [currentMetadata.latitude, currentMetadata.longitude];
  }, [currentMetadata]);

  // Create flight path coordinates
  const flightPath: LatLngExpression[] = validMetadata.map((entry) => [
    entry.latitude,
    entry.longitude,
  ]);

  // Create completed path (up to current frame)
  const completedPath: LatLngExpression[] = validMetadata
    .slice(0, Math.min(currentFrame + 1, validMetadata.length))
    .map((entry) => [entry.latitude, entry.longitude]);

  if (!currentMetadata) {
    return (
      <div
        className={`${className} flex items-center justify-center bg-gray-100 rounded-lg`}
      >
        <div className="text-gray-500 text-center">
          <div className="text-4xl mb-2">🗺️</div>
          <p>No GPS data available</p>
        </div>
      </div>
    );
  }

  const velocityArrow =
    currentVelocity &&
    currentVelocity.heading !== null &&
    currentVelocity.speed > 0.05
      ? projectPoint(
          currentMetadata.latitude,
          currentMetadata.longitude,
          currentVelocity.heading,
          clamp(
            currentVelocity.speed * 2.5,
            VELOCITY_ARROW_MIN_METERS,
            VELOCITY_ARROW_MAX_METERS
          )
        )
      : null;

  return (
    <div className={`${className} relative`}>
      {/* Map Controls */}
      <div className="absolute bottom-4 left-4 z-[1000] bg-tactical-surface text-tactical-text rounded-lg shadow-lg p-3 space-y-2">
        <div className="flex items-center space-x-2">
          <label className="text-sm font-medium">Zoom:</label>
          <select
            value={mapZoom}
            onChange={(e) => setMapZoom(Number(e.target.value))}
            className="text-sm border rounded px-2 py-1 tactical-input"
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
          <label htmlFor="flight-path" className="text-sm">
            Flight Path
          </label>
        </div>

        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            id="fov"
            checked={showFOV}
            onChange={(e) => setShowFOV(e.target.checked)}
            className="rounded"
          />
          <label htmlFor="fov" className="text-sm">
            Field of View
          </label>
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
            <label htmlFor="camera-footprints" className="text-sm">
              📹 Camera Footprints
            </label>
          </div>
        )}
      </div>

      {/* Position Info */}
      <div className="absolute top-4 right-4 z-[1000] bg-tactical-surface text-tactical-text rounded-lg shadow-lg p-3">
        <div className="text-sm space-y-1">
          <div>
            <strong>Lat:</strong> {currentMetadata.latitude.toFixed(6)}
          </div>
          <div>
            <strong>Lon:</strong> {currentMetadata.longitude.toFixed(6)}
          </div>
          <div>
            <strong>Alt:</strong> {currentMetadata.altitude.toFixed(1)}m
          </div>
          <div>
            <strong>Yaw:</strong> {currentMetadata.yaw.toFixed(1)}°
          </div>
          <div>
            <strong>Frame:</strong> {currentFrame}/{metadata.length}
          </div>
          {currentVelocity && currentVelocity.heading !== null && (
            <div>
              <strong>Speed:</strong> {currentVelocity.speed.toFixed(1)} m/s
              <span className="ml-2">
                <strong>Heading:</strong> {currentVelocity.heading.toFixed(0)}°
              </span>
            </div>
          )}
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
            gimbal_azimuth={currentMetadata.gimbal_azimuth}
            hfov={currentMetadata.hfov}
            range={900} // 900 meters range (3x longer)
          />
        )}

        {/* Velocity Arrow */}
        {velocityArrow &&
          currentVelocity &&
          currentVelocity.heading !== null && (
            <>
              <Polyline
                positions={[
                  [currentMetadata.latitude, currentMetadata.longitude],
                  [velocityArrow[0], velocityArrow[1]],
                ]}
                color="#ff2e63"
                weight={3}
                opacity={0.9}
              />
              <Marker
                position={[velocityArrow[0], velocityArrow[1]]}
                icon={createVelocityIcon(currentVelocity.heading)}
              />
            </>
          )}

        {/* Camera Footprints (Enhanced Telemetry) */}
        {showCameraFootprints &&
          enhancedTelemetry &&
          enhancedTelemetry[currentFrame]?.footprint && (
            <Polygon
              positions={enhancedTelemetry[currentFrame].footprint!.map(
                (coord) => [coord[1], coord[0]]
              )} // Convert [lon, lat] to [lat, lon]
              pathOptions={{
                color: "#ff6b35",
                fillColor: "#ff6b35",
                fillOpacity: 0.2,
                weight: 2,
                opacity: 0.8,
              }}
            />
          )}

        {/* Center Point (What camera is looking at) */}
        {enhancedTelemetry &&
          enhancedTelemetry[currentFrame] &&
          "center_latitude" in enhancedTelemetry[currentFrame] && (
            <Marker
              position={[
                (enhancedTelemetry[currentFrame] as EnhancedTelemetryPoint)
                  .center_latitude,
                (enhancedTelemetry[currentFrame] as EnhancedTelemetryPoint)
                  .center_longitude,
              ]}
              icon={L.divIcon({
                className: "camera-target",
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
                iconAnchor: [6, 6],
              })}
            />
          )}
      </MapContainer>
    </div>
  );
};

export default DroneMapViewer;
