import React, { useEffect, useRef, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Asset } from '@/types/target';
import mapboxgl from 'mapbox-gl';
import Papa from 'papaparse';
import 'mapbox-gl/dist/mapbox-gl.css';

interface DroneData {
  timestamp: string;
  vfov: number;
  hfov: number;
  roll: number;
  pitch: number;
  yaw: number;
  latitude: number;
  longitude: number;
  altitude: number;
  gimbal_elevation: number;
  gimbal_azimuth: number;
  center_latitude: number;
  center_longitude: number;
  center_elevation: number;
  slant_range: number;
  relativeTime?: number; // Time in seconds from video start
}

interface MiniMapProps {
  asset: Asset;
  videoTime?: number; // Current video time in seconds
}

const MiniMap: React.FC<MiniMapProps> = ({ asset, videoTime = 0 }) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const marker = useRef<mapboxgl.Marker | null>(null);
  const mapboxToken = 'pk.eyJ1IjoianVhbnF1aTUiLCJhIjoiY21hZmozMTB5MDJoaTJqcjZsZGx4cnk1YiJ9.VmpnhrEA_T10fiiuOG-P3Q';
  const [droneData, setDroneData] = useState<DroneData[]>([]);
  const [currentPosition, setCurrentPosition] = useState<DroneData | null>(null);
  const [trajectoryPath, setTrajectoryPath] = useState<[number, number][]>([]);
  const [videoStartTime, setVideoStartTime] = useState<number>(0);
  
  // Hardcoded CSV path - now in public directory
  const hardcodedCsvPath = '/2025_09_17-15_02_07_MovingObjects_44_segment_1_55.0s-80.0s.csv';

  // Area bounds for the specified coordinates
  const bounds: [number, number, number, number] = [
    11.32179297519343, // west
    48.06686293222302, // south  
    11.35184127505622, // east
    48.0847503226986   // north
  ];

  // Function to load hardcoded CSV data
  const loadHardcodedCsv = async () => {
    try {
      // Load CSV from public directory
      const response = await fetch(hardcodedCsvPath);
      if (response.ok) {
        const csvText = await response.text();
        Papa.parse(csvText, {
          header: true,
          complete: (results) => {
            const data = results.data as DroneData[];
            const validData = data.filter(row => row.latitude && row.longitude);
            
            // Each CSV line represents one video frame
            // Calculate frame rate from the data or use a reasonable default
            const frameRate = 25; // Default 25fps, can be adjusted based on actual video
            const processedData = validData.map((row, index) => ({
              ...row,
              relativeTime: index / frameRate // Time in seconds for each frame
            }));
            
            setDroneData(processedData);
            
            // Create trajectory path
            const path = processedData.map(row => [row.longitude, row.latitude] as [number, number]);
            setTrajectoryPath(path);
            
            // Set video start time (first timestamp converted to seconds)
            if (processedData.length > 0) {
              const firstTimestamp = parseInt(processedData[0].timestamp) / 1000000; // Convert microseconds to seconds
              setVideoStartTime(firstTimestamp);
              setCurrentPosition(processedData[0]);
            }
            
            console.log('Loaded drone data:', processedData.length, 'points');
            console.log('Trajectory path:', path.length, 'coordinates');
          },
          error: (error) => {
            console.error('CSV parsing error:', error);
            // Fallback to mock data
            loadMockData();
          }
        });
      } else {
        console.error('Failed to load CSV file');
        loadMockData();
      }
    } catch (error) {
      console.error('Error loading CSV:', error);
      loadMockData();
    }
  };

  // Fallback mock data
  const loadMockData = () => {
    const mockCsvData: DroneData[] = [
      {
        timestamp: '2024-01-01T10:00:00Z',
        vfov: 45,
        hfov: 60,
        roll: 0,
        pitch: -15,
        yaw: 135,
        latitude: 48.075,
        longitude: 11.336,
        altitude: 1200,
        gimbal_elevation: -20,
        gimbal_azimuth: 0,
        center_latitude: 48.075,
        center_longitude: 11.336,
        center_elevation: 0,
        slant_range: 1500
      },
      {
        timestamp: '2024-01-01T10:01:00Z',
        vfov: 45,
        hfov: 60,
        roll: 2,
        pitch: -12,
        yaw: 140,
        latitude: 48.076,
        longitude: 11.338,
        altitude: 1200,
        gimbal_elevation: -18,
        gimbal_azimuth: 5,
        center_latitude: 48.076,
        center_longitude: 11.338,
        center_elevation: 0,
        slant_range: 1480
      }
    ];
    
    setDroneData(mockCsvData);
    if (mockCsvData.length > 0) {
      setCurrentPosition(mockCsvData[0]);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type === 'text/csv') {
      Papa.parse(file, {
        header: true,
        complete: (results) => {
          const data = results.data as DroneData[];
          setDroneData(data.filter(row => row.latitude && row.longitude));
          if (data.length > 0) {
            setCurrentPosition(data[0]);
          }
        },
        error: (error) => {
          console.error('CSV parsing error:', error);
        }
      });
    }
  };

  // Load hardcoded CSV data on component mount
  useEffect(() => {
    loadHardcodedCsv();
  }, []);

  // Update drone position based on video time
  useEffect(() => {
    if (droneData.length === 0) return;

    // Calculate which frame we should be on based on video time
    const frameRate = 25; // Same as used in data processing
    const currentFrame = Math.floor(videoTime * frameRate);
    
    // Get the data for the current frame (or closest available frame)
    const currentData = droneData[currentFrame] || droneData[droneData.length - 1];

    if (currentData) {
      setCurrentPosition(currentData);
      console.log(`Video time: ${videoTime.toFixed(2)}s, Frame: ${currentFrame}, Position: [${currentData.latitude.toFixed(6)}, ${currentData.longitude.toFixed(6)}]`);
    }
  }, [videoTime, droneData]);

  useEffect(() => {
    if (!mapboxToken || !mapContainer.current) return;

    mapboxgl.accessToken = mapboxToken;
    
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/satellite-v9',
      bounds: bounds,
      fitBoundsOptions: { padding: 20 }
    });

    map.current.addControl(new mapboxgl.NavigationControl(), 'top-left');

    return () => {
      map.current?.remove();
    };
  }, [mapboxToken]);

  useEffect(() => {
    if (!map.current || !currentPosition) return;

    // Remove existing marker
    marker.current?.remove();

    // Add drone marker with enhanced styling
    marker.current = new mapboxgl.Marker({
      color: '#00ff00',
      rotation: currentPosition.yaw,
      scale: 1.2
    })
      .setLngLat([currentPosition.longitude, currentPosition.latitude])
      .addTo(map.current);

    // Add a pulsing circle around the current position
    if (map.current.getSource('current-position')) {
      map.current.removeLayer('current-position-circle');
      map.current.removeSource('current-position');
    }

    map.current.addSource('current-position', {
      type: 'geojson',
      data: {
        type: 'Feature',
        properties: {},
        geometry: {
          type: 'Point',
          coordinates: [currentPosition.longitude, currentPosition.latitude]
        }
      }
    });

    map.current.addLayer({
      id: 'current-position-circle',
      type: 'circle',
      source: 'current-position',
      paint: {
        'circle-radius': 8,
        'circle-color': '#00ff00',
        'circle-opacity': 0.3,
        'circle-stroke-width': 2,
        'circle-stroke-color': '#00ff00',
        'circle-stroke-opacity': 0.8
      }
    });

    // Center map on drone
    map.current.easeTo({
      center: [currentPosition.longitude, currentPosition.latitude],
      duration: 1000
    });
  }, [currentPosition]);

  // Add trajectory path to map
  useEffect(() => {
    if (!map.current || trajectoryPath.length === 0) return;

    // Remove existing trajectory source and layer
    if (map.current.getSource('trajectory')) {
      map.current.removeLayer('trajectory');
      map.current.removeSource('trajectory');
    }

    // Add trajectory as a line
    map.current.addSource('trajectory', {
      type: 'geojson',
      data: {
        type: 'Feature',
        properties: {},
        geometry: {
          type: 'LineString',
          coordinates: trajectoryPath
        }
      }
    });

    map.current.addLayer({
      id: 'trajectory',
      type: 'line',
      source: 'trajectory',
      layout: {
        'line-join': 'round',
        'line-cap': 'round'
      },
      paint: {
        'line-color': '#ff6b6b',
        'line-width': 3,
        'line-opacity': 0.8
      }
    });
  }, [trajectoryPath]);

  return (
    <div className="space-y-4">

      {/* CSV Data */}
      <Card className="p-4 bg-card border-border">
        <h3 className="hud-text font-semibold mb-3 text-accent">DRONE DATA</h3>
        <div className="space-y-2">
          <div className="hud-text text-sm text-muted-foreground">
            Hardcoded CSV: {hardcodedCsvPath}
          </div>
          <div className="hud-text text-sm text-muted-foreground">
            Data points loaded: {droneData.length}
          </div>
          <div className="hud-text text-sm text-muted-foreground">
            Video time: {videoTime.toFixed(2)}s
          </div>
          <div className="hud-text text-sm text-muted-foreground">
            Current frame: {Math.floor(videoTime * 25)}
          </div>
          <Input
            type="file"
            accept=".csv"
            onChange={handleFileUpload}
            className="bg-background border-border text-foreground"
          />
        </div>
      </Card>

      {/* Mini Map */}
      <Card className="p-4 bg-card border-border">
        <h3 className="hud-text font-semibold mb-3 text-accent">TACTICAL MAP</h3>
        <div ref={mapContainer} className="w-full h-96 rounded-sm border border-border" />
      </Card>

      {/* Asset Info */}
      <Card className="p-4 bg-card border-border">
        <h3 className="hud-text font-semibold mb-3 text-accent">ASSET STATUS</h3>
        <div className="space-y-2 text-sm hud-text">
          <div className="flex justify-between">
            <span className="text-muted-foreground">ID:</span>
            <span className="text-foreground">{asset.id}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Type:</span>
            <span className="text-foreground capitalize">{asset.type}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Status:</span>
            <span className="status-active">● {asset.status.toUpperCase()}</span>
          </div>
        </div>
      </Card>

      {/* Position Details */}
      <Card className="p-4 bg-card border-border">
        <h3 className="hud-text font-semibold mb-3 text-accent">POSITION</h3>
        <div className="space-y-2 text-sm hud-text">
          <div className="flex justify-between">
            <span className="text-muted-foreground">LAT:</span>
            <span className="text-foreground">{asset.position.latitude.toFixed(6)}°</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">LON:</span>
            <span className="text-foreground">{asset.position.longitude.toFixed(6)}°</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">ALT:</span>
            <span className="text-foreground">{asset.position.altitude.toFixed(0)}m</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">HDG:</span>
            <span className="text-foreground">{asset.heading}°</span>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default MiniMap;