import React, { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Box, Cylinder, Sphere, Environment } from '@react-three/drei';
import * as THREE from 'three';
import type { DroneMetadata } from './types';

interface DroneVisualization3DProps {
  metadata?: DroneMetadata;
  className?: string;
  currentFrameIndex?: number;
}

interface DroneModelProps {
  metadata: DroneMetadata;
}

const PropellerBlade: React.FC<{ position: [number, number, number]; spinning?: boolean }> = ({ position, spinning = true }) => {
  const propellerRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (propellerRef.current && spinning) {
      propellerRef.current.rotation.y = state.clock.elapsedTime * 50; // Fast spinning
    }
  });

  return (
    <group ref={propellerRef} position={position}>
      {/* Motor */}
      <Cylinder args={[0.12, 0.15, 0.15]} position={[0, 0, 0]}>
        <meshStandardMaterial color="#2a2a2a" metalness={0.8} roughness={0.2} />
      </Cylinder>
      
      {/* Propeller blades */}
      <group>
        <Box args={[1.2, 0.02, 0.08]} position={[0, 0.08, 0]}>
          <meshStandardMaterial 
            color="#1a1a1a" 
            metalness={0.7} 
            roughness={0.3}
            transparent 
            opacity={spinning ? 0.3 : 1}
          />
        </Box>
        <Box args={[0.08, 0.02, 1.2]} position={[0, 0.08, 0]}>
          <meshStandardMaterial 
            color="#1a1a1a" 
            metalness={0.7} 
            roughness={0.3}
            transparent 
            opacity={spinning ? 0.3 : 1}
          />
        </Box>
      </group>
    </group>
  );
};

const DroneArm: React.FC<{ position: [number, number, number]; rotation: [number, number, number] }> = ({ position, rotation }) => {
  return (
    <group position={position} rotation={rotation}>
      <Box args={[0.06, 0.04, 1.0]}>
        <meshStandardMaterial color="#ffffff" metalness={0.6} roughness={0.4} />
      </Box>
    </group>
  );
};

const CameraGimbal: React.FC<{ gimbalElevation?: number; gimbalAzimuth?: number }> = ({ 
  gimbalElevation = 0, 
  gimbalAzimuth = 0 
}) => {
  const gimbalRef = useRef<THREE.Group>(null);
  const cameraRef = useRef<THREE.Group>(null);

  useFrame(() => {
    if (gimbalRef.current) {
      gimbalRef.current.rotation.y = THREE.MathUtils.lerp(
        gimbalRef.current.rotation.y,
        gimbalAzimuth * Math.PI / 180,
        0.1
      );
    }
    if (cameraRef.current) {
      cameraRef.current.rotation.x = THREE.MathUtils.lerp(
        cameraRef.current.rotation.x,
        gimbalElevation * Math.PI / 180,
        0.1
      );
    }
  });

  return (
    <group position={[0, -0.25, 0]}>
      {/* Gimbal outer ring */}
      <group ref={gimbalRef}>
        <Cylinder args={[0.25, 0.25, 0.02]} rotation={[Math.PI / 2, 0, 0]}>
          <meshStandardMaterial color="#ff8000" metalness={0.8} roughness={0.2} />
        </Cylinder>
        
        {/* Gimbal inner ring */}
        <group ref={cameraRef}>
          <Cylinder args={[0.2, 0.2, 0.02]} rotation={[0, 0, Math.PI / 2]}>
            <meshStandardMaterial color="#ff6000" metalness={0.8} roughness={0.2} />
          </Cylinder>
          
          {/* Camera body */}
          <Box args={[0.25, 0.15, 0.3]}>
            <meshStandardMaterial color="#333333" metalness={0.7} roughness={0.3} />
          </Box>
          
          {/* Camera lens */}
          <Cylinder args={[0.08, 0.06, 0.12]} position={[0, 0, 0.21]} rotation={[Math.PI / 2, 0, 0]}>
            <meshStandardMaterial color="#000000" metalness={0.9} roughness={0.1} />
          </Cylinder>
          
          {/* Lens glass */}
          <Cylinder args={[0.07, 0.07, 0.02]} position={[0, 0, 0.27]} rotation={[Math.PI / 2, 0, 0]}>
            <meshStandardMaterial 
              color="#000033" 
              metalness={0.1} 
              roughness={0.0} 
              transparent 
              opacity={0.8}
            />
          </Cylinder>
          
          {/* FOV indicator */}
          <Box args={[0.01, 0.01, 1.5]} position={[0, 0, 0.75]}>
            <meshStandardMaterial 
              color="#ffff00" 
              emissive="#ffff00" 
              emissiveIntensity={0.3}
              transparent 
              opacity={0.6}
            />
          </Box>
        </group>
      </group>
    </group>
  );
};

const DroneModel: React.FC<DroneModelProps> = ({ metadata }) => {
  const droneRef = useRef<THREE.Group>(null);
  const bodyRef = useRef<THREE.Mesh>(null);

  const { roll = 0, pitch = 0, yaw = 0, gimbal_elevation = 0, gimbal_azimuth = 0 } = metadata;

  useFrame(() => {
    if (droneRef.current) {
      // Smooth rotation interpolation
      const targetRotationX = pitch * Math.PI / 180;
      const targetRotationY = yaw * Math.PI / 180;
      const targetRotationZ = roll * Math.PI / 180;
      
      droneRef.current.rotation.x = THREE.MathUtils.lerp(droneRef.current.rotation.x, targetRotationX, 0.1);
      droneRef.current.rotation.y = THREE.MathUtils.lerp(droneRef.current.rotation.y, targetRotationY, 0.1);
      droneRef.current.rotation.z = THREE.MathUtils.lerp(droneRef.current.rotation.z, targetRotationZ, 0.1);
    }

    // Subtle hover animation
    if (bodyRef.current) {
      bodyRef.current.position.y = Math.sin(Date.now() * 0.001) * 0.02;
    }
  });

  // Arm positions and rotations
  const armConfigs = [
    { position: [-0.5, 0, -0.5], rotation: [0, Math.PI / 4, 0], propPos: [-0.8, 0.1, -0.8] },
    { position: [0.5, 0, -0.5], rotation: [0, -Math.PI / 4, 0], propPos: [0.8, 0.1, -0.8] },
    { position: [-0.5, 0, 0.5], rotation: [0, -Math.PI / 4, 0], propPos: [-0.8, 0.1, 0.8] },
    { position: [0.5, 0, 0.5], rotation: [0, Math.PI / 4, 0], propPos: [0.8, 0.1, 0.8] },
  ];

  return (
    <group ref={droneRef}>
      {/* Main drone body */}
      <Box ref={bodyRef} args={[0.8, 0.2, 0.6]}>
        <meshStandardMaterial 
          color="#00ddff" 
          metalness={0.7} 
          roughness={0.3}
          emissive="#003344"
          emissiveIntensity={0.1}
        />
      </Box>

      {/* Top cover with pattern */}
      <Box args={[0.6, 0.05, 0.4]} position={[0, 0.125, 0]}>
        <meshStandardMaterial 
          color="#ffffff" 
          metalness={0.8} 
          roughness={0.2}
        />
      </Box>

      {/* LED strips */}
      <Box args={[0.7, 0.02, 0.02]} position={[0, 0.11, -0.29]}>
        <meshStandardMaterial 
          color="#ff0080" 
          emissive="#ff0080" 
          emissiveIntensity={0.5}
        />
      </Box>
      <Box args={[0.7, 0.02, 0.02]} position={[0, 0.11, 0.29]}>
        <meshStandardMaterial 
          color="#00ff00" 
          emissive="#00ff00" 
          emissiveIntensity={0.5}
        />
      </Box>

      {/* Arms and propellers */}
      {armConfigs.map((config, index) => (
        <group key={index}>
          <DroneArm 
            position={config.position as [number, number, number]} 
            rotation={config.rotation as [number, number, number]} 
          />
          <PropellerBlade position={config.propPos as [number, number, number]} />
        </group>
      ))}

      {/* Landing gear */}
      {[-0.3, 0.3].map((x, i) => (
        <group key={i}>
          {[-0.2, 0.2].map((z, j) => (
            <Cylinder key={j} args={[0.02, 0.02, 0.3]} position={[x, -0.25, z]}>
              <meshStandardMaterial color="#444444" metalness={0.6} roughness={0.4} />
            </Cylinder>
          ))}
        </group>
      ))}

      {/* Battery indicator */}
      <Box args={[0.4, 0.08, 0.15]} position={[0, -0.05, 0]}>
        <meshStandardMaterial color="#222222" metalness={0.5} roughness={0.5} />
      </Box>

      {/* Camera gimbal */}
      <CameraGimbal gimbalElevation={gimbal_elevation} gimbalAzimuth={gimbal_azimuth} />
    </group>
  );
};

const AxisHelper: React.FC = () => {
  return (
    <group position={[0, -1, 0]}>
      {/* X-axis (Red) */}
      <Box args={[2, 0.02, 0.02]} position={[1, 0, 0]}>
        <meshBasicMaterial color="#ff4444" />
      </Box>
      <Text position={[2.2, 0, 0]} fontSize={0.2} color="#ff4444">X</Text>

      {/* Y-axis (Green) */}
      <Box args={[0.02, 2, 0.02]} position={[0, 1, 0]}>
        <meshBasicMaterial color="#44ff44" />
      </Box>
      <Text position={[0, 2.2, 0]} fontSize={0.2} color="#44ff44">Y</Text>

      {/* Z-axis (Blue) */}
      <Box args={[0.02, 0.02, 2]} position={[0, 0, 1]}>
        <meshBasicMaterial color="#4444ff" />
      </Box>
      <Text position={[0, 0, 2.2]} fontSize={0.2} color="#4444ff">Z</Text>
    </group>
  );
};

const TelemetryHUD: React.FC<{ metadata: DroneMetadata }> = ({ metadata }) => {
  const { viewport } = useThree();
  const hudRef = useRef<THREE.Group>(null);

  return (
    <group ref={hudRef} position={[-viewport.width * 0.4, viewport.height * 0.35, 0]}>
      <Text
        fontSize={viewport.width * 0.02}
        color="#00ffff"
        anchorX="left"
        anchorY="top"
        font="monospace"
        maxWidth={2}
      >
        {`ATTITUDE:
YAW   ${(metadata.yaw || 0).toFixed(1).padStart(6)}°
PITCH ${(metadata.pitch || 0).toFixed(1).padStart(6)}°
ROLL  ${(metadata.roll || 0).toFixed(1).padStart(6)}°

GIMBAL:
ELEV  ${(metadata.gimbal_elevation || 0).toFixed(1).padStart(6)}°
AZIM  ${(metadata.gimbal_azimuth || 0).toFixed(1).padStart(6)}°

ALTITUDE: ${(metadata.altitude || 0).toFixed(1)}m
HFOV:     ${(metadata.hfov || 0).toFixed(1)}°`}
      </Text>
    </group>
  );
};

const Scene: React.FC<{ metadata?: DroneMetadata }> = ({ metadata }) => {
  const { camera } = useThree();

  useEffect(() => {
    camera.position.set(3, 2, 3);
    camera.lookAt(0, 0, 0);
  }, [camera]);

  return (
    <>
      {/* Environment lighting */}
      <Environment preset="studio" />
      
      {/* Additional lighting */}
      <ambientLight intensity={0.2} />
      <directionalLight
        position={[10, 10, 5]}
        intensity={0.8}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />
      <pointLight position={[0, 3, 0]} intensity={0.3} color="#00ffff" />

      {/* Ground plane for shadows */}
      <mesh position={[0, -1.01, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeBufferGeometry args={[10, 10]} />
        <meshStandardMaterial color="#000033" opacity={0.3} transparent />
      </mesh>

      {/* Grid floor */}
      <primitive object={new THREE.GridHelper(10, 20, "#003333", "#001111")} position={[0, -1, 0]} />

      {/* Coordinate system */}
      <AxisHelper />

      {/* Drone model */}
      {metadata ? (
        <>
          <DroneModel metadata={metadata} />
          <TelemetryHUD metadata={metadata} />
        </>
      ) : (
        <Text
          position={[0, 0, 0]}
          fontSize={0.3}
          color="#666666"
          anchorX="center"
          anchorY="middle"
        >
          NO TELEMETRY DATA
          {'\n'}WAITING FOR DRONE SIGNAL...
        </Text>
      )}

      {/* Camera controls */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={1}
        maxDistance={10}
        maxPolarAngle={Math.PI * 0.45}
        autoRotate={false}
        autoRotateSpeed={0.5}
      />
    </>
  );
};

const DroneVisualization3D: React.FC<DroneVisualization3DProps> = ({
  metadata,
  className = '',
}) => {
  return (
    <div className={`${className} bg-cyber-black border border-cyber-border relative overflow-hidden`}>
      <Canvas
        shadows
        dpr={[1, 2]}
        camera={{ position: [3, 2, 3], fov: 60 }}
        gl={{ 
          antialias: true,
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 1.2
        }}
      >
        <Scene metadata={metadata} />
      </Canvas>
      
      {/* Status indicators */}
      <div className="absolute top-3 right-3 space-y-1">
        <div className={`font-mono text-xs px-2 py-1 border ${
          metadata ? 'text-neon-green border-neon-green bg-neon-green bg-opacity-10' : 'text-red-500 border-red-500 bg-red-500 bg-opacity-10'
        }`}>
          {metadata ? 'TELEMETRY_ONLINE' : 'TELEMETRY_OFFLINE'}
        </div>
        <div className="font-mono text-xs px-2 py-1 border text-neon-cyan border-neon-cyan bg-neon-cyan bg-opacity-10">
          3D_RENDER_ACTIVE
        </div>
      </div>
      
      {/* Controls hint */}
      <div className="absolute bottom-3 left-3 font-mono text-xs text-cyber-muted">
        MOUSE: ORBIT • WHEEL: ZOOM • RIGHT: PAN
      </div>
    </div>
  );
};

export default DroneVisualization3D;
