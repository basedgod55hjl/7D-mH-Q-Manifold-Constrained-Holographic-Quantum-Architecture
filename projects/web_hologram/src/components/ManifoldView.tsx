"use client";

import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Stars, Text } from "@react-three/drei";
import * as THREE from "three";

const PHI = 1.618033988749895;

function CrystalNode({ position, color = "cyan", size = 0.1 }: { position: [number, number, number], color?: string, size?: number }) {
    const meshRef = useRef<THREE.Mesh>(null);

    useFrame((state) => {
        if (!meshRef.current) return;
        const t = state.clock.getElapsedTime();
        meshRef.current.rotation.x = Math.sin(t * 0.5) * 0.2;
        meshRef.current.rotation.y += 0.01;
    });

    return (
        <mesh ref={meshRef} position={position}>
            <dodecahedronGeometry args={[size, 0]} />
            <meshStandardMaterial color={color} emissive={color} emissiveIntensity={2} wireframe />
        </mesh>
    );
}

function Manifold() {
    const points = useMemo(() => {
        const p: [number, number, number][] = [];
        const count = 50;
        for (let i = 0; i < count; i++) {
            // Simple hyperbolic distribution simulation
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const r = Math.pow(Math.random(), PHI); // Phi-weighted distribution

            p.push([
                r * Math.sin(phi) * Math.cos(theta) * 4,
                r * Math.sin(phi) * Math.sin(theta) * 4,
                r * Math.cos(phi) * 4
            ]);
        }
        return p;
    }, []);

    return (
        <group>
            {/* The Poincaré Ball Boundary */}
            <mesh>
                <sphereGeometry args={[4.5, 32, 32]} />
                <meshStandardMaterial color="black" transparent opacity={0.1} wireframe />
            </mesh>

            {/* Central Core */}
            <CrystalNode position={[0, 0, 0]} color="#00ffcc" size={0.5} />

            {/* Scattered Manifold Nodes */}
            {points.map((pos, i) => (
                <CrystalNode key={i} position={pos} color={i % 2 === 0 ? "#ff00ff" : "#00ffff"} size={0.15} />
            ))}

            {/* Orbital Rings */}
            <mesh rotation={[Math.PI / 2, 0, 0]}>
                <torusGeometry args={[3, 0.02, 16, 100]} />
                <meshStandardMaterial color="#ffffff" emissive="#ffffff" />
            </mesh>
            <mesh rotation={[0, Math.PI / 6, 0]}>
                <torusGeometry args={[3.5, 0.02, 16, 100]} />
                <meshStandardMaterial color="#ffffff" emissive="#ffffff" />
            </mesh>
        </group>
    );
}

export default function ManifoldView() {
    return (
        <div className="w-full h-screen bg-black relative">
            <div className="absolute top-8 left-8 z-10 pointer-events-none">
                <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-600">
                    7D HOLOGRAPHIC INTERFACE
                </h1>
                <p className="text-cyan-200 uppercase tracking-widest text-xs mt-2">
                    Manifold Status: STABLE (Φ = 1.618)
                </p>
            </div>

            <Canvas camera={{ position: [0, 0, 10], fov: 60 }}>
                <color attach="background" args={["#050505"]} />
                <fog attach="fog" args={["#050505", 5, 20]} />

                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} intensity={1} />
                <pointLight position={[-10, -10, -10]} color="purple" intensity={1} />

                <Manifold />
                <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
                <OrbitControls autoRotate autoRotateSpeed={0.5} />
            </Canvas>
        </div>
    );
}
