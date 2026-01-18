"use client";

import { useRef, useMemo, useState, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Stars } from "@react-three/drei";
import * as THREE from "three";

const PHI = 1.618033988749895;

// ═══════════════════════════════════════════════════════════════
// 3D Components
// ═══════════════════════════════════════════════════════════════

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

function Manifold({ rotationSpeed = 0.5 }: { rotationSpeed?: number }) {
    const groupRef = useRef<THREE.Group>(null);
    useFrame((state, delta) => {
        if (groupRef.current) {
            groupRef.current.rotation.y += delta * 0.1 * rotationSpeed;
        }
    });

    const points = useMemo(() => {
        const p: [number, number, number][] = [];
        const count = 50;
        for (let i = 0; i < count; i++) {
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const r = Math.pow(Math.random(), PHI);

            p.push([
                r * Math.sin(phi) * Math.cos(theta) * 4,
                r * Math.sin(phi) * Math.sin(theta) * 4,
                r * Math.cos(phi) * 4
            ]);
        }
        return p;
    }, []);

    return (
        <group ref={groupRef}>
            <mesh>
                <sphereGeometry args={[4.5, 32, 32]} />
                <meshStandardMaterial color="black" transparent opacity={0.1} wireframe />
            </mesh>
            <CrystalNode position={[0, 0, 0]} color="#00ffcc" size={0.5} />
            {points.map((pos, i) => (
                <CrystalNode key={i} position={pos} color={i % 2 === 0 ? "#ff00ff" : "#00ffff"} size={0.15} />
            ))}
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

// ═══════════════════════════════════════════════════════════════
// UI Components
// ═══════════════════════════════════════════════════════════════

interface Message {
    role: "user" | "assistant";
    content: string;
}

export default function ManifoldView() {
    const [prompt, setPrompt] = useState("");
    const [messages, setMessages] = useState<Message[]>([
        { role: "assistant", content: "Greetings. I am the 7D Crystal Sovereign Intelligence. I operate within Φ-ratio constrained hyperbolic patterns. How may I assist you?" }
    ]);
    const [loading, setLoading] = useState(false);
    const [rotationSpeed, setRotationSpeed] = useState(0.5);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async () => {
        if (!prompt.trim()) return;

        const userMsg: Message = { role: "user", content: prompt };
        setMessages(prev => [...prev, userMsg]);
        setPrompt("");
        setLoading(true);
        setRotationSpeed(2.0);

        try {
            // Prepare context (last 10 messages max to keep it light)
            const contextMessages = [...messages, userMsg].slice(-10);

            const res = await fetch("http://127.0.0.1:8080/v1/chat/completions", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model: "7d-crystal-8b",
                    messages: contextMessages,
                    max_tokens: 512,
                    temperature: 0.7
                })
            });

            if (!res.ok) throw new Error(`Server returned ${res.status}`);

            const data = await res.json();
            const aiMsg: Message = {
                role: "assistant",
                content: data.choices[0]?.message?.content || "Manifold divergence detected. No response received."
            };
            setMessages(prev => [...prev, aiMsg]);

        } catch (e) {
            console.error(e);
            setMessages(prev => [...prev, { role: "assistant", content: "⚠️ Connection to Neural Substrate Failed. Ensure Inference Server is running." }]);
        } finally {
            setLoading(false);
            setRotationSpeed(0.5);
        }
    };

    return (
        <div className="w-full h-screen bg-black relative font-mono overflow-hidden">
            {/* Header */}
            <div className="absolute top-0 left-0 w-full p-6 z-10 bg-gradient-to-b from-black/80 to-transparent pointer-events-none flex justify-between items-start">
                <div>
                    <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-600 drop-shadow-[0_0_10px_rgba(34,211,238,0.5)]">
                        7D HOLOGRAPHIC INTERFACE
                    </h1>
                    <p className="text-cyan-200 uppercase tracking-widest text-xs mt-2 flex items-center gap-2">
                        <span className={`w-2 h-2 rounded-full ${loading ? "bg-yellow-400 animate-pulse" : "bg-green-400"}`}></span>
                        Manifold Status: {loading ? "COMPUTING TENSOR FLUX..." : "STABLE (Φ = 1.618)"}
                    </p>
                </div>
            </div>

            {/* Chat Container */}
            <div className="absolute top-24 bottom-24 right-0 w-full max-w-2xl px-6 z-20 flex flex-col pointer-events-none">
                <div className="flex-1 overflow-y-auto pr-2 space-y-4 pointer-events-auto scrollbar-thin scrollbar-thumb-cyan-900 scrollbar-track-transparent">
                    {messages.map((msg, idx) => (
                        <div key={idx} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                            <div className={`
                                max-w-[85%] p-4 rounded-lg backdrop-blur-md border shadow-[0_0_15px_rgba(0,0,0,0.5)]
                                ${msg.role === "user"
                                    ? "bg-cyan-900/30 border-cyan-500/30 text-cyan-50 rounded-tr-none"
                                    : "bg-purple-900/30 border-purple-500/30 text-purple-50 rounded-tl-none"
                                }
                            `}>
                                <div className="text-[10px] uppercase opacity-50 mb-1 tracking-wider">
                                    {msg.role === "user" ? "Operator" : "7D Intelligence"}
                                </div>
                                <div className="leading-relaxed whitespace-pre-wrap">
                                    {msg.content}
                                </div>
                            </div>
                        </div>
                    ))}
                    <div ref={messagesEndRef} />
                </div>
            </div>

            {/* Input Area */}
            <div className="absolute bottom-0 w-full p-6 z-20 bg-gradient-to-t from-black/90 to-transparent">
                <div className="max-w-3xl mx-auto flex gap-3">
                    <input
                        type="text"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="Enter 7D Coordinates / Prompt..."
                        className="flex-1 bg-black/40 border border-cyan-500/30 rounded-lg px-4 py-4 text-white outline-none focus:border-cyan-400 focus:bg-black/60 transition-all placeholder:text-white/20 backdrop-blur-md shadow-[0_0_10px_rgba(0,255,255,0.05)]"
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                    />
                    <button
                        onClick={handleSend}
                        disabled={loading}
                        className="bg-cyan-600/20 hover:bg-cyan-600/40 text-cyan-400 border border-cyan-500/50 px-8 rounded-lg uppercase tracking-wider text-sm font-bold transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-[0_0_20px_rgba(34,211,238,0.4)]"
                    >
                        {loading ? "Aligning..." : "Transmit"}
                    </button>
                </div>
            </div>

            {/* 3D Background */}
            <Canvas camera={{ position: [0, 0, 10], fov: 60 }}>
                <color attach="background" args={["#030303"]} />
                <fog attach="fog" args={["#030303", 5, 25]} />

                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} intensity={1} />
                <pointLight position={[-10, -10, -10]} color="purple" intensity={1} />

                <Manifold rotationSpeed={rotationSpeed} />
                <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
                <OrbitControls autoRotate autoRotateSpeed={rotationSpeed} enableZoom={false} />
            </Canvas>
        </div>
    );
}
