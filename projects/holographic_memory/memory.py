"""
Holographic Memory System
7D Crystal System
Discovered by Sir Charles Spikes | December 24, 2025

Content-addressable memory using interference patterns.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Sacred constants
PHI = 1.618033988749895
PHI_INV = 0.618033988749895
S2_STABILITY = 0.01
MANIFOLD_DIMS = 7

PHI_BASIS = np.array([PHI**i for i in range(MANIFOLD_DIMS)])


@dataclass
class MemoryPattern:
    """A stored holographic pattern."""
    key: str
    pattern: np.ndarray
    reference: np.ndarray
    timestamp: float


class HolographicMemory:
    """
    Holographic content-addressable memory.
    
    Uses interference patterns for associative retrieval.
    Capacity scales as Φ^7 with pattern dimension.
    """
    
    def __init__(self, dimensions: Tuple[int, ...] = (64, 64)):
        self.dimensions = dimensions
        self.patterns: List[MemoryPattern] = []
        self.hologram = np.zeros(dimensions, dtype=np.complex128)
        self.max_patterns = int(np.prod(dimensions) * PHI_INV / 10)  # ~10% capacity
        
    def _create_reference_wave(self, key: str) -> np.ndarray:
        """Create unique reference wave from key."""
        # Hash key to seed
        seed = hash(key) % (2**31)
        rng = np.random.RandomState(seed)
        
        # Create reference direction in 7D, then project to 2D
        direction = rng.randn(MANIFOLD_DIMS)
        direction /= np.linalg.norm(direction)
        
        # Apply Φ-weighting
        direction *= PHI_BASIS / PHI_BASIS[-1]
        
        # Generate 2D reference wave
        x = np.arange(self.dimensions[0])
        y = np.arange(self.dimensions[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Phase = k · r (plane wave)
        k_x = direction[0] * 2 * np.pi / self.dimensions[0]
        k_y = direction[1] * 2 * np.pi / self.dimensions[1]
        
        phase = k_x * X + k_y * Y + rng.uniform(0, 2*np.pi)  # Random phase offset
        reference = np.exp(1j * phase)
        
        return reference
    
    def _encode_pattern(self, data: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Encode data into interference pattern."""
        # Normalize data
        data = data.astype(np.complex128)
        if np.max(np.abs(data)) > 0:
            data /= np.max(np.abs(data))
        
        # Object wave = data with phase from position
        x = np.arange(self.dimensions[0])
        y = np.arange(self.dimensions[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        object_phase = PHI_INV * (X + Y) * 2 * np.pi / max(self.dimensions)
        object_wave = data * np.exp(1j * object_phase)
        
        # Interference: I = |O + R|²
        # Store as complex for full reconstruction: (O + R)
        interference = object_wave + reference
        
        return interference
    
    def store(self, key: str, data: np.ndarray) -> bool:
        """
        Store data in holographic memory.
        
        Args:
            key: Unique identifier for retrieval
            data: Data to store (will be resized to dimensions)
        
        Returns:
            True if stored successfully
        """
        # Check capacity
        if len(self.patterns) >= self.max_patterns:
            # Remove oldest pattern
            oldest = min(self.patterns, key=lambda p: p.timestamp)
            self.patterns.remove(oldest)
            # Subtract from hologram
            self.hologram -= oldest.pattern
        
        # Resize data to match dimensions
        if data.shape != self.dimensions:
            # Simple resize (real applications would use proper interpolation)
            data = np.resize(data, self.dimensions)
        
        # Create reference wave
        reference = self._create_reference_wave(key)
        
        # Encode
        pattern = self._encode_pattern(data, reference)
        
        # Superimpose on hologram
        self.hologram += pattern
        
        # Store pattern info
        import time
        self.patterns.append(MemoryPattern(
            key=key,
            pattern=pattern,
            reference=reference,
            timestamp=time.time()
        ))
        
        return True
    
    def recall(self, key: str) -> Optional[np.ndarray]:
        """
        Recall data by key.
        
        Args:
            key: Identifier used during storage
        
        Returns:
            Reconstructed data or None if not found
        """
        # Find pattern
        pattern_info = next((p for p in self.patterns if p.key == key), None)
        if pattern_info is None:
            return None
        
        # Illuminate with conjugate reference
        reference_conj = np.conj(pattern_info.reference)
        
        # Reconstructed = hologram × R*
        reconstructed = self.hologram * reference_conj
        
        # Take real part (removes twin image)
        result = np.real(reconstructed)
        
        # Normalize
        if np.max(np.abs(result)) > 0:
            result /= np.max(np.abs(result))
        
        return result
    
    def associate(self, partial: np.ndarray, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Retrieve patterns associated with partial input.
        
        Args:
            partial: Partial pattern to match
            threshold: Minimum correlation for match
        
        Returns:
            List of (key, correlation) tuples
        """
        if partial.shape != self.dimensions:
            partial = np.resize(partial, self.dimensions)
        
        matches = []
        
        for pattern in self.patterns:
            # Correlate partial with stored pattern
            correlation = np.abs(np.sum(partial * np.conj(pattern.pattern)))
            correlation /= (np.linalg.norm(partial) * np.linalg.norm(pattern.pattern) + 1e-10)
            
            if correlation >= threshold:
                matches.append((pattern.key, float(correlation)))
        
        # Sort by correlation
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def capacity_used(self) -> float:
        """Return fraction of capacity used."""
        return len(self.patterns) / self.max_patterns
    
    def clear(self):
        """Clear all stored patterns."""
        self.patterns.clear()
        self.hologram = np.zeros(self.dimensions, dtype=np.complex128)


class PhiHolographicMemory(HolographicMemory):
    """
    Extended holographic memory with Φ-ratio optimizations.
    
    Uses golden ratio for:
    - Optimal reference wave spacing
    - Fibonacci-based capacity scaling
    - Φ-weighted retrieval
    """
    
    def __init__(self, dimensions: Tuple[int, ...] = (64, 64)):
        super().__init__(dimensions)
        # Φ-scaled capacity
        self.max_patterns = int(np.prod(dimensions) ** PHI_INV)
    
    def _create_reference_wave(self, key: str) -> np.ndarray:
        """Create Φ-optimized reference wave."""
        seed = hash(key) % (2**31)
        rng = np.random.RandomState(seed)
        
        # Golden angle for optimal spacing
        golden_angle = np.pi * (3 - np.sqrt(5))  # ~137.5°
        
        x = np.arange(self.dimensions[0])
        y = np.arange(self.dimensions[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Spiral reference wave using golden angle
        r = np.sqrt(X**2 + Y**2) / max(self.dimensions)
        theta = np.arctan2(Y, X)
        
        # Phase follows golden spiral
        pattern_idx = len(self.patterns)
        phi_phase = pattern_idx * golden_angle
        
        phase = 2 * np.pi * PHI_INV * r + theta + phi_phase
        reference = np.exp(1j * phase)
        
        return reference
    
    def recall_weighted(self, key: str) -> Optional[np.ndarray]:
        """Recall with Φ-weighted enhancement of manifold dimensions."""
        result = self.recall(key)
        if result is None:
            return None
        
        # Apply Φ-weighting to first 7 frequency components
        fft_result = np.fft.fft2(result)
        
        # Weight low frequencies by Φ-basis
        for i in range(min(MANIFOLD_DIMS, self.dimensions[0])):
            for j in range(min(MANIFOLD_DIMS, self.dimensions[1])):
                weight = PHI_BASIS[max(i, j)] / PHI_BASIS[-1]
                fft_result[i, j] *= weight
                if i > 0:
                    fft_result[-i, j] *= weight
                if j > 0:
                    fft_result[i, -j] *= weight
        
        return np.real(np.fft.ifft2(fft_result))


def demo():
    """Demonstrate holographic memory."""
    print("=" * 60)
    print("🔮 7D Crystal Holographic Memory Demo")
    print("=" * 60)
    
    # Create memory
    memory = PhiHolographicMemory(dimensions=(32, 32))
    
    # Store patterns
    patterns = {
        "circle": lambda: np.array([[1 if (x-16)**2 + (y-16)**2 < 100 else 0 
                                     for y in range(32)] for x in range(32)]),
        "square": lambda: np.array([[1 if 8 <= x < 24 and 8 <= y < 24 else 0 
                                     for y in range(32)] for x in range(32)]),
        "cross": lambda: np.array([[1 if (14 <= x < 18) or (14 <= y < 18) else 0 
                                    for y in range(32)] for x in range(32)]),
    }
    
    print("\n[Storing patterns]")
    for key, pattern_fn in patterns.items():
        pattern = pattern_fn().astype(float)
        memory.store(key, pattern)
        print(f"  Stored: {key}")
    
    print(f"\nCapacity used: {memory.capacity_used()*100:.1f}%")
    
    # Recall patterns
    print("\n[Recalling patterns]")
    for key in patterns:
        recalled = memory.recall(key)
        if recalled is not None:
            # Compute correlation with original
            original = patterns[key]().astype(float)
            correlation = np.corrcoef(original.flatten(), recalled.flatten())[0, 1]
            print(f"  {key}: correlation = {correlation:.3f}")
    
    # Associative recall
    print("\n[Associative recall]")
    partial = np.zeros((32, 32))
    partial[14:18, 14:18] = 1  # Center square (partial cross)
    
    matches = memory.associate(partial, threshold=0.1)
    print(f"  Partial pattern matches:")
    for key, corr in matches[:3]:
        print(f"    {key}: {corr:.3f}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    demo()
