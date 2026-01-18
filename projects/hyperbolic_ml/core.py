"""
Hyperbolic ML Research Platform - Core Module
7D Crystal System
Discovered by Sir Charles Spikes | December 24, 2025
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════

PHI = 1.618033988749895
PHI_INV = 0.618033988749895
S2_STABILITY = 0.01
MANIFOLD_DIMS = 7
CURVATURE = -PHI_INV  # Negative curvature for hyperbolic space

PHI_BASIS = np.array([
    1.0, PHI, PHI**2, PHI**3, PHI**4, PHI**5, PHI**6
])


# ═══════════════════════════════════════════════════════════════
# POINCARÉ BALL OPERATIONS
# ═══════════════════════════════════════════════════════════════

def project_to_poincare(v: np.ndarray, curvature: float = CURVATURE) -> np.ndarray:
    """
    Project vector onto Poincaré ball with 7D constraints.
    
    x → x / (1 + ||v|| + Φ⁻¹ + |κ|)
    """
    norm = np.linalg.norm(v)
    denom = 1.0 + norm + PHI_INV + abs(curvature)
    
    # Apply S² stability bound
    if norm > S2_STABILITY:
        scale = 1.0 / (denom * (norm / S2_STABILITY))
    else:
        scale = 1.0 / denom
    
    projected = v * scale
    
    # Apply Φ-weighting to first 7 dimensions
    phi_weights = np.ones_like(v)
    phi_weights[:min(MANIFOLD_DIMS, len(v))] = PHI_BASIS[:min(MANIFOLD_DIMS, len(v))] / PHI_BASIS[6]
    
    return projected * phi_weights


def hyperbolic_distance(u: np.ndarray, v: np.ndarray, curvature: float = CURVATURE) -> float:
    """
    Compute hyperbolic distance in Poincaré ball.
    
    d(u, v) = acosh(1 + 2*||u-v||² / ((1-||u||²)(1-||v||²)))
    """
    c = abs(curvature)
    u_norm_sq = np.sum(u**2)
    v_norm_sq = np.sum(v**2)
    diff_norm_sq = np.sum((u - v)**2)
    
    numerator = 2.0 * c * diff_norm_sq
    denominator = max((1 - c * u_norm_sq) * (1 - c * v_norm_sq), 1e-10)
    
    return np.arccosh(1 + numerator / denominator)


def mobius_add(u: np.ndarray, v: np.ndarray, curvature: float = CURVATURE) -> np.ndarray:
    """
    Möbius addition in hyperbolic space.
    
    u ⊕ v = ((1 + 2c⟨u,v⟩ + c||v||²)u + (1 - c||u||²)v) / (1 + 2c⟨u,v⟩ + c²||u||²||v||²)
    """
    c = abs(curvature)
    u_norm_sq = np.sum(u**2)
    v_norm_sq = np.sum(v**2)
    uv_dot = np.dot(u, v)
    
    numerator_u = 1 + 2*c*uv_dot + c*v_norm_sq
    numerator_v = 1 - c*u_norm_sq
    denominator = 1 + 2*c*uv_dot + c*c*u_norm_sq*v_norm_sq
    
    return (numerator_u * u + numerator_v * v) / max(denominator, 1e-10)


def exp_map(v: np.ndarray, curvature: float = CURVATURE) -> np.ndarray:
    """Exponential map at origin: Euclidean → Hyperbolic"""
    c = abs(curvature)
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    sqrt_c = np.sqrt(c)
    return np.tanh(sqrt_c * norm / 2) * v / (sqrt_c * norm)


def log_map(v: np.ndarray, curvature: float = CURVATURE) -> np.ndarray:
    """Logarithmic map at origin: Hyperbolic → Euclidean"""
    c = abs(curvature)
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    sqrt_c = np.sqrt(c)
    return 2 * np.arctanh(sqrt_c * norm) * v / (sqrt_c * norm)


# ═══════════════════════════════════════════════════════════════
# HYPERBOLIC EMBEDDINGS
# ═══════════════════════════════════════════════════════════════

@dataclass
class HyperbolicEmbeddingConfig:
    """Configuration for hyperbolic embeddings."""
    vocab_size: int = 50000
    embedding_dim: int = 128
    curvature: float = CURVATURE
    manifold_project: bool = True
    phi_weighted: bool = True


class HyperbolicEmbedding:
    """
    Word embeddings in hyperbolic space.
    
    Uses Poincaré ball model with Φ-weighted initialization.
    """
    
    def __init__(self, config: HyperbolicEmbeddingConfig):
        self.config = config
        
        # Initialize embeddings uniformly, then project to ball
        scale = 0.1 / np.sqrt(config.embedding_dim)
        self.embeddings = np.random.randn(config.vocab_size, config.embedding_dim) * scale
        
        # Project to Poincaré ball
        if config.manifold_project:
            for i in range(config.vocab_size):
                self.embeddings[i] = project_to_poincare(
                    self.embeddings[i], 
                    config.curvature
                )
    
    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        """Look up embeddings for token IDs."""
        embeds = self.embeddings[token_ids]
        
        if self.config.manifold_project:
            # Ensure outputs are on manifold
            batch_shape = embeds.shape[:-1]
            flat = embeds.reshape(-1, embeds.shape[-1])
            projected = np.array([project_to_poincare(v, self.config.curvature) for v in flat])
            return projected.reshape(*batch_shape, -1)
        
        return embeds
    
    def distance(self, i: int, j: int) -> float:
        """Compute hyperbolic distance between two tokens."""
        return hyperbolic_distance(
            self.embeddings[i], 
            self.embeddings[j], 
            self.config.curvature
        )
    
    def nearest_neighbors(self, token_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """Find k nearest neighbors in hyperbolic space."""
        query = self.embeddings[token_id]
        distances = []
        
        for i in range(self.config.vocab_size):
            if i != token_id:
                d = hyperbolic_distance(query, self.embeddings[i], self.config.curvature)
                distances.append((i, d))
        
        distances.sort(key=lambda x: x[1])
        return distances[:k]


# ═══════════════════════════════════════════════════════════════
# HYPERBOLIC ATTENTION
# ═══════════════════════════════════════════════════════════════

class HyperbolicAttention:
    """
    Attention mechanism in hyperbolic space.
    
    Uses hyperbolic distance for attention scores instead of dot product.
    Applies Φ-weighted scaling.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        curvature: float = CURVATURE,
        phi_weighted: bool = True
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.curvature = curvature
        self.phi_weighted = phi_weighted
        
        # Initialize projections
        scale = 1.0 / np.sqrt(hidden_dim)
        self.W_q = np.random.randn(hidden_dim, hidden_dim) * scale
        self.W_k = np.random.randn(hidden_dim, hidden_dim) * scale
        self.W_v = np.random.randn(hidden_dim, hidden_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim) * scale
    
    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply hyperbolic attention.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            mask: Optional attention mask [batch, seq_len, seq_len]
        
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = x @ self.W_q  # [batch, seq, hidden]
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Reshape to heads
        Q = Q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch, seq_len, self.num_heads, self.head_dim)
        
        # Project Q, K to Poincaré ball
        Q = np.array([[[[project_to_poincare(Q[b,s,h], self.curvature) 
                         for h in range(self.num_heads)] 
                        for s in range(seq_len)] 
                       for b in range(batch)])
        K = np.array([[[[project_to_poincare(K[b,s,h], self.curvature) 
                         for h in range(self.num_heads)] 
                        for s in range(seq_len)] 
                       for b in range(batch)])
        
        # Compute hyperbolic attention scores
        # score(q, k) = -d_H(q, k) (negative distance → higher similarity)
        scores = np.zeros((batch, self.num_heads, seq_len, seq_len))
        
        for b in range(batch):
            for h in range(self.num_heads):
                for i in range(seq_len):
                    for j in range(seq_len):
                        d = hyperbolic_distance(Q[b,i,h], K[b,j,h], self.curvature)
                        scores[b, h, i, j] = -d
        
        # Apply Φ-weighting to heads
        if self.phi_weighted:
            phi_weights = PHI_BASIS[:min(self.num_heads, MANIFOLD_DIMS)]
            phi_weights = phi_weights / phi_weights.sum()
            for h in range(min(self.num_heads, len(phi_weights))):
                scores[:, h, :, :] *= phi_weights[h]
        
        # Apply mask
        if mask is not None:
            scores = scores + mask * (-1e9)
        
        # Softmax
        scores = scores - scores.max(axis=-1, keepdims=True)
        attn_weights = np.exp(scores) / (np.exp(scores).sum(axis=-1, keepdims=True) + 1e-10)
        
        # Apply to values
        V = V.transpose(0, 2, 1, 3)  # [batch, heads, seq, dim]
        output = attn_weights @ V  # [batch, heads, seq, dim]
        
        # Combine heads
        output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        
        # Output projection
        output = output @ self.W_o
        
        return output


# ═══════════════════════════════════════════════════════════════
# Φ-OPTIMIZED LAYERS
# ═══════════════════════════════════════════════════════════════

class PhiRMSNorm:
    """RMSNorm with 7D stability constraints."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.weight = np.ones(dim)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Compute RMS
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        normalized = x / rms * self.weight
        
        # Apply 7D stability bounds
        for i in range(min(MANIFOLD_DIMS, normalized.shape[-1])):
            bound = S2_STABILITY * 100.0 * (PHI_BASIS[i] / PHI_BASIS[6])
            normalized[..., i] = np.clip(normalized[..., i], -bound, bound)
        
        return normalized


class PhiSwiGLU:
    """SwiGLU FFN with Φ constraints."""
    
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        
        scale = 1.0 / np.sqrt(hidden_dim)
        self.W_gate = np.random.randn(hidden_dim, intermediate_dim) * scale
        self.W_up = np.random.randn(hidden_dim, intermediate_dim) * scale
        self.W_down = np.random.randn(intermediate_dim, hidden_dim) * scale
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Gate and up projections
        gate = x @ self.W_gate
        up = x @ self.W_up
        
        # SwiGLU activation
        silu = gate / (1 + np.exp(-gate))
        hidden = silu * up
        
        # Apply Φ constraint to first 7 dims
        hidden[..., :MANIFOLD_DIMS] *= PHI_INV
        
        # Down projection
        return hidden @ self.W_down


# ═══════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════

def run_tests():
    """Run mathematical verification tests."""
    print("=" * 60)
    print("7D Crystal Hyperbolic ML - Mathematical Verification")
    print("=" * 60)
    
    # Test 1: Poincaré projection stays inside ball
    print("\n[TEST 1] Poincaré Ball Containment")
    for _ in range(100):
        v = np.random.randn(128) * 10
        p = project_to_poincare(v)
        norm = np.linalg.norm(p)
        assert norm < 1.0, f"Norm {norm} >= 1.0"
    print("  ✓ All projections inside unit ball")
    
    # Test 2: Φ identity
    print("\n[TEST 2] Φ Identity Verification")
    assert abs(PHI * PHI - (PHI + 1)) < 1e-14, "Φ² ≠ Φ + 1"
    assert abs(1/PHI - (PHI - 1)) < 1e-14, "Φ⁻¹ ≠ Φ - 1"
    print("  ✓ Φ² = Φ + 1")
    print("  ✓ Φ⁻¹ = Φ - 1")
    
    # Test 3: Fibonacci property
    print("\n[TEST 3] Fibonacci Property")
    for i in range(5):
        sum_val = PHI_BASIS[i] + PHI_BASIS[i+1]
        assert abs(sum_val - PHI_BASIS[i+2]) < 1e-10
    print("  ✓ PHI_BASIS[i+2] = PHI_BASIS[i+1] + PHI_BASIS[i]")
    
    # Test 4: Hyperbolic distance triangle inequality
    print("\n[TEST 4] Triangle Inequality")
    for _ in range(50):
        a = project_to_poincare(np.random.randn(32) * 0.1)
        b = project_to_poincare(np.random.randn(32) * 0.1)
        c = project_to_poincare(np.random.randn(32) * 0.1)
        
        d_ab = hyperbolic_distance(a, b)
        d_bc = hyperbolic_distance(b, c)
        d_ac = hyperbolic_distance(a, c)
        
        assert d_ac <= d_ab + d_bc + 1e-6, "Triangle inequality violated"
    print("  ✓ d(a,c) ≤ d(a,b) + d(b,c)")
    
    # Test 5: Möbius addition identity
    print("\n[TEST 5] Möbius Addition Identity")
    for _ in range(50):
        v = project_to_poincare(np.random.randn(32) * 0.1)
        zero = np.zeros_like(v)
        result = mobius_add(v, zero)
        assert np.allclose(result, v, atol=1e-6), "v ⊕ 0 ≠ v"
    print("  ✓ v ⊕ 0 = v")
    
    # Test 6: Embedding creation
    print("\n[TEST 6] Hyperbolic Embedding")
    config = HyperbolicEmbeddingConfig(vocab_size=1000, embedding_dim=64)
    embeddings = HyperbolicEmbedding(config)
    
    token_ids = np.array([0, 1, 2, 3, 4])
    embeds = embeddings(token_ids)
    assert embeds.shape == (5, 64)
    
    for e in embeds:
        assert np.linalg.norm(e) < 1.0, "Embedding outside ball"
    print("  ✓ Embeddings created and inside ball")
    
    # Test 7: Attention forward
    print("\n[TEST 7] Hyperbolic Attention")
    attn = HyperbolicAttention(hidden_dim=64, num_heads=4)
    x = np.random.randn(2, 8, 64) * 0.1
    output = attn(x)
    assert output.shape == (2, 8, 64)
    print("  ✓ Attention forward pass successful")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
