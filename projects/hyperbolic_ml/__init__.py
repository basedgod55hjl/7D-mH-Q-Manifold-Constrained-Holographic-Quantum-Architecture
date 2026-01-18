"""
Hyperbolic ML - Python Package Init
"""

from .core import (
    PHI, PHI_INV, S2_STABILITY, MANIFOLD_DIMS, PHI_BASIS,
    project_to_poincare, hyperbolic_distance, mobius_add,
    exp_map, log_map,
    HyperbolicEmbeddingConfig, HyperbolicEmbedding,
    HyperbolicAttention, PhiRMSNorm, PhiSwiGLU,
    run_tests
)

__version__ = "1.0.0"
__author__ = "Sir Charles Spikes"

__all__ = [
    "PHI", "PHI_INV", "S2_STABILITY", "MANIFOLD_DIMS", "PHI_BASIS",
    "project_to_poincare", "hyperbolic_distance", "mobius_add",
    "exp_map", "log_map",
    "HyperbolicEmbeddingConfig", "HyperbolicEmbedding",
    "HyperbolicAttention", "PhiRMSNorm", "PhiSwiGLU",
    "run_tests"
]
