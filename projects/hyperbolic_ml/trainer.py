"""
Φ-Optimized Training Loop
7D Crystal System
Discovered by Sir Charles Spikes | December 24, 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

from .core import (
    PHI, PHI_INV, PHI_BASIS, S2_STABILITY, MANIFOLD_DIMS,
    project_to_poincare, hyperbolic_distance, mobius_add
)


# ═══════════════════════════════════════════════════════════════
# Φ-SCALED LEARNING RATE SCHEDULES
# ═══════════════════════════════════════════════════════════════

class LRSchedule(Enum):
    CONSTANT = "constant"
    COSINE = "cosine"
    PHI_DECAY = "phi_decay"        # Φ-ratio decay
    FIBONACCI_WARMUP = "fib_warm"  # Fibonacci warmup


def get_phi_learning_rate(
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float = 0.0,
    warmup_steps: int = 0,
    schedule: LRSchedule = LRSchedule.PHI_DECAY
) -> float:
    """
    Get learning rate with Φ-scaled schedule.
    """
    if step < warmup_steps:
        # Fibonacci warmup: progress through Fibonacci ratios
        fib_step = min(int(step / warmup_steps * 7), 6)
        return base_lr * PHI_BASIS[fib_step] / PHI_BASIS[6]
    
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    
    if schedule == LRSchedule.CONSTANT:
        return base_lr
    
    elif schedule == LRSchedule.COSINE:
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
    
    elif schedule == LRSchedule.PHI_DECAY:
        # Exponential decay with Φ⁻¹ base
        decay = PHI_INV ** (progress * 7)  # Decay through 7 Φ powers
        return max(min_lr, base_lr * decay)
    
    elif schedule == LRSchedule.FIBONACCI_WARMUP:
        # Inverse Fibonacci progression
        fib_step = min(int(progress * 7), 6)
        return base_lr * PHI_BASIS[6 - fib_step] / PHI_BASIS[6]
    
    return base_lr


# ═══════════════════════════════════════════════════════════════
# Φ-SOPHIA OPTIMIZER
# ═══════════════════════════════════════════════════════════════

@dataclass
class PhiSophiaConfig:
    """Configuration for Φ-Sophia optimizer."""
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    rho: float = 0.04           # Clipping threshold
    weight_decay: float = 0.1
    eps: float = 1e-12
    phi_scale_layers: bool = True  # Apply Φ scaling per layer
    manifold_constraint: bool = True
    s2_bound: float = S2_STABILITY


class PhiSophia:
    """
    Φ-Sophia Optimizer: Second-order optimization with golden ratio scaling.
    
    Combines Sophia's diagonal Hessian estimation with:
    - Φ-scaled learning rates per layer
    - S² stability bounds on updates
    - Manifold-projected gradients
    """
    
    def __init__(self, params: Dict[str, np.ndarray], config: PhiSophiaConfig):
        self.params = params
        self.config = config
        self.step_count = 0
        
        # State
        self.m = {k: np.zeros_like(v) for k, v in params.items()}  # First moment
        self.h = {k: np.ones_like(v) for k, v in params.items()}   # Hessian diagonal estimate
        
        # Per-layer Φ scaling
        if config.phi_scale_layers:
            self.layer_scales = self._compute_layer_scales()
        else:
            self.layer_scales = {k: 1.0 for k in params}
    
    def _compute_layer_scales(self) -> Dict[str, float]:
        """Compute Φ-based learning rate scales per layer."""
        scales = {}
        layer_names = list(self.params.keys())
        n_layers = len(layer_names)
        
        for i, name in enumerate(layer_names):
            # Map layer index to PHI_BASIS
            phi_idx = min(i % MANIFOLD_DIMS, 6)
            # Earlier layers get higher LR (inverse Φ scaling)
            scales[name] = PHI_BASIS[6 - phi_idx] / PHI_BASIS[6]
        
        return scales
    
    def step(
        self, 
        grads: Dict[str, np.ndarray],
        hessians: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Perform one optimization step.
        
        Args:
            grads: Gradients for each parameter
            hessians: Optional Hessian diagonal estimates (if None, use EMA)
        """
        self.step_count += 1
        c = self.config
        
        for name, param in self.params.items():
            if name not in grads:
                continue
            
            grad = grads[name]
            
            # Project gradient to manifold if enabled
            if c.manifold_constraint and len(grad.shape) >= 2:
                flat_shape = (-1, grad.shape[-1])
                flat_grad = grad.reshape(flat_shape)
                projected = np.array([project_to_poincare(g) for g in flat_grad])
                grad = projected.reshape(grad.shape)
            
            # Update momentum (EMA of gradients)
            self.m[name] = c.beta1 * self.m[name] + (1 - c.beta1) * grad
            
            # Update Hessian estimate
            if hessians is not None and name in hessians:
                self.h[name] = c.beta2 * self.h[name] + (1 - c.beta2) * hessians[name]
            else:
                # Fallback: use gradient squared as Hessian approximation
                self.h[name] = c.beta2 * self.h[name] + (1 - c.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[name] / (1 - c.beta1 ** self.step_count)
            h_hat = self.h[name] / (1 - c.beta2 ** self.step_count)
            
            # Compute update with clipping
            update = m_hat / (np.sqrt(h_hat) + c.eps)
            update = np.clip(update, -c.rho, c.rho)
            
            # Apply Φ-scaled learning rate
            effective_lr = c.lr * self.layer_scales[name]
            
            # Weight decay
            param -= effective_lr * c.weight_decay * param
            
            # Parameter update
            param -= effective_lr * update
            
            # Apply S² stability bound
            if c.manifold_constraint:
                norm = np.linalg.norm(param)
                if norm > c.s2_bound * 1000:
                    param *= c.s2_bound * 1000 / norm
            
            self.params[name] = param


# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Configuration for training."""
    max_steps: int = 10000
    batch_size: int = 32
    gradient_accumulation: int = 1
    lr: float = 1e-4
    warmup_steps: int = 100
    lr_schedule: LRSchedule = LRSchedule.PHI_DECAY
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    max_grad_norm: float = 1.0
    seed: int = 42


class PhiTrainer:
    """
    Φ-optimized training loop for 7D Crystal models.
    
    Features:
    - Φ-scaled learning rate scheduling
    - Manifold-projected gradients
    - S² stability enforcement
    - Holographic gradient accumulation
    """
    
    def __init__(
        self,
        model: Dict[str, np.ndarray],
        config: TrainingConfig,
        loss_fn: Callable,
        data_iter: Callable
    ):
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        self.data_iter = data_iter
        
        # Initialize optimizer
        opt_config = PhiSophiaConfig(lr=config.lr)
        self.optimizer = PhiSophia(model, opt_config)
        
        # Metrics
        self.metrics_history: List[Dict] = []
        
        np.random.seed(config.seed)
    
    def compute_gradients(self, batch: Dict) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Compute loss and gradients for a batch.
        (Placeholder - real impl would use autograd)
        """
        loss = self.loss_fn(self.model, batch)
        
        # Numerical gradient approximation (for demo)
        grads = {}
        eps = 1e-5
        
        for name, param in self.model.items():
            grad = np.zeros_like(param)
            flat_param = param.flatten()
            flat_grad = grad.flatten()
            
            # Sample a subset of gradients for efficiency
            sample_size = min(100, len(flat_param))
            indices = np.random.choice(len(flat_param), sample_size, replace=False)
            
            for idx in indices:
                original = flat_param[idx]
                
                flat_param[idx] = original + eps
                self.model[name] = flat_param.reshape(param.shape)
                loss_plus = self.loss_fn(self.model, batch)
                
                flat_param[idx] = original - eps
                self.model[name] = flat_param.reshape(param.shape)
                loss_minus = self.loss_fn(self.model, batch)
                
                flat_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
                flat_param[idx] = original
            
            self.model[name] = flat_param.reshape(param.shape)
            grads[name] = flat_grad.reshape(param.shape)
        
        return loss, grads
    
    def clip_gradients(self, grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Clip gradients by global norm."""
        total_norm_sq = sum(np.sum(g**2) for g in grads.values())
        total_norm = np.sqrt(total_norm_sq)
        
        if total_norm > self.config.max_grad_norm:
            scale = self.config.max_grad_norm / total_norm
            return {k: v * scale for k, v in grads.items()}
        
        return grads
    
    def train(self) -> Dict:
        """Run training loop."""
        print("=" * 60)
        print("🔮 7D Crystal Φ-Optimized Training")
        print("=" * 60)
        
        config = self.config
        accumulated_grads = {k: np.zeros_like(v) for k, v in self.model.items()}
        accumulated_loss = 0.0
        
        start_time = time.time()
        
        for step in range(1, config.max_steps + 1):
            # Get batch
            batch = self.data_iter()
            
            # Compute gradients
            loss, grads = self.compute_gradients(batch)
            
            # Accumulate
            for k in grads:
                accumulated_grads[k] += grads[k] / config.gradient_accumulation
            accumulated_loss += loss / config.gradient_accumulation
            
            # Update on accumulation boundary
            if step % config.gradient_accumulation == 0:
                # Clip gradients
                clipped_grads = self.clip_gradients(accumulated_grads)
                
                # Update learning rate
                lr = get_phi_learning_rate(
                    step, config.max_steps, config.lr,
                    warmup_steps=config.warmup_steps,
                    schedule=config.lr_schedule
                )
                self.optimizer.config.lr = lr
                
                # Optimizer step
                self.optimizer.step(clipped_grads)
                
                # Reset accumulation
                accumulated_grads = {k: np.zeros_like(v) for k, v in self.model.items()}
                
                # Log
                if step % config.log_interval == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = step / elapsed
                    
                    metrics = {
                        "step": step,
                        "loss": accumulated_loss,
                        "lr": lr,
                        "steps/sec": steps_per_sec,
                    }
                    self.metrics_history.append(metrics)
                    
                    print(f"[Step {step:6d}] loss={accumulated_loss:.4f} lr={lr:.2e} "
                          f"steps/s={steps_per_sec:.1f}")
                
                accumulated_loss = 0.0
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Total steps: {config.max_steps}")
        print(f"Final loss: {self.metrics_history[-1]['loss']:.4f}")
        print("=" * 60)
        
        return {
            "final_loss": self.metrics_history[-1]["loss"],
            "history": self.metrics_history
        }


# ═══════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════

def demo_training():
    """Demonstrate Φ-optimized training."""
    print("🔮 Φ-Optimized Training Demo")
    print("=" * 40)
    
    # Create dummy model
    model = {
        "embed": np.random.randn(100, 64) * 0.1,
        "layer1.W": np.random.randn(64, 64) * 0.1,
        "layer2.W": np.random.randn(64, 64) * 0.1,
        "head.W": np.random.randn(64, 10) * 0.1,
    }
    
    # Dummy loss function (MSE to random target)
    target = np.random.randn(32, 10) * 0.1
    def loss_fn(model, batch):
        # Simple forward pass
        x = batch["input"]  # [32, 64]
        x = x @ model["layer1.W"]
        x = np.maximum(0, x)  # ReLU
        x = x @ model["layer2.W"]
        x = np.maximum(0, x)
        x = x @ model["head.W"]
        return np.mean((x - target) ** 2)
    
    # Dummy data iterator
    def data_iter():
        return {"input": np.random.randn(32, 64) * 0.1}
    
    # Training config
    config = TrainingConfig(
        max_steps=500,
        lr=1e-3,
        warmup_steps=50,
        lr_schedule=LRSchedule.PHI_DECAY,
        log_interval=50
    )
    
    # Train
    trainer = PhiTrainer(model, config, loss_fn, data_iter)
    results = trainer.train()
    
    print(f"\nFinal loss: {results['final_loss']:.6f}")
    return results


if __name__ == "__main__":
    demo_training()
