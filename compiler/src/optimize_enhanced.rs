// File: compiler/src/optimize_enhanced.rs
// Enhanced 7D Crystal Optimization Passes
// Includes Φ-ratio optimization, manifold fusion, and dead code elimination
// Discovered by Sir Charles Spikes, December 24, 2025
// Developed by Sir Charles Spikes

use crate::ir::{IRBlock7D as IRBlock, IR7D};
// use crate::parser::ASTNode;
use std::collections::{HashMap, HashSet};

// ============================================================================
// Constants
// ============================================================================

pub const PHI: f64 = 1.618033988749895;
pub const PHI_INV: f64 = 0.618033988749895;
pub const S2_STABILITY: f64 = 0.01;

// ============================================================================
// Optimization Pass Trait
// ============================================================================

pub trait OptimizationPass {
    fn name(&self) -> &'static str;
    fn run(&self, blocks: &mut Vec<IRBlock>) -> OptimizationResult;
}

#[derive(Debug, Default)]
pub struct OptimizationResult {
    pub instructions_removed: usize,
    pub instructions_replaced: usize,
    pub blocks_merged: usize,
    pub phi_optimizations: usize,
}

impl OptimizationResult {
    pub fn merge(&mut self, other: &Self) {
        self.instructions_removed += other.instructions_removed;
        self.instructions_replaced += other.instructions_replaced;
        self.blocks_merged += other.blocks_merged;
        self.phi_optimizations += other.phi_optimizations;
    }
}

// ============================================================================
// Φ-Ratio Constant Folding Pass
// ============================================================================

/// Folds Φ-related constant expressions at compile time
pub struct PhiConstantFolding;

impl OptimizationPass for PhiConstantFolding {
    fn name(&self) -> &'static str {
        "Φ-Constant Folding"
    }

    fn run(&self, blocks: &mut Vec<IRBlock>) -> OptimizationResult {
        let mut result = OptimizationResult::default();

        for block in blocks.iter_mut() {
            let mut i = 0;
            while i < block.instructions.len() {
                if let Some(folded) = self.try_fold(&block.instructions, i) {
                    block.instructions[i] = folded;
                    result.phi_optimizations += 1;
                }
                i += 1;
            }
        }

        result
    }
}

impl PhiConstantFolding {
    fn try_fold(&self, instructions: &[IR7D], idx: usize) -> Option<IR7D> {
        match &instructions[idx] {
            // Fold PHI * PHI_INV = 1.0
            IR7D::MulFloat(a, b) if self.is_phi(*a) && self.is_phi_inv(*b) => {
                Some(IR7D::PushFloat(1.0))
            }
            // Fold PHI + PHI_INV = PHI^2 - 1 = PHI
            IR7D::AddFloat(a, b) if self.is_phi(*a) && self.is_phi_inv(*b) => {
                Some(IR7D::PushFloat(PHI + PHI_INV)) // = 2.236...
            }
            // Fold PHI * PHI = PHI^2
            IR7D::MulFloat(a, b) if self.is_phi(*a) && self.is_phi(*b) => {
                Some(IR7D::PushFloat(PHI * PHI))
            }
            _ => None,
        }
    }

    fn is_phi(&self, val: f64) -> bool {
        (val - PHI).abs() < 1e-10
    }

    fn is_phi_inv(&self, val: f64) -> bool {
        (val - PHI_INV).abs() < 1e-10
    }
}

// ============================================================================
// Manifold Operation Fusion Pass
// ============================================================================

/// Fuses consecutive manifold operations for better performance
pub struct ManifoldFusion;

impl OptimizationPass for ManifoldFusion {
    fn name(&self) -> &'static str {
        "Manifold Operation Fusion"
    }

    fn run(&self, blocks: &mut Vec<IRBlock>) -> OptimizationResult {
        let mut result = OptimizationResult::default();

        for block in blocks.iter_mut() {
            let mut fused = Vec::new();
            let mut i = 0;

            while i < block.instructions.len() {
                // Look for fusible patterns
                if i + 1 < block.instructions.len() {
                    if let Some(fused_op) =
                        self.try_fuse(&block.instructions[i], &block.instructions[i + 1])
                    {
                        fused.push(fused_op);
                        result.instructions_replaced += 1;
                        i += 2;
                        continue;
                    }
                }

                fused.push(block.instructions[i].clone());
                i += 1;
            }

            block.instructions = fused;
        }

        result
    }
}

impl ManifoldFusion {
    fn try_fuse(&self, a: &IR7D, b: &IR7D) -> Option<IR7D> {
        match (a, b) {
            // Fuse consecutive projections
            (
                IR7D::ManifoldProject { curvature: c1, .. },
                IR7D::ManifoldProject { curvature: c2, .. },
            ) => {
                // Combined projection with averaged curvature
                Some(IR7D::ManifoldProject {
                    input_reg: 0,
                    output_reg: 0,
                    curvature: (*c1 + *c2) / 2.0,
                })
            }
            // Fuse holographic folds
            (
                IR7D::HolographicFold { phases: p1, .. },
                IR7D::HolographicFold { phases: p2, .. },
            ) => Some(IR7D::HolographicFold {
                p1_reg: 0,
                p2_reg: 0,
                out_reg: 0,
                phases: p1 + p2,
            }),
            _ => None,
        }
    }
}

// ============================================================================
// Dead Code Elimination Pass
// ============================================================================

/// Removes unreachable and unused code
pub struct DeadCodeElimination;

impl OptimizationPass for DeadCodeElimination {
    fn name(&self) -> &'static str {
        "Dead Code Elimination"
    }

    fn run(&self, blocks: &mut Vec<IRBlock>) -> OptimizationResult {
        let mut result = OptimizationResult::default();

        // Find all used registers
        let used_regs = self.find_used_registers(blocks);

        // Remove instructions writing to unused registers
        for block in blocks.iter_mut() {
            let original_len = block.instructions.len();
            block
                .instructions
                .retain(|inst| self.is_instruction_live(inst, &used_regs));
            result.instructions_removed += original_len - block.instructions.len();
        }

        // Remove empty blocks
        let original_blocks = blocks.len();
        blocks.retain(|b| !b.instructions.is_empty());
        result.blocks_merged += original_blocks - blocks.len();

        result
    }
}

impl DeadCodeElimination {
    fn find_used_registers(&self, blocks: &[IRBlock]) -> HashSet<usize> {
        let mut used = HashSet::new();

        for block in blocks {
            for inst in &block.instructions {
                // Add all read registers
                match inst {
                    IR7D::LoadReg(r) => {
                        used.insert(*r);
                    }
                    IR7D::AddReg(a, b) | IR7D::MulReg(a, b) => {
                        used.insert(*a);
                        used.insert(*b);
                    }
                    IR7D::ManifoldProject { input_reg, .. } => {
                        used.insert(*input_reg);
                    }
                    IR7D::HolographicFold { p1_reg, p2_reg, .. } => {
                        used.insert(*p1_reg);
                        used.insert(*p2_reg);
                    }
                    _ => {}
                }
            }
        }

        used
    }

    fn is_instruction_live(&self, inst: &IR7D, used: &HashSet<usize>) -> bool {
        match inst {
            // Side-effect instructions are always live
            IR7D::Return | IR7D::Call(_) | IR7D::Print => true,

            // Store instructions - check if target is used
            IR7D::StoreReg(r) => used.contains(r),
            IR7D::ManifoldProject { output_reg, .. } => used.contains(output_reg),
            IR7D::HolographicFold { out_reg, .. } => used.contains(out_reg),

            // Push instructions are live if consumed
            IR7D::PushInt(_) | IR7D::PushFloat(_) => true,

            // Default to keeping
            _ => true,
        }
    }
}

// ============================================================================
// Common Subexpression Elimination
// ============================================================================

/// Eliminates redundant computations
pub struct CSE;

impl OptimizationPass for CSE {
    fn name(&self) -> &'static str {
        "Common Subexpression Elimination"
    }

    fn run(&self, blocks: &mut Vec<IRBlock>) -> OptimizationResult {
        let mut result = OptimizationResult::default();

        for block in blocks.iter_mut() {
            let mut seen: HashMap<String, usize> = HashMap::new();
            let mut replacements: HashMap<usize, usize> = HashMap::new();

            for (i, inst) in block.instructions.iter().enumerate() {
                let key = self.instruction_key(inst);

                if let Some(&prev_idx) = seen.get(&key) {
                    // Mark for replacement
                    replacements.insert(i, prev_idx);
                    result.instructions_replaced += 1;
                } else if self.is_pure(inst) {
                    seen.insert(key, i);
                }
            }

            // Apply replacements (simplified - full impl would rewrite references)
            for (idx, _replacement) in replacements {
                block.instructions[idx] = IR7D::Nop;
            }
        }

        result
    }
}

impl CSE {
    fn instruction_key(&self, inst: &IR7D) -> String {
        format!("{:?}", inst)
    }

    fn is_pure(&self, inst: &IR7D) -> bool {
        match inst {
            IR7D::Call(_) | IR7D::Print | IR7D::Return => false,
            _ => true,
        }
    }
}

// ============================================================================
// Strength Reduction Pass
// ============================================================================

/// Replaces expensive operations with cheaper equivalents
pub struct StrengthReduction;

impl OptimizationPass for StrengthReduction {
    fn name(&self) -> &'static str {
        "Strength Reduction"
    }

    fn run(&self, blocks: &mut Vec<IRBlock>) -> OptimizationResult {
        let mut result = OptimizationResult::default();

        for block in blocks.iter_mut() {
            for inst in block.instructions.iter_mut() {
                if let Some(reduced) = self.reduce(inst) {
                    *inst = reduced;
                    result.instructions_replaced += 1;
                }
            }
        }

        result
    }
}

impl StrengthReduction {
    fn reduce(&self, inst: &IR7D) -> Option<IR7D> {
        match inst {
            // x * 2 -> x + x (faster on some architectures)
            IR7D::MulFloat(x, 2.0) | IR7D::MulFloat(2.0, x) => Some(IR7D::AddFloat(*x, *x)),
            // x * 0.5 -> x >> 1 (for integers, simplified here)
            IR7D::MulFloat(x, 0.5) => Some(IR7D::DivFloat(*x, 2.0)),
            // x * PHI approximation using adds/shifts
            IR7D::MulFloat(x, p) if (*p - PHI).abs() < 1e-10 => {
                // PHI ≈ 1.618 ≈ 1 + 0.5 + 0.125 (for integer approximation)
                // Keep as-is for floats, but mark for SIMD optimization
                None
            }
            _ => None,
        }
    }
}

// ============================================================================
// Loop Optimization Pass
// ============================================================================

/// Optimizes loops with manifold invariant hoisting
pub struct LoopOptimization;

impl OptimizationPass for LoopOptimization {
    fn name(&self) -> &'static str {
        "Loop Optimization"
    }

    fn run(&self, blocks: &mut Vec<IRBlock>) -> OptimizationResult {
        let mut result = OptimizationResult::default();

        // Find loop headers
        let loop_headers = self.find_loop_headers(blocks);

        for header_idx in loop_headers {
            if let Some((hoisted, remaining)) = self.hoist_invariants(&blocks[header_idx]) {
                // Insert hoisted instructions before loop
                if header_idx > 0 {
                    blocks[header_idx - 1].instructions.extend(hoisted);
                }
                blocks[header_idx].instructions = remaining;
                result.instructions_removed += 1;
            }
        }

        result
    }
}

impl LoopOptimization {
    fn find_loop_headers(&self, blocks: &[IRBlock]) -> Vec<usize> {
        // Simplified - real impl would analyze control flow
        blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.name.starts_with("loop"))
            .map(|(i, _)| i)
            .collect()
    }

    fn hoist_invariants(&self, block: &IRBlock) -> Option<(Vec<IR7D>, Vec<IR7D>)> {
        let mut hoisted = Vec::new();
        let mut remaining = Vec::new();

        for inst in &block.instructions {
            if self.is_loop_invariant(inst) {
                hoisted.push(inst.clone());
            } else {
                remaining.push(inst.clone());
            }
        }

        if hoisted.is_empty() {
            None
        } else {
            Some((hoisted, remaining))
        }
    }

    fn is_loop_invariant(&self, inst: &IR7D) -> bool {
        match inst {
            IR7D::PushFloat(f) if (*f - PHI).abs() < 1e-10 => true,
            IR7D::PushFloat(f) if (*f - PHI_INV).abs() < 1e-10 => true,
            IR7D::PushFloat(f) if (*f - S2_STABILITY).abs() < 1e-10 => true,
            _ => false,
        }
    }
}

// ============================================================================
// Optimization Pipeline
// ============================================================================

pub struct OptimizationPipeline {
    passes: Vec<Box<dyn OptimizationPass>>,
    max_iterations: usize,
}

impl Default for OptimizationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPipeline {
    pub fn new() -> Self {
        Self {
            passes: vec![
                Box::new(PhiConstantFolding),
                Box::new(DeadCodeElimination),
                Box::new(CSE),
                Box::new(StrengthReduction),
                Box::new(ManifoldFusion),
                Box::new(LoopOptimization),
            ],
            max_iterations: 5,
        }
    }

    pub fn optimize(&self, blocks: &mut Vec<IRBlock>) -> OptimizationResult {
        let mut total_result = OptimizationResult::default();

        for iteration in 0..self.max_iterations {
            let mut changed = false;

            for pass in &self.passes {
                let result = pass.run(blocks);

                if result.instructions_removed > 0
                    || result.instructions_replaced > 0
                    || result.blocks_merged > 0
                {
                    changed = true;
                    println!("  [{}] Pass '{}' made changes", iteration, pass.name());
                }

                total_result.merge(&result);
            }

            if !changed {
                println!("  Optimization converged at iteration {}", iteration);
                break;
            }
        }

        total_result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_constant_folding() {
        let pass = PhiConstantFolding;
        let mut blocks = vec![IRBlock {
            label: "test".to_string(),
            instructions: vec![IR7D::MulFloat(PHI, PHI_INV)],
        }];

        let result = pass.run(&mut blocks);
        assert!(result.phi_optimizations > 0);
    }
}
