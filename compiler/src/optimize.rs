use crate::ir::{IRBlock7D, IR7D};
use std::collections::{HashMap, HashSet};

/// Advanced optimization passes for 7D Crystal
pub struct Optimizer7D {
    optimization_level: OptimizationLevel,
}

#[derive(Clone, Copy, PartialEq)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
}

impl Optimizer7D {
    pub fn new(level: OptimizationLevel) -> Self {
        Self {
            optimization_level: level,
        }
    }

    /// Run all optimization passes
    pub fn optimize(&self, blocks: &mut Vec<IRBlock7D>) {
        match self.optimization_level {
            OptimizationLevel::None => return,
            OptimizationLevel::Basic => {
                for block in blocks.iter_mut() {
                    self.optimize_block_basic(block);
                }
            }
            OptimizationLevel::Aggressive => {
                // Inter-procedural optimizations first
                self.optimize_interprocedural(blocks);

                // Then per-block optimizations
                for block in blocks.iter_mut() {
                    self.optimize_block_aggressive(block);
                }
            }
        }
    }

    fn optimize_block_basic(&self, block: &mut IRBlock7D) {
        // Basic optimizations that can be done locally
        self.eliminate_dead_code(block);
        self.constant_fold(block);
        self.simplify_phi_operations(block);
        self.remove_redundant_moves(block);
    }

    fn optimize_block_aggressive(&self, block: &mut IRBlock7D) {
        // Aggressive optimizations
        self.optimize_block_basic(block);
        self.loop_invariant_code_motion(block);
        self.strength_reduction(block);
        self.common_subexpression_elimination(block);
        self.fuse_kernels(block);
    }

    fn optimize_interprocedural(&self, blocks: &mut Vec<IRBlock7D>) {
        // Inter-procedural optimizations
        self.function_inlining(blocks);
        self.constant_propagation_interproc(blocks);
    }

    // =========================================================================
    // BASIC OPTIMIZATION PASSES
    // =========================================================================

    fn eliminate_dead_code(&self, block: &mut IRBlock7D) {
        let mut used_regs = HashSet::new();
        let _defined_regs: HashSet<usize> = HashSet::new();

        // First pass: find all used registers
        for instr in block.instructions.iter() {
            match instr {
                IR7D::Store(reg, _) => {
                    used_regs.insert(*reg);
                }
                IR7D::Load(reg, _) => {
                    used_regs.insert(*reg);
                }
                IR7D::Add => {} // Uses implicit registers
                IR7D::Sub => {}
                IR7D::Mul => {}
                IR7D::Div => {}
                IR7D::ManifoldProject {
                    input_reg,
                    output_reg,
                    ..
                } => {
                    used_regs.insert(*input_reg);
                    used_regs.insert(*output_reg);
                }
                IR7D::HolographicFold {
                    p1_reg,
                    p2_reg,
                    out_reg,
                    ..
                } => {
                    used_regs.insert(*p1_reg);
                    used_regs.insert(*p2_reg);
                    used_regs.insert(*out_reg);
                }
                _ => {}
            }
        }

        // Second pass: remove stores to unused registers
        block.instructions.retain(|instr| match instr {
            IR7D::Store(reg, _) => used_regs.contains(reg),
            _ => true,
        });
    }

    fn constant_fold(&self, block: &mut IRBlock7D) {
        let mut constants = HashMap::new();
        let mut i = 0;

        while i < block.instructions.len() {
            match &block.instructions[i] {
                IR7D::PushInt(value) => {
                    // Track constant value
                    let reg = self.get_next_reg(&block.instructions[i..]);
                    constants.insert(reg, *value as f64);
                    i += 1;
                }
                IR7D::PushFloat(value) => {
                    let reg = self.get_next_reg(&block.instructions[i..]);
                    constants.insert(reg, *value);
                    i += 1;
                }
                IR7D::Add if i >= 2 => {
                    // Check if we have constant addition
                    if let (Some(a), Some(b)) =
                        self.get_last_two_constants(&constants, &block.instructions[..i])
                    {
                        // Replace with constant result
                        let result_reg = self.get_next_reg(&block.instructions[i..]);
                        constants.insert(result_reg, a + b);

                        // Remove the push instructions and add
                        let remove_count =
                            self.remove_constant_operation(&mut block.instructions, i);
                        i -= remove_count - 1; // Adjust for removed instructions
                    } else {
                        i += 1;
                    }
                }
                IR7D::Sub if i >= 2 => {
                    if let (Some(a), Some(b)) =
                        self.get_last_two_constants(&constants, &block.instructions[..i])
                    {
                        let result_reg = self.get_next_reg(&block.instructions[i..]);
                        constants.insert(result_reg, a - b);
                        let remove_count =
                            self.remove_constant_operation(&mut block.instructions, i);
                        i -= remove_count - 1;
                    } else {
                        i += 1;
                    }
                }
                IR7D::Mul if i >= 2 => {
                    if let (Some(a), Some(b)) =
                        self.get_last_two_constants(&constants, &block.instructions[..i])
                    {
                        let result_reg = self.get_next_reg(&block.instructions[i..]);
                        constants.insert(result_reg, a * b);
                        let remove_count =
                            self.remove_constant_operation(&mut block.instructions, i);
                        i -= remove_count - 1;
                    } else {
                        i += 1;
                    }
                }
                IR7D::Div if i >= 2 => {
                    if let (Some(a), Some(b)) =
                        self.get_last_two_constants(&constants, &block.instructions[..i])
                    {
                        if b != 0.0 {
                            let result_reg = self.get_next_reg(&block.instructions[i..]);
                            constants.insert(result_reg, a / b);
                            let remove_count =
                                self.remove_constant_operation(&mut block.instructions, i);
                            i -= remove_count - 1;
                        } else {
                            i += 1;
                        }
                    } else {
                        i += 1;
                    }
                }
                _ => {
                    i += 1;
                }
            }
        }
    }

    fn simplify_phi_operations(&self, block: &mut IRBlock7D) {
        // Simplify Φ-related constant operations
        // Φ + Φ = 2Φ, Φ * Φ = Φ², etc.
        let phi = 1.618033988749895;
        let phi_inv = 0.618033988749895;

        let mut i = 0;
        while i < block.instructions.len() {
            match &block.instructions[i] {
                IR7D::PushFloat(val) if (*val - phi).abs() < 1e-10 => {
                    // This is Φ, look for operations with it
                    if i + 2 < block.instructions.len() {
                        match (&block.instructions[i + 1], &block.instructions[i + 2]) {
                            (IR7D::PushFloat(val2), IR7D::Add) if (*val2 - phi).abs() < 1e-10 => {
                                // Φ + Φ = 2Φ
                                block
                                    .instructions
                                    .splice(i..i + 3, vec![IR7D::PushFloat(2.0 * phi)]);
                                continue;
                            }
                            (IR7D::PushFloat(val2), IR7D::Mul) if (*val2 - phi).abs() < 1e-10 => {
                                // Φ * Φ = Φ²
                                block
                                    .instructions
                                    .splice(i..i + 3, vec![IR7D::PushFloat(phi * phi)]);
                                continue;
                            }
                            (IR7D::PushFloat(val2), IR7D::Sub)
                                if (*val2 - phi_inv).abs() < 1e-10 =>
                            {
                                // Φ - Φ⁻¹ = (Φ² - 1)/Φ
                                let result = (phi * phi - 1.0) / phi;
                                block
                                    .instructions
                                    .splice(i..i + 3, vec![IR7D::PushFloat(result)]);
                                continue;
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }

    fn remove_redundant_moves(&self, block: &mut IRBlock7D) {
        let mut i = 0;
        while i + 1 < block.instructions.len() {
            match (&block.instructions[i], &block.instructions[i + 1]) {
                (IR7D::Store(reg1, offset1), IR7D::Load(reg2, offset2))
                    if reg1 == reg2 && offset1 == offset2 =>
                {
                    // Store followed by load of same location - can be removed
                    // if the load result is not used elsewhere
                    block.instructions.remove(i + 1);
                    // Don't increment i, check the next instruction
                }
                _ => {
                    i += 1;
                }
            }
        }
    }

    // =========================================================================
    // AGGRESSIVE OPTIMIZATION PASSES
    // =========================================================================

    fn loop_invariant_code_motion(&self, block: &mut IRBlock7D) {
        // Move loop-invariant computations outside loops
        // This is a simplified version - full implementation would need loop detection
        let mut invariants = Vec::new();
        let mut variant_regs = HashSet::new();

        // Find registers that are modified in loops (simplified)
        for instr in block.instructions.iter() {
            match instr {
                IR7D::Store(reg, _) => {
                    variant_regs.insert(*reg);
                }
                IR7D::ManifoldProject { output_reg, .. } => {
                    variant_regs.insert(*output_reg);
                }
                IR7D::HolographicFold { out_reg, .. } => {
                    variant_regs.insert(*out_reg);
                }
                _ => {}
            }
        }

        // Find computations that don't depend on variant registers
        let mut i = 0;
        while i < block.instructions.len() {
            match &block.instructions[i] {
                IR7D::PushInt(_) | IR7D::PushFloat(_) => {
                    // Check if this constant is used by variant operations
                    let uses_variant =
                        self.check_uses_variant_reg(&block.instructions, i, &variant_regs);
                    if !uses_variant {
                        invariants.push(block.instructions[i].clone());
                        block.instructions.remove(i);
                        continue;
                    }
                }
                _ => {}
            }
            i += 1;
        }

        // Prepend invariants to block
        for invariant in invariants.into_iter().rev() {
            block.instructions.insert(0, invariant);
        }
    }

    fn strength_reduction(&self, _block: &mut IRBlock7D) {
        // Replace expensive operations with cheaper equivalents
    }

    fn common_subexpression_elimination(&self, block: &mut IRBlock7D) {
        let mut seen_expressions = HashMap::new();
        let mut _reg_map = HashMap::new();

        let mut i = 0;
        while i < block.instructions.len() {
            match &block.instructions[i] {
                IR7D::Add | IR7D::Sub | IR7D::Mul | IR7D::Div => {
                    // Check if we have the same operation on same operands
                    if let Some(existing_reg) =
                        self.find_common_subexpr(&block.instructions, i, &seen_expressions)
                    {
                        // Replace with move from existing result
                        let result_reg = self.get_result_reg(&block.instructions, i);
                        block.instructions[i] = IR7D::Load(existing_reg, 0); // Placeholder offset
                        _reg_map.insert(result_reg, existing_reg);
                    } else {
                        // Record this expression
                        let expr_key = self.get_expression_key(&block.instructions, i);
                        let result_reg = self.get_result_reg(&block.instructions, i);
                        seen_expressions.insert(expr_key, result_reg);
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }

    fn fuse_kernels(&self, block: &mut IRBlock7D) {
        let mut optimized = Vec::new();
        let mut i = 0;

        while i < block.instructions.len() {
            let mut fusable = Vec::new();

            // Look for consecutive fusable operations
            while i < block.instructions.len() {
                let instr = &block.instructions[i];
                if self.is_fusable(instr) {
                    fusable.push(instr.clone());
                    i += 1;
                } else {
                    break;
                }
            }

            if fusable.len() > 1 {
                // Create a FusedOp
                let name = format!("kernel_fused_{}", optimized.len());
                optimized.push(IR7D::FusedOp {
                    name,
                    ops: fusable,
                    reg_map: HashMap::new(), // Placeholder for register tracking
                });
            } else if fusable.len() == 1 {
                optimized.push(fusable[0].clone());
            } else if i < block.instructions.len() {
                optimized.push(block.instructions[i].clone());
                i += 1;
            }
        }

        block.instructions = optimized;
    }

    fn is_fusable(&self, instr: &IR7D) -> bool {
        match instr {
            IR7D::ManifoldProject { .. } => true,
            IR7D::HolographicFold { .. } => true,
            IR7D::Add | IR7D::Sub | IR7D::Mul | IR7D::Div => true,
            _ => false,
        }
    }

    // =========================================================================
    // INTER-PROCEDURAL OPTIMIZATION PASSES
    // =========================================================================

    fn function_inlining(&self, blocks: &mut Vec<IRBlock7D>) {
        // Inline small functions
        let mut small_functions = Vec::new();

        // Find functions smaller than threshold
        for block in blocks.iter() {
            if block.instructions.len() < 10 && block.name != "_start" {
                small_functions.push(block.name.clone());
            }
        }

        // Inline small functions (simplified - would need call graph analysis)
        // This is a placeholder for full function inlining
    }

    fn constant_propagation_interproc(&self, blocks: &mut Vec<IRBlock7D>) {
        // Propagate constants across function boundaries
        let mut _global_constants: HashMap<String, f64> = HashMap::new();

        // Find constant assignments that can be propagated
        for block in blocks.iter() {
            for instr in block.instructions.iter() {
                match instr {
                    IR7D::PushFloat(_val) => {
                        // Check if this is assigned to a global or parameter
                        // This is simplified - full implementation would track data flow
                    }
                    _ => {}
                }
            }
        }
    }

    // =========================================================================
    // HELPER FUNCTIONS
    // =========================================================================

    fn get_next_reg(&self, _instructions: &[IR7D]) -> usize {
        // Simplified register allocation
        static mut REG_COUNTER: usize = 0;
        unsafe {
            REG_COUNTER += 1;
            REG_COUNTER
        }
    }

    fn get_last_two_constants(
        &self,
        constants: &HashMap<usize, f64>,
        instructions: &[IR7D],
    ) -> (Option<f64>, Option<f64>) {
        // Find the last two constant pushes
        let mut found = Vec::new();
        for (i, instr) in instructions.iter().enumerate().rev() {
            match instr {
                IR7D::PushInt(_) => {
                    if let Some(reg) = self.find_reg_for_instruction(instructions, i) {
                        if let Some(&const_val) = constants.get(&reg) {
                            found.push(const_val);
                        }
                    }
                    if found.len() >= 2 {
                        break;
                    }
                }
                IR7D::PushFloat(val) => {
                    found.push(*val);
                    if found.len() >= 2 {
                        break;
                    }
                }
                _ => {}
            }
        }

        match found.len() {
            2 => (Some(found[1]), Some(found[0])),
            1 => (Some(found[0]), None),
            _ => (None, None),
        }
    }

    fn remove_constant_operation(&self, instructions: &mut Vec<IR7D>, op_index: usize) -> usize {
        // Remove the two push instructions and the operation
        let mut removed = 0;
        let mut to_remove = vec![op_index - 2, op_index - 1, op_index];

        // Remove in reverse order to maintain indices
        to_remove.sort_by(|a, b| b.cmp(a));
        for idx in to_remove {
            if idx < instructions.len() {
                instructions.remove(idx);
                removed += 1;
            }
        }
        removed
    }

    fn check_uses_variant_reg(
        &self,
        instructions: &[IR7D],
        start_idx: usize,
        variant_regs: &HashSet<usize>,
    ) -> bool {
        // Check if a constant is used by operations on variant registers
        let const_reg = self.find_reg_for_instruction(instructions, start_idx);

        if let Some(reg) = const_reg {
            for i in start_idx..instructions.len() {
                match &instructions[i] {
                    IR7D::Store(vreg, _) if variant_regs.contains(vreg) => {
                        // Check if this store uses our constant
                        if self.instruction_uses_reg(&instructions[i - 1], reg) {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
        }
        false
    }

    fn find_reg_for_instruction(&self, _instructions: &[IR7D], idx: usize) -> Option<usize> {
        // Simplified - would need proper register tracking
        Some(idx % 10) // Placeholder
    }

    fn instruction_uses_reg(&self, instr: &IR7D, reg: usize) -> bool {
        match instr {
            IR7D::Load(r, _) => *r == reg,
            IR7D::Store(r, _) => *r == reg,
            _ => false,
        }
    }

    fn find_common_subexpr(
        &self,
        instructions: &[IR7D],
        idx: usize,
        seen: &HashMap<String, usize>,
    ) -> Option<usize> {
        let expr_key = self.get_expression_key(instructions, idx);
        seen.get(&expr_key).copied()
    }

    fn get_expression_key(&self, instructions: &[IR7D], idx: usize) -> String {
        // Create a key representing the expression
        if idx >= 2 {
            match (
                &instructions[idx - 2],
                &instructions[idx - 1],
                &instructions[idx],
            ) {
                (IR7D::PushInt(a), IR7D::PushInt(b), IR7D::Add) => format!("{}+{}", a, b),
                (IR7D::PushInt(a), IR7D::PushInt(b), IR7D::Mul) => format!("{}*{}", a, b),
                _ => format!("expr_{}", idx),
            }
        } else {
            format!("expr_{}", idx)
        }
    }

    fn get_result_reg(&self, _instructions: &[IR7D], idx: usize) -> usize {
        // Simplified - would need proper register tracking
        idx % 10
    }
}
