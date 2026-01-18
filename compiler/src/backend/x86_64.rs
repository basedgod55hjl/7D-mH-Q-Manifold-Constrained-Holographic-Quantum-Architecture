use super::Backend;
use crate::ir::{IRBlock, IROpcode, Value};
use std::collections::HashSet;

pub struct X86Backend;

impl Backend for X86Backend {
    fn emit(&self, blocks: &[IRBlock]) -> Result<String, String> {
        let mut output = String::new();
        let mut variables = HashSet::new();
        let mut strings = Vec::new();
        // Register parameters for x86_64 calling convention: rdi, rsi, rdx, rcx, r8, r9
        let param_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"];
        let mut param_index = 0;

        // First pass: collect all variables and strings
        for block in blocks {
            for instr in &block.instructions {
                if let IROpcode::Store(var_name) = instr {
                    variables.insert(var_name.clone());
                }
                if let IROpcode::Push(Value::String(s)) = instr {
                    strings.push(s.clone());
                }
            }
        }

        // Generate data section for variables and strings
        if !variables.is_empty() || !strings.is_empty() {
            output.push_str("section .data\n");
            for var in &variables {
                output.push_str(&format!("{}: dq 0\n", var));
            }
            for (i, s) in strings.iter().enumerate() {
                output.push_str(&format!("str_{}: db '{}', 0\n", i, s.replace("'", "\\'")));
            }
            output.push_str("\n");
        }

        output.push_str("section .text\n");
        output.push_str("extern crystal_config_manifold\n");
        output.push_str("extern crystal_config_crystal\n");
        output.push_str("extern crystal_config_entropy\n");
        output.push_str("extern crystal_yield\n");
        output.push_str("extern crystal_project_7d\n");
        output.push_str("extern crystal_tensor_product_7d\n");
        output.push_str("extern crystal_superpose_7d\n");
        output.push_str("extern crystal_holo_fold_7d\n");
        output.push_str("extern crystal_string_concat\n");
        output.push_str("global _start\n\n");

        for block in blocks {
            output.push_str(&format!("{}:\n", block.name));
            for instr in &block.instructions {
                match instr {
                    IROpcode::Push(v) => match v {
                        Value::Int(i) => output.push_str(&format!("    push {}\n", i)),
                        Value::Float(f) => {
                            // Represent float bits as integer for push
                            let bits = f.to_bits();
                            output.push_str(&format!("    mov rax, {}\n", bits));
                            output.push_str(&format!("    push rax ; {}\n", f));
                        }
                        Value::Phi => {
                            let phi_bits = 1.618033988749895_f64.to_bits();
                            output.push_str(&format!("    mov rax, {}\n", phi_bits));
                            output.push_str("    push rax ; Φ\n");
                        }
                        Value::PhiInv => {
                            let phi_inv_bits = 0.618033988749895_f64.to_bits();
                            output.push_str(&format!("    mov rax, {}\n", phi_inv_bits));
                            output.push_str("    push rax ; Φ⁻¹\n");
                        }
                        Value::S2 => {
                            let s2_bits = 0.01_f64.to_bits();
                            output.push_str(&format!("    mov rax, {}\n", s2_bits));
                            output.push_str("    push rax ; S²\n");
                        }
                        Value::Lambda => output.push_str("    push 0 ; Lambda placeholder\n"),
                        Value::String(s) => {
                            // Find string index and push its address
                            if let Some(idx) = strings.iter().position(|str| str == s) {
                                output.push_str(&format!("    mov rax, str_{}\n", idx));
                                output.push_str("    push rax ; string pointer\n");
                            } else {
                                output.push_str("    push 0 ; string not found\n");
                            }
                        }
                        _ => output.push_str("    ; push complex value\n"),
                    },
                    IROpcode::Pop => output.push_str("    add rsp, 8\n"),
                    IROpcode::Dup => output.push_str("    pop rax\n    push rax\n    push rax\n"),
                    IROpcode::Store(name) => {
                        output.push_str(&format!("    pop rax\n    mov [{}], rax\n", name));
                    }
                    IROpcode::Load(name) => {
                        output.push_str(&format!("    mov rax, [{}]\n    push rax\n", name));
                    }
                    IROpcode::Add => {
                        output
                            .push_str("    pop rbx\n    pop rax\n    add rax, rbx\n    push rax\n");
                    }
                    IROpcode::Sub => {
                        output
                            .push_str("    pop rbx\n    pop rax\n    sub rax, rbx\n    push rax\n");
                    }
                    IROpcode::Mul => {
                        output.push_str(
                            "    pop rbx\n    pop rax\n    imul rax, rbx\n    push rax\n",
                        );
                    }
                    IROpcode::Div => {
                        output.push_str(
                            "    pop rbx\n    pop rax\n    cqo\n    idiv rbx\n    push rax\n",
                        );
                    }
                    IROpcode::Eq => {
                        output.push_str("    pop rbx\n    pop rax\n    cmp rax, rbx\n    sete al\n    movzx rax, al\n    push rax\n");
                    }
                    IROpcode::Ne => {
                        output.push_str("    pop rbx\n    pop rax\n    cmp rax, rbx\n    setne al\n    movzx rax, al\n    push rax\n");
                    }
                    IROpcode::Lt => {
                        output.push_str("    pop rbx\n    pop rax\n    cmp rax, rbx\n    setl al\n    movzx rax, al\n    push rax\n");
                    }
                    IROpcode::Gt => {
                        output.push_str("    pop rbx\n    pop rax\n    cmp rax, rbx\n    setg al\n    movzx rax, al\n    push rax\n");
                    }
                    IROpcode::Concat => {
                        output.push_str("    pop rsi ; string2\n    pop rdi ; string1\n    call crystal_string_concat\n    push rax\n");
                    }
                    IROpcode::Project7D => {
                        output.push_str(
                            "    pop rdi ; input\n    call crystal_project_7d\n    push rax\n",
                        );
                    }
                    IROpcode::ProjectTo => {
                        output.push_str(
                            "    pop rsi ; target\n    pop rdi ; input\n    call crystal_project_to_manifold\n    push rax\n",
                        );
                    }
                    IROpcode::TensorProduct => {
                        output.push_str("    pop rdx ; right\n    pop rcx ; left\n    call crystal_tensor_product_7d\n    push rax\n");
                    }
                    IROpcode::Superpose => {
                        output.push_str("    pop rdx ; right\n    pop rcx ; left\n    call crystal_superpose_7d\n    push rax\n");
                    }
                    IROpcode::HoloFold => {
                        output.push_str("    pop rdx ; pattern2\n    pop rcx ; pattern1\n    call crystal_holo_fold_7d\n    push rax\n");
                    }
                    IROpcode::Return => {
                        output.push_str("    ret\n");
                    }
                    IROpcode::Yield => {
                        output.push_str("    call crystal_yield\n");
                    }
                    IROpcode::Label(lbl) => {
                        output.push_str(&format!("{}:\n", lbl));
                    }
                    IROpcode::Jump(lbl) => {
                        output.push_str(&format!("    jmp {}\n", lbl));
                    }
                    IROpcode::JumpIfFalse(lbl) => {
                        output.push_str("    pop rax\n    test rax, rax\n");
                        output.push_str(&format!("    jz {}\n", lbl));
                    }
                    IROpcode::ConfigEntropy(name, mod_, pur) => {
                        // In real impl, we'd pass these args to the runtime
                        output.push_str(&format!("    ; ConfigEntropy: {}, mod={}, pur={}\n    call crystal_config_entropy\n", name, mod_, pur));
                    }
                    IROpcode::ConfigCrystal(name, _) => {
                        output.push_str(&format!(
                            "    ; ConfigCrystal: {}\n    call crystal_config_crystal\n",
                            name
                        ));
                    }
                    IROpcode::ConfigManifold(name, dim, curv) => {
                        output.push_str(&format!("    ; ConfigManifold: {}, dim={}, curv={}\n    call crystal_config_manifold\n", name, dim, curv));
                    }
                    IROpcode::Call(name, arg_count) => {
                        // Reset param index for safety/simplicity before call?
                        // For standard calling convention, args are already pushed or in regs.
                        // But our IR pushed generic values.
                        // The implementation for individual functions (like crystal_*)
                        // expects args in specific registers/stack.

                        // Hack for now: generic call assumes stack based or we assume regs handled by previous Param ops?
                        // Actually, IR puts args on stack (Push).
                        // We need to pop them into registers if using System V AMD64 ABI (rdi, rsi, rdx...)
                        // OR just call and let the callee handle stack (cdecl-ish but 64bit uses regs).

                        // Current `crystal_...` stubs probably expect registers?
                        // Let's implement a simple "Pop args to registers" loop here for the first 6 args.

                        let regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"];
                        let num_regs = regs.len().min(*arg_count);

                        // Args are on stack in order: Arg1, Arg2... (pushed in loop)
                        // Wait, `for arg in args { generate(arg) }` pushes Arg1, then Arg2.
                        // Stack: [Arg1, Arg2]. Top is Arg2.
                        // So popping gives Arg(N), Arg(N-1)...
                        // We need to fill registers in reverse?
                        // No, Arg1 goes to RDI. Arg2 to RSI.
                        // Stack has Arg2 at top.

                        // We need to pop into temporary storage or registers in reverse order.
                        // Reversing the mapping:
                        // Top (ArgN) -> Reg(N-1)
                        // ...
                        // Bottom (Arg1) -> Reg(0) = RDI

                        for i in (0..num_regs).rev() {
                            // This assumes arg_count matches what was pushed.
                            output.push_str(&format!("    pop {} ; Arg {}\n", regs[i], i));
                        }

                        // If more than 6 args, they remain on stack (which is correct for x64 ABI overflow args).
                        // However, stack might be misaligned?
                        // We need 16-byte alignment before call.
                        // Ignoring alignment for this bootleg kernel for now (might crash in SSE instructions).

                        output.push_str(&format!("    call {}\n", name));
                        output.push_str("    push rax ; push return value\n");
                    }
                    IROpcode::Param(name) => {
                        if param_index < param_regs.len() {
                            output.push_str(&format!(
                                "    push {} ; Param: {}\n",
                                param_regs[param_index], name
                            ));
                            param_index += 1;
                        } else {
                            output.push_str(&format!(
                                "    ; Param: {} (Stack handling not impl)\n",
                                name
                            ));
                        }
                    }
                }
            }
            output.push_str("\n");
            // Reset param index for next function/block?
            // Simplified: Assuming linear block flow for now, but really this should reset per function.
            // For now, let's keep it simple as we primarily have one main flow.
        }

        Ok(output)
    }
}
