// 7D Crystal Runtime - C FFI Bridge
// Exposes the 7D runtime to other languages (C/C++, Python, etc.)
// Discovered by Sir Charles Spikes, December 24, 2025

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::ptr;

use crate::allocator::ManifoldAllocator;
use crate::executor::{ExecutionContext, ExecutionResult, ExecutionTarget, HardwareExecutor};
use crate::ir::IR7D;
use crate::quantum::{Basis, QuantumStateManager, StateID};

/// Opaque handle to a HardwareExecutor
pub struct SevenDExecutorHandle(Box<HardwareExecutor>);

/// Opaque handle to an ExecutionContext
pub struct SevenDContextHandle(Box<ExecutionContext>);

/// Create a new 7D Hardware Executor
#[unsafe(no_mangle)]
pub extern "C" fn sevend_executor_create() -> *mut SevenDExecutorHandle {
    let executor = HardwareExecutor::new();
    Box::into_raw(Box::new(SevenDExecutorHandle(Box::new(executor))))
}

/// Destroy a 7D Hardware Executor
#[unsafe(no_mangle)]
pub extern "C" fn sevend_executor_destroy(handle: *mut SevenDExecutorHandle) {
    if !handle.is_null() {
        unsafe { Box::from_raw(handle) };
    }
}

/// Create a new 7D Execution Context
#[unsafe(no_mangle)]
pub extern "C" fn sevend_context_create(heap_size: usize) -> *mut SevenDContextHandle {
    match ExecutionContext::new(heap_size) {
        Ok(ctx) => Box::into_raw(Box::new(SevenDContextHandle(Box::new(ctx)))),
        Err(_) => ptr::null_mut(),
    }
}

/// Destroy a 7D Execution Context
#[unsafe(no_mangle)]
pub extern "C" fn sevend_context_destroy(handle: *mut SevenDContextHandle) {
    if !handle.is_null() {
        unsafe { Box::from_raw(handle) };
    }
}

/// Register a function in the context (raw IR bytes)
/// This is a simplified version; real implementation would parse IR
#[unsafe(no_mangle)]
pub extern "C" fn sevend_context_register_function(
    handle: *mut SevenDContextHandle,
    name: *const c_char,
    _ir_data: *const c_void,
    _ir_len: usize,
) -> bool {
    if handle.is_null() || name.is_null() {
        return false;
    }

    let c_name = unsafe { CStr::from_ptr(name) };
    let func_name = match c_name.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return false,
    };

    let ctx = unsafe { &mut *((*handle).0) };

    // For now, register a dummy return-0 function
    ctx.register_function(func_name, vec![IR7D::PushInt(0), IR7D::Return]);

    true
}

/// Execute a function from the context
#[unsafe(no_mangle)]
pub extern "C" fn sevend_execute(
    executor_handle: *mut SevenDExecutorHandle,
    context_handle: *mut SevenDContextHandle,
    entry_point: *const c_char,
) -> c_int {
    if executor_handle.is_null() || context_handle.is_null() || entry_point.is_null() {
        return -1;
    }

    let executor = unsafe { &mut *((*executor_handle).0) };
    let ctx = unsafe { &mut *((*context_handle).0) };

    let c_name = unsafe { CStr::from_ptr(entry_point) };
    let func_name = match c_name.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    match executor.execute(ctx, func_name) {
        Ok(result) => result.return_value,
        Err(_) => -1,
    }
}

/// Quantum: Create a new state
#[unsafe(no_mangle)]
pub extern "C" fn sevend_quantum_create_state(
    handle: *mut SevenDContextHandle,
    dimension: usize,
) -> u64 {
    if handle.is_null() {
        return 0;
    }
    let ctx = unsafe { &mut *((*handle).0) };
    ctx.create_quantum_state(dimension).into()
}

/// Quantum: Measure a state
#[unsafe(no_mangle)]
pub extern "C" fn sevend_quantum_measure(
    handle: *mut SevenDContextHandle,
    state_id: u64,
) -> c_double {
    if handle.is_null() {
        return -1.0;
    }
    let ctx = unsafe { &mut *((*handle).0) };

    // Convert u64 to state ID (assuming it matches internal u64 representation)
    // In real implementation, StateID would be properly exposed
    let id = unsafe { std::mem::transmute(state_id) };

    match ctx.quantum.measure(id, Basis::Computational) {
        Ok(res) => res as c_double,
        Err(_) => -1.0,
    }
}

/// Helper to expose StateID into u64
impl From<StateID> for u64 {
    fn from(id: StateID) -> Self {
        unsafe { std::mem::transmute(id) }
    }
}

/// Get hardware capabilities summary
#[unsafe(no_mangle)]
pub extern "C" fn sevend_get_capabilities(
    executor_handle: *mut SevenDExecutorHandle,
    out_buf: *mut c_char,
    buf_len: usize,
) -> bool {
    if executor_handle.is_null() || out_buf.is_null() {
        return false;
    }

    let executor = unsafe { &*((*executor_handle).0) };
    let caps = executor.capabilities();

    let info = format!(
        "Cores: {}, AVX-512: {}, CUDA: {}, ROCm: {}",
        caps.cpu_cores, caps.has_avx512, caps.has_cuda, caps.has_rocm
    );

    let c_info = CString::new(info).unwrap();
    let bytes = c_info.as_bytes_with_nul();

    if bytes.len() > buf_len {
        return false;
    }

    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr() as *const i8, out_buf, bytes.len());
    }

    true
}

// Re-export modules
pub mod allocator;
pub mod executor;
pub mod ir;
pub mod jit;
pub mod quantum;
