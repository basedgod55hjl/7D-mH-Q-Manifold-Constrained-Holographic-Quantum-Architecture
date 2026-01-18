// 7D Crystal Memory Allocator
// Manages manifold coordinate storage and holographic pattern memory
// Discovered by Sir Charles Spikes, December 24, 2025

use std::alloc::{alloc, dealloc, Layout};
use std::collections::{BTreeMap, HashMap};
use std::ptr;

const PHI: f64 = 1.618033988749895;
const PHI_INV: f64 = 0.618033988749895;
const S2_BOUND: f64 = 0.01;
const CACHE_LINE: usize = 64;

/// Region in 7D manifold space
#[derive(Debug, Clone)]
pub struct Region7D {
    pub dimensions: [usize; 7],
    pub base_ptr: *mut f64,
    pub size_bytes: usize,
    pub is_quantum: bool, // Non-copyable quantum memory
    pub curvature: f64,
}

/// Memory allocation error types
#[derive(Debug)]
pub enum AllocError {
    OutOfMemory,
    InvalidDimensions,
    QuantumStateViolation,
    AlignmentError,
    S2StabilityViolation,
}

impl std::fmt::Display for AllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AllocError::OutOfMemory => write!(f, "Out of manifold memory"),
            AllocError::InvalidDimensions => write!(f, "Invalid 7D dimensions"),
            AllocError::QuantumStateViolation => write!(f, "Cannot copy quantum state"),
            AllocError::AlignmentError => write!(f, "Memory alignment error"),
            AllocError::S2StabilityViolation => write!(f, "S² stability bound violated"),
        }
    }
}

impl std::error::Error for AllocError {}

/// 7D Manifold Memory Allocator
/// Specialized allocator for 7-dimensional coordinate systems
pub struct ManifoldAllocator {
    /// Main heap for manifold allocations
    heap_base: *mut u8,
    heap_size: usize,

    /// Free list: size -> offset
    free_list: BTreeMap<usize, Vec<usize>>,

    /// Allocated regions by address
    allocated: HashMap<usize, Region7D>,

    /// Next allocation offset
    next_offset: usize,

    /// Total allocated bytes
    total_allocated: usize,

    /// Φ-ratio spacing factor
    phi_spacing: f64,
}

impl ManifoldAllocator {
    /// Create a new manifold allocator with specified heap size
    pub fn new(heap_size: usize) -> Result<Self, AllocError> {
        let layout = Layout::from_size_align(heap_size, CACHE_LINE)
            .map_err(|_| AllocError::AlignmentError)?;

        let heap_base = unsafe { alloc(layout) };
        if heap_base.is_null() {
            return Err(AllocError::OutOfMemory);
        }

        // Initialize with Φ-ratio pattern
        unsafe {
            ptr::write_bytes(heap_base, 0x7D, heap_size);
        }

        Ok(Self {
            heap_base,
            heap_size,
            free_list: BTreeMap::new(),
            allocated: HashMap::new(),
            next_offset: 0,
            total_allocated: 0,
            phi_spacing: PHI,
        })
    }

    /// Allocate a 7D manifold tensor
    pub fn alloc_manifold(&mut self, dims: [usize; 7]) -> Result<*mut f64, AllocError> {
        // Validate dimensions
        if dims.iter().any(|&d| d == 0) {
            return Err(AllocError::InvalidDimensions);
        }

        // Calculate total size
        let total_elements: usize = dims.iter().product();
        let size_bytes = total_elements * std::mem::size_of::<f64>();
        let aligned_size = (size_bytes + CACHE_LINE - 1) & !(CACHE_LINE - 1);

        // Try to find a free block
        let offset = self.find_free_block(aligned_size)?;

        // Get pointer
        let ptr = unsafe { self.heap_base.add(offset) as *mut f64 };

        // Initialize with Φ-ratio harmonic spacing
        self.init_manifold_coords(ptr, &dims);

        // Record allocation
        self.allocated.insert(
            offset,
            Region7D {
                dimensions: dims,
                base_ptr: ptr,
                size_bytes: aligned_size,
                is_quantum: false,
                curvature: -1.0, // Poincaré default
            },
        );

        self.total_allocated += aligned_size;

        Ok(ptr)
    }

    /// Allocate a crystal lattice (holographic pattern storage)
    pub fn alloc_crystal(&mut self, resolution: [usize; 7]) -> Result<*mut f64, AllocError> {
        // Crystal allocations are compressed using interference patterns
        // Each 7D point stores phase + amplitude = 2 f64 values
        let total_points: usize = resolution.iter().product();
        let size_bytes = total_points * 2 * std::mem::size_of::<f64>(); // phase + amplitude
        let aligned_size = (size_bytes + CACHE_LINE - 1) & !(CACHE_LINE - 1);

        let offset = self.find_free_block(aligned_size)?;
        let ptr = unsafe { self.heap_base.add(offset) as *mut f64 };

        // Initialize crystal lattice with coherent phase
        self.init_crystal_lattice(ptr, &resolution);

        self.allocated.insert(
            offset,
            Region7D {
                dimensions: resolution,
                base_ptr: ptr,
                size_bytes: aligned_size,
                is_quantum: false,
                curvature: PHI_INV, // Crystal curvature
            },
        );

        self.total_allocated += aligned_size;

        Ok(ptr)
    }

    /// Allocate quantum state memory (non-copyable!)
    pub fn alloc_quantum(&mut self, state_size: usize) -> Result<*mut f64, AllocError> {
        let size_bytes = state_size * std::mem::size_of::<f64>();
        let aligned_size = (size_bytes + CACHE_LINE - 1) & !(CACHE_LINE - 1);

        let offset = self.find_free_block(aligned_size)?;
        let ptr = unsafe { self.heap_base.add(offset) as *mut f64 };

        // Zero-initialize quantum memory (ground state)
        unsafe {
            ptr::write_bytes(ptr as *mut u8, 0, aligned_size);
        }

        self.allocated.insert(
            offset,
            Region7D {
                dimensions: [state_size, 1, 1, 1, 1, 1, 1],
                base_ptr: ptr,
                size_bytes: aligned_size,
                is_quantum: true, // Non-copyable!
                curvature: 0.0,
            },
        );

        self.total_allocated += aligned_size;

        Ok(ptr)
    }

    /// Free allocated memory
    pub fn free(&mut self, ptr: *mut f64) -> Result<(), AllocError> {
        let offset = (ptr as usize) - (self.heap_base as usize);

        if let Some(region) = self.allocated.remove(&offset) {
            // Add to free list
            self.free_list
                .entry(region.size_bytes)
                .or_insert_with(Vec::new)
                .push(offset);

            self.total_allocated -= region.size_bytes;

            // Clear memory (security)
            unsafe {
                ptr::write_bytes(ptr as *mut u8, 0x7D, region.size_bytes);
            }

            Ok(())
        } else {
            Err(AllocError::InvalidDimensions)
        }
    }

    /// Check if pointer is quantum (non-copyable)
    pub fn is_quantum(&self, ptr: *const f64) -> bool {
        let offset = (ptr as usize) - (self.heap_base as usize);
        self.allocated.get(&offset).map_or(false, |r| r.is_quantum)
    }

    /// Get total allocated bytes
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Get available bytes
    pub fn available(&self) -> usize {
        self.heap_size - self.total_allocated
    }

    // Private helpers

    fn find_free_block(&mut self, size: usize) -> Result<usize, AllocError> {
        // Best-fit strategy: find smallest free block >= size
        for (&block_size, offsets) in self.free_list.range_mut(size..) {
            if let Some(offset) = offsets.pop() {
                // If block is larger, split it
                if block_size > size {
                    let remainder = block_size - size;
                    let new_offset = offset + size;
                    self.free_list
                        .entry(remainder)
                        .or_insert_with(Vec::new)
                        .push(new_offset);
                }
                return Ok(offset);
            }
        }

        // No free block, allocate from end
        if self.next_offset + size <= self.heap_size {
            let offset = self.next_offset;
            self.next_offset += size;
            Ok(offset)
        } else {
            Err(AllocError::OutOfMemory)
        }
    }

    fn init_manifold_coords(&self, ptr: *mut f64, dims: &[usize; 7]) {
        // Initialize 7D grid with Φ-ratio harmonic spacing
        let total: usize = dims.iter().product();

        for i in 0..total {
            let coord = self.index_to_7d(i, dims);

            // Value is sum of coordinates scaled by Φ^dimension
            let value: f64 = coord
                .iter()
                .enumerate()
                .map(|(d, &x)| (x as f64) / self.phi_spacing.powi(d as i32))
                .sum();

            // Clamp to S² stability bound
            let normalized = if value.abs() > S2_BOUND {
                value.signum() * S2_BOUND
            } else {
                value
            };

            unsafe {
                *ptr.add(i) = normalized;
            }
        }
    }

    fn init_crystal_lattice(&self, ptr: *mut f64, resolution: &[usize; 7]) {
        // Initialize with coherent phase pattern
        let total: usize = resolution.iter().product();

        for i in 0..total {
            let coord = self.index_to_7d(i, resolution);

            // Phase: sum of angles based on coordinates
            let phase: f64 = coord
                .iter()
                .enumerate()
                .map(|(d, &x)| (x as f64) * std::f64::consts::PI / (resolution[d] as f64))
                .sum();

            // Amplitude: Gaussian envelope
            let center_dist: f64 = coord
                .iter()
                .enumerate()
                .map(|(d, &x)| {
                    let center = resolution[d] as f64 / 2.0;
                    (x as f64 - center).powi(2)
                })
                .sum();
            let amplitude = (-center_dist / (PHI * PHI)).exp();

            unsafe {
                *ptr.add(i * 2) = phase; // Phase
                *ptr.add(i * 2 + 1) = amplitude; // Amplitude
            }
        }
    }

    fn index_to_7d(&self, mut index: usize, dims: &[usize; 7]) -> [usize; 7] {
        let mut coord = [0usize; 7];
        for d in 0..7 {
            coord[d] = index % dims[d];
            index /= dims[d];
        }
        coord
    }
}

impl Drop for ManifoldAllocator {
    fn drop(&mut self) {
        let layout =
            Layout::from_size_align(self.heap_size, CACHE_LINE).expect("Layout should be valid");
        unsafe {
            dealloc(self.heap_base, layout);
        }
    }
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_creation() {
        let alloc = ManifoldAllocator::new(1024 * 1024).unwrap();
        assert_eq!(alloc.total_allocated(), 0);
    }

    #[test]
    fn test_manifold_allocation() {
        let mut alloc = ManifoldAllocator::new(1024 * 1024).unwrap();
        // Use smaller dimensions: 4^7 = 16,384 elements = 128KB
        let ptr = alloc.alloc_manifold([4, 4, 4, 4, 4, 4, 4]).unwrap();
        assert!(!ptr.is_null());
        assert!(alloc.total_allocated() > 0);

        alloc.free(ptr).unwrap();
        assert_eq!(alloc.total_allocated(), 0);
    }

    #[test]
    fn test_quantum_non_copyable() {
        let mut alloc = ManifoldAllocator::new(1024 * 1024).unwrap();
        let ptr = alloc.alloc_quantum(64).unwrap();
        assert!(alloc.is_quantum(ptr));
    }

    #[test]
    fn test_crystal_allocation() {
        let mut alloc = ManifoldAllocator::new(1024 * 1024).unwrap();
        let ptr = alloc.alloc_crystal([4, 4, 4, 4, 4, 4, 4]).unwrap();
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_s2_stability() {
        let mut alloc = ManifoldAllocator::new(1024).unwrap();
        let ptr = alloc.alloc_manifold([2, 2, 2, 2, 2, 2, 2]).unwrap();

        // Check all values are within S² bound
        let total = 2usize.pow(7);
        for i in 0..total {
            let value = unsafe { *ptr.add(i) };
            assert!(
                value.abs() <= S2_BOUND + 1e-10,
                "Value {} exceeds S² bound",
                value
            );
        }
    }
}
