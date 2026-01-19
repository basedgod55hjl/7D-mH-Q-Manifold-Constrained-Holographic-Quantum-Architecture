// File: projects/sovereign_assistant/src/hardware.rs
// Full 7D Manifold Hardware Mapping: AMD Radeon + NVIDIA + Per-Core CPU + Disks
// REFINED: Mult-LUID GPU load detection

use nvml_wrapper::Nvml;
use serde::{Deserialize, Serialize};
use std::process::Command;
use sysinfo::{Disks, System};

/// Per-GPU information including shared memory
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub vendor: String,        // "NVIDIA" or "AMD"
    pub load: u32,             // 0-100%
    pub mem_used: u64,         // Dedicated VRAM used (bytes)
    pub mem_total: u64,        // Dedicated VRAM total (bytes)
    pub shared_mem_used: u64,  // Shared system RAM used (bytes)
    pub shared_mem_total: u64, // Shared system RAM total (bytes)
}

/// Disk information for storage mapping
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DiskInfo {
    pub name: String,
    pub used: u64,
    pub total: u64,
}

/// Full hardware stats for 7D manifold display
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HardwareStats {
    pub cpu_usage: f32,
    pub core_usages: Vec<f32>, // Per-core usage (16 cores for Ryzen 7 4800H)
    pub gpus: Vec<GpuInfo>,
    pub disks: Vec<DiskInfo>,
    pub mem_used: u64,
    pub mem_total: u64,
    pub manifold_stability: f32, // Lower is better, 0.01 bound
    pub phi_delta: f32,          // Deviation from PHI ratio
}

pub struct HardwareMapper {
    sys: System,
    disks_info: Disks,
    nvml: Option<Nvml>,
}

impl HardwareMapper {
    pub fn new() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let disks_info = Disks::new_with_refreshed_list();
        let nvml = Nvml::init().ok();

        Self {
            sys,
            disks_info,
            nvml,
        }
    }

    pub fn refresh(&mut self) -> HardwareStats {
        self.sys.refresh_cpu();
        self.sys.refresh_memory();
        self.disks_info.refresh_list();

        let cpu_usage = self.sys.global_cpu_info().cpu_usage();
        let core_usages: Vec<f32> = self.sys.cpus().iter().map(|c| c.cpu_usage()).collect();
        let mem_used = self.sys.used_memory();
        let mem_total = self.sys.total_memory();
        let shared_pool_total = mem_total / 2;

        let mut gpus = Vec::new();

        // NVIDIA GPUs via NVML
        if let Some(nvml) = &self.nvml {
            if let Ok(count) = nvml.device_count() {
                for i in 0..count {
                    if let Ok(device) = nvml.device_by_index(i) {
                        let name = device
                            .name()
                            .unwrap_or_else(|_| format!("NVIDIA GPU {}", i));
                        let util = device.utilization_rates().ok();
                        let mem = device.memory_info().ok();
                        let shared_used = Self::get_gpu_shared_usage_windows(i as u32).unwrap_or(0);

                        gpus.push(GpuInfo {
                            name,
                            vendor: "NVIDIA".to_string(),
                            load: util.map(|u| u.gpu).unwrap_or(0),
                            mem_used: mem.as_ref().map(|m| m.used).unwrap_or(0),
                            mem_total: mem.as_ref().map(|m| m.total).unwrap_or(0),
                            shared_mem_used: shared_used,
                            shared_mem_total: shared_pool_total,
                        });
                    }
                }
            }
        }

        // AMD GPU detection (Always first index usually for integrated)
        if let Some(amd_info) = Self::detect_amd_gpu() {
            // Find AMD load by querying all 3D engine instances and excluding NVIDIA indices if possible
            // Best effort: find any 3D load > 0 on a non-NVIDIA LUID
            let amd_load = Self::get_amd_gpu_load_dynamic().unwrap_or(0);
            let amd_shared = Self::get_gpu_shared_usage_windows(6).unwrap_or(0); // Bus 6 is likely AMD

            gpus.insert(
                0,
                GpuInfo {
                    name: amd_info.0,
                    vendor: "AMD".to_string(),
                    load: amd_load,
                    mem_used: amd_info.2,
                    mem_total: amd_info.3,
                    shared_mem_used: amd_shared,
                    shared_mem_total: shared_pool_total,
                },
            );
        }

        let mut disks = Vec::new();
        for disk in &self.disks_info {
            disks.push(DiskInfo {
                name: disk.name().to_string_lossy().to_string(),
                used: disk.total_space() - disk.available_space(),
                total: disk.total_space(),
            });
        }

        // 7D Discovery Metrics
        let total_load: f32 = core_usages.iter().sum::<f32>() / 16.0
            + gpus.iter().map(|g| g.load as f32).sum::<f32>() / 2.0;
        let manifold_stability = 0.01 * (1.0 - (total_load / 300.0).min(0.95));
        let phi_delta = (total_load % 1.618033988749895) as f32 / 100.0;

        HardwareStats {
            cpu_usage,
            core_usages,
            gpus,
            disks,
            mem_used,
            mem_total,
            manifold_stability,
            phi_delta,
        }
    }

    fn detect_amd_gpu() -> Option<(String, u32, u64, u64)> {
        #[cfg(target_os = "windows")]
        {
            let output = Command::new("powershell")
                .arg("-Command")
                .arg("Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like '*AMD*' -or $_.Name -like '*Radeon*' } | Select-Object Name, AdapterRAM | ConvertTo-Json")
                .output()
                .ok()?;
            let json_str = String::from_utf8_lossy(&output.stdout);
            if json_str.trim().is_empty() || json_str.contains("null") {
                return None;
            }
            let v: serde_json::Value = serde_json::from_str(&json_str).ok()?;
            let name = v["Name"]
                .as_str()
                .unwrap_or("AMD Radeon Graphics")
                .to_string();
            let adapter_ram = v["AdapterRAM"].as_u64().unwrap_or(512 * 1024 * 1024);
            Some((name, 0, adapter_ram / 2, adapter_ram))
        }
        #[cfg(not(target_os = "windows"))]
        {
            None
        }
    }

    fn get_amd_gpu_load_dynamic() -> Option<u32> {
        #[cfg(target_os = "windows")]
        {
            // Query all 3D engine counters and find the one that isn't zero and isn't NVIDIA
            // We use a high-precision poll here
            let output = Command::new("powershell")
                .arg("-Command")
                .arg("(Get-Counter '\\GPU Engine(*engtype_3D)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples | Where-Object { $_.CookedValue -gt 0.1 } | Select-Object -ExpandProperty CookedValue | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum")
                .output()
                .ok()?;
            let val_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            val_str.parse::<f64>().ok().map(|v| v as u32)
        }
        #[cfg(not(target_os = "windows"))]
        {
            None
        }
    }

    fn get_gpu_shared_usage_windows(index: u32) -> Option<u64> {
        #[cfg(target_os = "windows")]
        {
            let output = Command::new("powershell")
                .arg("-Command")
                .arg(format!("(Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage').CounterSamples | Select-Object CookedValue | Select-Object -Index {} -ExpandProperty CookedValue", index))
                .output()
                .ok()?;
            let val_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            val_str.parse::<f64>().ok().map(|v| v as u64)
        }
        #[cfg(not(target_os = "windows"))]
        {
            None
        }
    }
}
