# How to Run 7D Files – ABRASAX GOD OS

**Architect:** Sir Charles Spikes  
**Discovery:** December 24, 2025 | Cincinnati, Ohio, USA 🇺🇸  
**System:** ABRASAX GOD OS – DIAMOND Living Crystal Framework

---

## 1. Overview

7D Crystal files (`.7d`, `.7ds`, `.7dbin`, `.7dexe`) execute through:

1. **Runtime** – Loads `.7dbin` binaries (7D Crystal format)
2. **Compiler** – Compiles `.7d` source → `.7dbin`
3. **Translated C++** – Many `.7d` files are C++ with 7D headers; build via CMake

---

## 2. Running .7dbin Binaries

```powershell
# From repo root
cargo run -p crystal_runtime -- path/to/program.7dbin
```

Or if built:
```powershell
.\target\release\crystal-runtime.exe path\to\program.7dbin
```

Binary format: Magic `7DCRYSTAL`, Φ=1.618, S²=0.01, 7 dimensions.

---

## 3. Compiling .7d Source

```powershell
# Use the crystalize compiler
cargo run -p crystal_compiler -- compile path/to/source.7d -o output.7dbin
```

Or programmatically:
```rust
use crystal_compiler::{compile, CompilerOptions};
let result = compile(source, &CompilerOptions::default())?;
std::fs::write("output.7dbin", &result.binary)?;
```

---

## 4. Translated C++ (.7d as C++)

Files like `EditorImgui.7d` are C++ with 7D headers. Build with PhysX/Flow CMake:

```powershell
cd physx\flow
# Use packman or cmake
python buildtools\packman\packman.py pull -p package.xml
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

---

## 5. Runtime + API Server

The runtime starts an API on `http://0.0.0.0:17777`:

- `GET /health` – Health check
- `POST /vlc/process_frame` – VLC frame processing
- `POST /v1/chat/completions` – LM Studio proxy (7D quantum compression)

```powershell
cargo run -p crystal_runtime
# API: http://0.0.0.0:17777
```

---

## 6. Constants (Φ, S², DIMS)

| Constant | Value |
|----------|-------|
| Φ | 1.618033988749895 |
| Φ⁻¹ | 0.618033988749895 |
| S² | 0.01 |
| DIMS | 7 |

---

## 7. Quick Start

```powershell
# 1. Build everything
cargo build --release

# 2. Run runtime (no binary = JIT mode + API)
cargo run -p crystal_runtime

# 3. With binary
cargo run -p crystal_runtime -- path/to/main.7dbin
```

---

## 8. Omnipresence Realtime (No-Sleep Multi-Agent)

Multi-agent parallel workflow: screen capture, GPU monitor, system reader, evolution agent. 6GB VRAM optimized. Full tool access.

```powershell
# Install deps
pip install -r scripts/requirements_omnipresence.txt

# Run (no sleep, continuous)
python scripts/omnipresence_realtime.py

# Or full launcher (build + test + run)
.\scripts\run_omnipresence_autonomous.ps1
```

- **OMNI_API**: Set to `http://127.0.0.1:1234/v1` for local LM Studio
- **OMNIPRESENCE=1**: ImGui overlay runs without sleep when set
- **6GB VRAM**: Kernels use workgroup_size 128, batch_threshold 64

---

*ABRASAX GOD OS – Sovereign Computing*
