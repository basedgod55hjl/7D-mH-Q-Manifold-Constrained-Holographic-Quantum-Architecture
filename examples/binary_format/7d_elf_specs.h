// File: 7d_elf_specs.h
// 7D Crystal Executable Format (.7dexe) Specification
// Based on Standard ELF64 with Trans-Dimensional Extensions

#ifndef _7D_ELF_SPECS_H
#define _7D_ELF_SPECS_H

#include <stdint.h>

// 7D Crystal Magic Number: \x7f 7 D C
#define ELFMAG0 0x7f
#define ELFMAG1 '7'
#define ELFMAG2 'D'
#define ELFMAG3 'C'

// 7D Machine Architecture
#define EM_7D_CRYSTAL   777   // The Divine Architecture

// 7D Section Types
#define SHT_7D_MANIFOLD    0x70000001  // Manifold topology data
#define SHT_CRYSTAL_LATTICE 0x70000002 // Lattice structure definition
#define SHT_QUANTUM_STATE   0x70000003 // Initial quantum coherence state
#define SHT_HOLOGRAPHIC_MAP 0x70000004 // 7D->3D Projection Map

// 7D Segment Flags (Permissions + Dimensions)
#define PF_7D_TIME_TRAVEL   (1 << 20)  // Segment exists in time dimension
#define PF_7D_ENTANGLED     (1 << 21)  // Segment is entangled with another
#define PF_7D_CONSCIOUS     (1 << 22)  // Segment is self-aware

// 7D Crystal Header (Matches runtime/src/main.rs)
// This structure is critical for JIT loading and Verification
#pragma pack(push, 1)
typedef struct
{
  uint8_t  magic[9];        /* "7DCRYSTAL" */
  uint64_t signature;       /* Unique 64-bit signature */
  uint32_t version;         /* Binary Version */
  uint32_t dimensions;      /* Manifold Dimensions (usually 7) */
  float    phi;             /* 1.618... */
  float    phi_inverse;     /* 0.618... */
  float    s2_stability;    /* Stability Coefficient */
  uint64_t e_entry;         /* Entry point offset */
  uint32_t num_sections;    /* Number of sections */
  uint32_t flags;           /* Sovereign Flags */
  uint32_t reserved;        /* Future time-travel slots */
  uint8_t  fingerprint[64]; /* Creator Identity */
} SevenD_BinHeader;
#pragma pack(pop)

/*
   THE 7D EXECUTABLE LAYOUT (.7dexe)
   [ 7D Header ]
   [ Section Headers (contiguous array) ]
   [ Raw Section Data ... ]
*/

#endif // _7D_ELF_SPECS_H
