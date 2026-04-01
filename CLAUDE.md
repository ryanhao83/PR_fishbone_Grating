# PR Fishbone Grating — Photonic Crystal Slow-Light Waveguide Design

## Project Goal

Design and optimize **fishbone-type photonic crystal waveguide** structures for broadband slow light using MIT Photonic Bands (MPB). Two structure variants are studied:

1. **Original fishbone** (`fishbone_gc_2d.py`): 3 nested ribs per side, symmetric, targets high ng (13–40)
2. **Simplified asymmetric** (`simplified_structure/`): 1 rib per side, asymmetric allowed, targets moderate ng = 6 with bandwidth > 5nm, fab-friendly (all features > 160nm)

The target application is silicon photonic integrated circuits at telecom C-band (~1550nm) on a 220nm SOI platform.

## Technical Approach

- **Effective-index 2D model**: The 3D slab waveguide (220nm Si on SiO2) is reduced to 2D using n_eff = 2.85 for the TE slab mode. This enables fast parameter sweeps.
- **MPB simulation**: Computes photonic band structure for 1D-periodic unit cells. Group index ng = 1/(df/dk) extracted from band dispersion near the Brillouin zone edge (k = 0.4–0.5).
- **Caching**: All MPB results are cached as `.npz` files keyed by MD5 hash of geometry parameters. Re-runs automatically skip cached results.
- **Single-mode verification**: For each slow-light band, check that no neighboring guided band has overlapping frequency range within the slow-light window. This ensures single-mode operation.

## Directory Structure

```
PR_fishbone_grating/
├── CLAUDE.md                    # this file
├── fishbone_gc_2d.py            # original 3-rib fishbone, 2D effective-index
├── fishbone_gc_3d.py            # 3D validation of fishbone (220nm Si slab)
├── fishbone_gc.py               # standalone reference script
├── postprocessing.py            # plotting utilities for fishbone results
├── cache_2d/                    # cached MPB results for fishbone
├── simplified_structure/
│   ├── plan.md                  # design plan for simplified structure
│   ├── simplified_gc_2d.py      # asymmetric simplified grating: sweep + optimize
│   ├── cache/                   # cached MPB results (~750 param sets)
│   └── output/                  # plots, CSV exports
└── *.png                        # fishbone result plots
```

## Conventions

- All spatial geometry parameters are in units of lattice constant `a`, except `a_nm` which is in nanometers
- Normalized frequency: f = a/lambda (dimensionless)
- Wavevector: k in units of 2*pi/a
- Group index: ng = 1 / (df/dk)
- Physical wavelength: lambda_nm = a_nm / f
- Band indexing: 0-based in Python arrays, 1-based in MPB internal calls
- Cache filenames encode resolution, num_bands, and an 8-char MD5 hash of geometry params

## Key Parameters

### Original fishbone (`fishbone_gc_2d.py`)
- a = 420nm, n_eff = 2.85, n_SiO2 = 1.44
- 3 rib pairs: (w1,h1), (w2,h2), (w3,h3) + central spine h_spine
- Y-mirror symmetric → `ms.run_yeven()`
- TARGET_BAND = 11 (hardcoded), RESOLUTION = 32, NUM_BANDS = 18

### Simplified structure (`simplified_structure/simplified_gc_2d.py`)
- a = 380–600nm (swept), n_eff = 2.85, n_SiO2 = 1.44
- 1 top rib (Wt, ht) + 1 bottom rib (Wb, hb) + spine h_spine + offset delta_s
- Asymmetric → `ms.run()` (no symmetry)
- Band auto-detected, RESOLUTION = 32, NUM_BANDS = 8
- Fabrication constraint: all features/gaps > 160nm

## Running the Code

Requires: `meep` with MPB support (`conda` environment `mpb_env`).

```bash
# Original fishbone
python fishbone_gc_2d.py --run --sweep-spine

# Simplified structure — full optimization
python simplified_structure/simplified_gc_2d.py --optimize --save-plots --no-show

# Single parameter set
python simplified_structure/simplified_gc_2d.py --params a_nm=389 h_spine=0.550 Wt=0.484 ht=0.514 Wb=0.569 hb=0.735 delta_s=0.0 --save-plots

# Plot from cache
python simplified_structure/simplified_gc_2d.py --plot-only
```

## Current Best Result (Simplified Structure, Single-Mode Verified)

| Parameter | Value |
|---|---|
| a | 389 nm |
| Spine full width | 428 nm |
| Top rib | Wt=188nm, ht=200nm |
| Bottom rib | Wb=221nm, hb=286nm |
| Offset delta_s | 0 nm |
| **ng_mean** | **5.64** |
| **Bandwidth (ng in [5,7])** | **61 nm** |
| **Center wavelength** | **1549 nm** |
| Single-mode | Verified (no neighbor band overlap) |
| All fab constraints | Passed (min feature 168nm > 160nm) |
