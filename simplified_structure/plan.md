# Plan: Asymmetric Simplified Fishbone Slow-Light Grating — MPB Sweep & Optimization

## Context

The existing `fishbone_gc_2d.py` simulates a 3-rib-per-side symmetric fishbone photonic crystal waveguide using MPB (effective-index 2D model, n_eff=2.85 for 220nm Si at 1550nm). The structure achieves slow light with ng = 13–40.

The new task is to design a **simpler, more fab-friendly asymmetric variant** (one rib per side, ribs can differ in size, with a lateral offset delta_s) and use MPB sweeps to find parameters achieving:
- **ng ≈ 6** (wide, flat band)
- **Bandwidth > 5nm** (wavelength range where 5 ≤ ng ≤ 7)
- **Fabrication constraint**: all features and gaps ≥ 160nm

The new file: `simplified_structure/simplified_gc_2d.py`.

---

## Structure Geometry (in 2D MPB unit cell)

- **Periodic direction**: x, period = 1a
- **Central spine**: continuous Si ridge, width = 2·h_spine in y
- **Top rib**: Si block, size = (Wt × ht), centered at x = +delta_s/2, y = h_spine + ht/2
- **Bottom rib**: Si block, size = (Wb × hb), centered at x = −delta_s/2, y = −(h_spine + hb/2)
- **Background**: SiO₂ (n=1.44)
- **No y-mirror symmetry** → use `ms.run()`, not `ms.run_yeven()`

Supercell height: `sy = 2*(h_spine + max(ht,hb)) + 2*pad_y`

---

## Fabrication Constraints (physical nm)

| Constraint | Formula | Minimum |
|---|---|---|
| Spine width | 2·h_spine·a_nm | > 160nm |
| Top rib width | Wt·a_nm | > 160nm |
| Top rib height | ht·a_nm | > 160nm |
| Bottom rib width | Wb·a_nm | > 160nm |
| Bottom rib height | hb·a_nm | > 160nm |
| Top rib gap (x) | (1 − Wt)·a_nm | > 160nm |
| Bottom rib gap (x) | (1 − Wb)·a_nm | > 160nm |
| Top rib in cell | delta_s/2 + Wt/2 ≤ 0.5 | — |

---

## Key Implementation Details

### MPB Call (asymmetric → no symmetry)
```python
ms = mpb.ModeSolver(
    geometry_lattice=lattice,
    geometry=geometry,
    k_points=k_points,
    resolution=resolution,
    num_bands=num_bands,
    default_material=mp.Medium(index=n_SiO2),   # SiO2 background
)
ms.run()   # NOT run_yeven()
freqs = np.array(ms.all_freqs)   # shape (nk, num_bands)
```

### Group Index
```python
ng = 1.0 / np.gradient(f_band, k_x)   # same as fishbone_gc_2d.py
wavelengths_nm = a_nm / f_band
```

### Figure of Merit
```
FOM = max contiguous bandwidth (nm) where ng ∈ [5, 7]
Goal: FOM > 5nm with center near 1550nm
```

### Auto Band Detection
For each band: skip if f≤0; compute ng, restrict to λ ∈ [1400, 1700nm]; score = bandwidth − 0.5·|ng_median − 6|. Return highest-scoring band.

### Caching
- Path: `simplified_structure/cache/simplified2d_res{R}_nb{N}_{hash8}.npz`
- Hash: MD5 of geometry params dict (a_nm, h_spine, Wt, ht, Wb, hb, delta_s, n_eff, n_SiO2, pad_y)
- Identical pattern to fishbone_gc_2d.py

---

## Optimization Workflow (Two-Phase Grid Search)

### Phase 1: Symmetric Sweep (Wt=Wb, ht=hb, delta_s=0)
Sweep grid satisfying fab constraints with 20nm safety margin:
- `a_nm` ∈ {380, 435, 490, 545, 600} (5 values)
- `h_spine` ∈ 4 values in [80/a + 20/a, 0.55]
- `W` (= Wt = Wb) ∈ 4 values in [160/a + 20/a, min((a−160)/a − 20/a, 0.60)]
- `h` (= ht = hb) ∈ 4 values in [160/a + 20/a, 0.70]

Total: 5 × 4 × 4 × 4 = **320 runs**

### Phase 2: Asymmetric Sweep (around top-5 Phase 1 results)
For each top Phase 1 base:
- Width ratio rW = Wt/Wb ∈ {0.7, 0.85, 1.0, 1.15, 1.3} (preserving average W)
- Height ratio rH = ht/hb ∈ {0.7, 0.85, 1.0, 1.15, 1.3} (preserving average h)
- delta_s ∈ {0, 0.075, 0.15, 0.225, 0.3} (units of a)

Total: 5 × 5 × 5 × 5 = **≤ 625 runs** (minus Phase 1 cache hits at rW=rH=1, delta_s=0)

### Constants
```python
RESOLUTION = 32
NUM_BANDS = 8
K_MIN = 0.4, K_MAX = 0.5, K_INTERP = 30
TARGET_NG = 6.0, NG_LOW = 5.0, NG_HIGH = 7.0
MIN_BW_NM = 5.0, WL_CENTER = 1550.0
MIN_FEAT_NM = 160.0
```

---

## File Structure

```
simplified_structure/
├── simplified_gc_2d.py          ← simulation & optimization script
├── plan.md                      ← this file
├── cache/                       ← MPB result cache (auto-created)
└── output/                      ← plots and CSV (auto-created)
```

---

## CLI Usage

```bash
# Full two-phase optimization (recommended)
python simplified_gc_2d.py --optimize --save-plots

# Phase 1 only (symmetric coarse sweep)
python simplified_gc_2d.py --sweep --save-plots

# Phase 2 only (requires Phase 1 cache)
python simplified_gc_2d.py --sweep-asym --save-plots

# Quick coarse test
python simplified_gc_2d.py --sweep --n-a 3 --n-spine 3 --n-W 3 --n-h 3

# Single parameter set
python simplified_gc_2d.py --params a_nm=490 Wt=0.45 ht=0.50 Wb=0.40 hb=0.55 delta_s=0.1

# Plot from existing cache (no new MPB runs)
python simplified_gc_2d.py --plot-only

# Export results to CSV
python simplified_gc_2d.py --plot-only --export-csv
```

---

## Verification

1. Run `--sweep` and confirm cache files are created in `cache/`
2. Check `print_top_results` shows at least one result with FOM > 5nm
3. Run `--params` with top result params, verify band plot shows flat ng ≈ 6 region
4. Confirm all reported dimensions satisfy 160nm fab constraint
