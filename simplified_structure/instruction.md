# Simplified Fishbone Grating — User Instruction Manual

## Overview

This toolkit designs and optimizes asymmetric fishbone-type photonic crystal waveguides for broadband slow light using MIT Photonic Bands (MPB). The workflow is:

1. **2D sweep** (`simplified_gc_2d.py`) — Fast parameter exploration using effective-index approximation
2. **3D validation** (`simplified_gc_3d.py`) — Full 3D slab waveguide simulation of top candidates
3. **Band unfolding** (`band_unfolding.py`) — Post-processing to resolve band crossings and analyze slow light

## Prerequisites

```bash
conda activate mpb_env   # meep + MPB environment
cd simplified_structure/
```

---

## Step 1: 2D Parameter Sweep

### Full optimization (Phase 1 symmetric + Phase 2 asymmetric)

```bash
python simplified_gc_2d.py --optimize --save-plots --no-show
```

This sweeps over lattice constant `a_nm`, spine width, rib dimensions, and asymmetry offset. Results are cached in `cache/` as `.npz` files. Subsequent runs skip cached parameter sets.

### Check a single parameter set

```bash
python simplified_gc_2d.py --params a_nm=389 h_spine=0.550 Wt=0.484 ht=0.514 Wb=0.569 hb=0.735 delta_s=0.0 --save-plots
```

### View top results from cache (no MPB runs)

```bash
python simplified_gc_2d.py --plot-only --save-plots --no-show
```

### Export results to CSV

```bash
python simplified_gc_2d.py --plot-only --export-csv
# Output: output/phase2_results.csv
```

### Key 2D parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| `a_nm` | Lattice constant | nm |
| `h_spine` | Half-width of central spine | a |
| `Wt`, `ht` | Top rib width and height | a |
| `Wb`, `hb` | Bottom rib width and height | a |
| `delta_s` | Asymmetry offset between top/bottom ribs | a |

All spatial parameters except `a_nm` are in units of the lattice constant.

---

## Step 2: 3D Validation

### Run 3D MPB on top 2D candidates

```bash
python simplified_gc_3d.py --run --save-plots --no-show
```

By default this runs the top 3 structures from the 2D ranking. Use `--id N` to run a specific one:

```bash
python simplified_gc_3d.py --run --id 0 --save-plots --no-show
```

### Higher resolution for publication

```bash
python simplified_gc_3d.py --run --resolution 32 --num-bands 8 --save-plots --no-show
```

Resolution 16 is for survey; resolution 32+ for quantitative results.

### Plot cached 3D results

```bash
python simplified_gc_3d.py --plot-only --save-plots --no-show
```

---

## Step 3: Band Unfolding & Diagnostics

MPB returns eigenvalues **sorted by frequency** at each k-point. When physical bands cross or undergo anti-crossings, band indices swap or exchange character. The band unfolding post-processor resolves these artifacts.

### Standard analysis (Tier 1: frequency-based)

Analyze all cached 3D structures:

```bash
python band_unfolding.py --analyze-3d --save-plots --no-show
```

Analyze a specific structure only:

```bash
python band_unfolding.py --analyze-3d --structure 0 --save-plots --no-show
```

**What this does:**
1. Loads cached 3D band data from `cache_3d/`
2. Unfolds bands using first-order prediction + Hungarian assignment
3. Detects and reconnects anti-crossings (avoided crossings)
4. Trims frequencies above the light cone (sets to NaN)
5. Finds flat ng regions (U-shaped slow-light criterion)
6. Checks single-mode operation (no neighbor band overlap below light cone)
7. Generates diagnostic plots in `output/`

**Output format:**
```
Structure 0: a=389nm
  Anti-crossings detected:
    Bands 1↔2 at k=0.4032 (index 11)
    Bands 3↔4 at k=0.4371 (index 18)
  Slow-light bands (flat ng region, Δng≤1.0):
    Band 2: BW=89.1nm  ng=3.07 [2.79,3.55]  λ=[1606,1695]nm  monotonic  nbr overlap [1]  → MULTIMODE
```

### Visualize mode fields at a specific k-point

To inspect what each band's mode looks like (|D|^2 at z=0):

```bash
python band_unfolding.py --plot-fields 0.40 --structure 0 --save-plots --no-show
```

This runs a **live MPB solve** (not from cache) and generates a multi-panel figure showing each band's field profile. Useful for:
- Visually confirming anti-crossings (mode character swaps between k-points)
- Identifying guided vs. radiation modes
- Checking mode symmetry

Compare fields across the anti-crossing:

```bash
python band_unfolding.py --plot-fields 0.39 --structure 0 --save-plots --no-show
python band_unfolding.py --plot-fields 0.40 --structure 0 --save-plots --no-show
python band_unfolding.py --plot-fields 0.41 --structure 0 --save-plots --no-show
```

### Tier 2: Field-overlap band tracking (definitive)

When Tier 1 (frequency-based) results are ambiguous, use Tier 2 which tracks bands by computing the physical overlap of D-fields between adjacent k-points:

```bash
python band_unfolding.py --field-overlap --structure 0 --save-plots --no-show
```

This runs a **live MPB solve** at every k-point and extracts fields, so it takes significantly longer (~2-3 min per structure at resolution 16). It also compares Tier 1 vs Tier 2 results automatically.

For higher accuracy:

```bash
python band_unfolding.py --field-overlap --structure 0 --resolution 32 --num-bands 8 --save-plots --no-show
```

---

## Diagnostic Workflow: How to Evaluate a Structure

### Quick check (< 1 min)

```bash
# Run Tier 1 analysis on all cached 3D structures
python band_unfolding.py --analyze-3d --save-plots --no-show
```

Look at the output for:
- **"SINGLE-MODE"** — the band has no neighbor overlap below the light cone
- **BW > 5nm** — sufficient slow-light bandwidth
- **ng in [5, 7]** — target group index range
- **λ near 1550nm** — target wavelength

### Deep investigation (if Tier 1 shows ambiguous crossings)

```bash
# 1. Check mode fields near the anti-crossing
python band_unfolding.py --plot-fields 0.40 --structure 0 --save-plots --no-show

# 2. Run Tier 2 for definitive tracking
python band_unfolding.py --field-overlap --structure 0 --save-plots --no-show
```

### Full pipeline from scratch

```bash
# 1. Run 2D sweep
python simplified_gc_2d.py --optimize --save-plots --no-show

# 2. Run 3D validation on top candidates
python simplified_gc_3d.py --run --save-plots --no-show

# 3. Post-process with band unfolding
python band_unfolding.py --analyze-3d --save-plots --no-show

# 4. (Optional) Field-based verification
python band_unfolding.py --field-overlap --structure 0 --save-plots --no-show
```

---

## Understanding the Plots

### Band diagram (left panel in unfolded plot)

- **Gray shaded region**: light cone (f > k/n_clad). Modes above are radiation modes.
- **Colored solid lines**: unfolded physical bands (each color = one physical band)
- **Dashed gray lines**: raw MPB bands (before unfolding) — shows where crossings were
- **Red circles**: detected anti-crossing points

### ng vs wavelength (right panel)

- **Green shaded band**: target ng range [5, 7]
- **Yellow shaded region**: detected flat ng region (the usable slow-light bandwidth)
- **Dashed blue line**: 1550nm target wavelength

### Mode field plots

- Each panel shows |D|^2 at the z=0 plane for one band
- Modes confined to the waveguide center are guided; modes spread to edges are radiation
- At an anti-crossing, two adjacent bands will show similar field patterns that gradually exchange character

---

## Key Concepts

### Slow-light criterion: flat ng region

The useful bandwidth of band-edge slow light is the **flat bottom of the U-shaped ng curve**, not where ng crosses a threshold on the steep sides. The code finds the widest wavelength window where ng variation (max - min) stays within `delta_ng = 1.0`.

### Anti-crossing vs. band crossing

- **Band crossing**: two bands pass through each other; indices swap but mode character is preserved. Detected by Hungarian assignment.
- **Anti-crossing** (avoided crossing): bands repel without crossing; indices don't swap but mode character exchanges. Detected by finding simultaneous local max (lower band) and local min (upper band).

### Light-cone trimming

After unfolding, all frequencies above the light cone (f >= k/n_clad) are set to NaN. This ensures:
- ng is only computed for guided modes
- Single-mode check only considers truly guided frequencies
- Bands that only overlap above the light cone are correctly ignored

### Single-mode check

A band is single-mode if no other guided band has overlapping frequency range below the light cone. The check is performed on the light-cone-trimmed data.

---

## File Outputs

| File | Description |
|------|-------------|
| `output/unfolded_3d_N_aXXXnm.png` | Tier 1 unfolded band diagram + ng for structure N |
| `output/unfolded_3d_N_aXXXnm_tier2.png` | Tier 2 (field-overlap) version |
| `output/mode_fields_N_kX.XXX.png` | Mode field profiles at given k |
| `output/phase2_results.csv` | 2D sweep results ranked by bandwidth |
| `output/single_mode_results.csv` | 2D results that pass single-mode check |
| `cache/*.npz` | 2D MPB cached results |
| `cache_3d/*.npz` | 3D MPB cached results |

---

## Troubleshooting

### "No cached 3D results found"
Run `simplified_gc_3d.py --run` first to generate 3D data.

### Anti-crossing not detected by Tier 1
Use `--plot-fields` to visually inspect the band character, then use `--field-overlap` for Tier 2 tracking.

### ng values look wrong
Check that the band is monotonic and has enough guided points. Very few points (< 5) give unreliable gradient estimates. Consider increasing `--k-interp` in the 3D run.

### Slow light found but at wrong wavelength
Adjust `a_nm`. Wavelength scales linearly: λ_new ≈ λ_current × (a_new / a_current).

### All bands are multimode
This is a fundamental structural limitation. Try:
1. Different geometry parameters (wider spine, different rib sizes)
2. Higher bands (may be single-mode even when lower bands overlap)
3. Larger `a_nm` to shift higher-band slow light to target wavelength
