# Plan: Asymmetric Simplified Fishbone Slow-Light Grating — MPB Sweep & Optimization

## Current Status (2026-04-02)

### Completed
1. 2D effective-index optimization: 746 parameter sets cached, top result BW=61nm at 1549nm
2. 2D single-mode screening (neighbor band overlap): 462/746 pass
3. 3D validation of top 3 structures (resolution=16, TE-only via `ms.run_zeven()`)
4. **Band unfolding** (`band_unfolding.py`): anti-crossing detection + reconnection working

### Key Finding: Anti-Crossing in 3D Band Structure

MPB returns eigenvalues sorted by frequency at each k-point. When two physical bands
undergo an **anti-crossing** (avoided crossing), they repel without swapping frequency
order, but exchange physical character. This cannot be detected by simple frequency
sorting — it requires slope analysis.

**Detected anti-crossings in all 3 structures:**
- Bands 1↔2 at k ≈ 0.403 (the slow-light region!)
- Bands 3↔4 at k ≈ 0.437

After reconnecting through anti-crossings:
- Band 2 becomes monotonically increasing (the true slow-light band, like user's "yellow curve")
- Band 1 becomes monotonically decreasing
- But Band 1 and Band 2 share guided frequencies → multimode in slow-light window

### 3D Results (After Unfolding)

| Structure | a (nm) | Band 2 BW | ng | λ_center | Band 2 mono? | Single-mode? |
|-----------|--------|-----------|-----|----------|-------------|-------------|
| 0 | 389 | 9.6 nm | 5.59 | 1532 nm | Yes | No (Band 1 overlap) |
| 1 | 380 | 9.4 nm | 5.58 | 1502 nm | Yes | No (Band 1 overlap) |
| 2 | 380 | 9.2 nm | 5.72 | 1498 nm | Yes | No (Band 1 overlap) |

However, from visual inspection of the unfolded band diagram:
- **Band 5** (highest plotted): appears to be single-mode below the light cone
- **Band 4**: overlaps with Band 3 near the light cone, but the overlap is mostly
  above the light cone → effectively single-mode for guided modes

### Key Insight: Trim Above Light Cone Before Single-Mode Check

The current `guided_freq_range()` computes the frequency range of each band below
the light cone, but the overlap check still counts bands whose guided ranges were
computed including borderline points near the light cone. A cleaner approach:

**After anti-crossing reconnection, set all frequencies above the light cone to NaN.**
This way:
- ng calculation only uses guided points
- Single-mode overlap check only considers truly guided frequencies
- Bands that only overlap above the light cone are correctly ignored

---

## Algorithm: Band Unfolding (implemented in `band_unfolding.py`)

### Two-Pass Algorithm
1. **Pass 1 — Hungarian assignment** with first-order (slope-based) prediction:
   - Start from k_max (zone edge, bands well-separated)
   - Track backwards using linear extrapolation: `f_pred = 2*f[i+1] - f[i+2]`
   - Solve optimal assignment with `scipy.optimize.linear_sum_assignment`
   - Handles simple band crossings (where indices swap)

2. **Pass 2 — Anti-crossing detection and reconnection:**
   - Detect: lower band has local max AND upper band has local min at same k
   - Reconnect: swap band assignments for k < k_anti_crossing
   - Handles avoided crossings (bands repel without swapping order)

3. **Pass 3 — Light-cone trimming (TODO):**
   - Set `freqs[i, b] = NaN` wherever `f > k/n_clad` (above light cone)
   - All downstream analysis (ng, single-mode) automatically ignores these points

### Future: Tier 2 (Field Overlap)
- Save D-field via `ms.get_dfield(band)` during MPB run
- Compute overlap matrix `O[p,q] = |<D_{k,p}|D_{k+1,q}>|`
- Cost = 1 - O for Hungarian assignment
- Definitive tracking independent of frequency ordering
- Requires modifying `simplified_gc_3d.py` to save fields (large data)

---

## Structure Geometry (in 2D MPB unit cell)

- **Periodic direction**: x, period = 1a
- **Central spine**: continuous Si ridge, width = 2·h_spine in y
- **Top rib**: Si block, size = (Wt × ht), centered at x = +delta_s/2, y = h_spine + ht/2
- **Bottom rib**: Si block, size = (Wb × hb), centered at x = −delta_s/2, y = −(h_spine + hb/2)
- **Background**: SiO2 (n=1.44)
- **No y-mirror symmetry** → use `ms.run()` (2D), no y-symmetry (3D)

## 3D Structure
- 211nm poly-Si (n=3.48) slab on SiO2 BOX
- SiO2 top cladding (fully clad) → z-mirror symmetry preserved
- TE-like: `ms.run_zeven()` (Ey even under z-reflection)
- TM not computed (only TE needed)

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

## File Structure

```
simplified_structure/
├── simplified_gc_2d.py      ← 2D optimization (complete, 746 cached results)
├── simplified_gc_3d.py      ← 3D validation (3 structures cached)
├── band_unfolding.py        ← band tracking post-processing (anti-crossing + analysis)
├── plan.md                  ← this file
├── cache/                   ← 2D MPB cache (~746 .npz files)
├── cache_3d/                ← 3D MPB cache (3 .npz files)
└── output/                  ← plots, CSV exports
```

---

## Detailed 3D Analysis After Full Pipeline (unfold + trim + analyze)

After light-cone trimming, the guided portions of each band (Structure 0, a=389nm):

| Band | Guided pts | f_range | wl_range (nm) | Monotonic | ng range | Slow light? |
|------|-----------|---------|---------------|-----------|----------|-------------|
| 0 | 32 | [0.186, 0.223] | [1743, 2093] | inc | [1.5, 6.2] | No (outside wl window) |
| 1 | 28 | [0.226, 0.256] | [1522, 1719] | dec | neg | No (backward propagating) |
| 2 | 32 | [0.230, 0.267] | [1459, 1695] | inc | [3.8, 45] | **Yes** BW=9.6nm λc=1532nm |
| 3 | 15 | [0.289, 0.299] | [1303, 1347] | non-mono | mixed | No |
| 4 | 18 | [0.290, 0.309] | [1260, 1342] | inc | [1.8, 57.7] | Has ng∈[5,7] at 1268-1320nm (**outside** 1400-1700 filter) |
| 5 | 5 | [0.332, 0.333] | [1169, 1171] | dec | neg | No (backward propagating) |

**Band 2** is the only candidate near 1550nm but is **MULTIMODE**: Band 1 (guided f up to 0.256)
overlaps Band 2's slow-light window (f starts at 0.245).

**Band 4** has slow light at ~1290nm — monotonic and appears single-mode below light cone, but
far from 1550nm. Would need a ≈ 480nm (389 × 1550/1290) to shift to target wavelength.

---

## Next Steps

1. **Screen 2D cache** with full pipeline (unfold + trim + single-mode check)
   - The 2D simulation uses `ms.run()` (no symmetry) with n_eff=2.85
   - Same anti-crossing issues likely exist in 2D results
   - May find structures where Band 1 and Band 2 are more separated

2. **Try larger a_nm** (400-500nm range) to shift higher bands' slow light to 1550nm
   - Band 4 at a ≈ 480nm might give single-mode slow light near 1550nm
   - Would need new 2D + 3D sweeps

3. **Higher resolution 3D** (resolution=32) for publication-quality results
   - Current results at resolution=16 are for survey only

4. **Tier 2 field overlap** — add `save_fields` to `simplified_gc_3d.py`
   - Required for definitive anti-crossing tracking at close crossings
   - Only needed if Tier 1 gives ambiguous results

---

## CLI Usage

```bash
# 2D optimization
python simplified_gc_2d.py --optimize --save-plots --no-show

# 3D validation
python simplified_gc_3d.py --run --save-plots --no-show

# Band unfolding analysis
python band_unfolding.py --analyze-3d --save-plots --no-show

# Single parameter set
python simplified_gc_2d.py --params a_nm=389 h_spine=0.550 Wt=0.484 ht=0.514 Wb=0.569 hb=0.735 delta_s=0.0 --save-plots
```

---

## Constants

```python
# 2D
RESOLUTION = 32, NUM_BANDS = 8
K_MIN = 0.4, K_MAX = 0.5, K_INTERP = 30
n_eff = 2.85, n_SiO2 = 1.44

# 3D
RESOLUTION = 16 (survey) / 32 (production)
NUM_BANDS = 6, K_MIN = 0.35, K_MAX = 0.50, K_INTERP = 30
N_POLY_SI = 3.48, N_SIO2 = 1.44, T_SLAB_NM = 211.0

# Targets
TARGET_NG = 6.0, NG_LOW = 5.0, NG_HIGH = 7.0
MIN_BW_NM = 5.0, WL_CENTER = 1550.0, MIN_FEAT_NM = 160.0
```
