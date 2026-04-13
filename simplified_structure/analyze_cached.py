"""
analyze_cached.py — Fast analysis of cached Phase-1 sweep results.

Correctly identifies the FLAT slow-light plateau near the band edge
(k ~ 0.47–0.50), NOT the steep transition where ng happens to cross
through the target range.

Strategy:
  1. For band 6 (0-indexed), compute ng(k)
  2. Find the flat plateau near band edge by looking at d(ng)/dk:
     the plateau is where |d(ng)/dk| is small AND k is large (near 0.5)
  3. Measure mean ng in the plateau
  4. Rank configs by how close plateau ng is to target [6, 7]
  5. Analytic alignment to 1550nm via a_nm rescaling

Usage:
  python analyze_cached.py
"""

import hashlib
import itertools
import json
import os

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_POLY_SI  = 3.48
N_SIO2     = 1.44
T_SLAB_NM  = 211.0
K_MIN      = 0.35
K_MAX      = 0.50
T_PARTIAL  = 71.0
WL_TARGET  = 1550.0
NG_LO      = 6.0
NG_HI      = 7.0

A_NM_REF   = 496.0
RESOLUTION = 16
NUM_BANDS  = 30
K_INTERP   = 25

TARGET_BAND = 6   # 0-indexed

H_SPINE_VALS = np.round(np.linspace(0.500, 0.540, 5), 4).tolist()
H_RIB_VALS   = np.round(np.linspace(0.520, 0.580, 4), 4).tolist()
W_RIB_VALS   = np.round(np.linspace(0.400, 0.550, 7), 4).tolist()


def _cache_key(a_nm, h_spine, W_rib, h_rib, resolution, num_bands, k_interp):
    d = dict(a_nm=a_nm, h_spine=h_spine, W_rib=W_rib, h_rib=h_rib,
             t_partial=T_PARTIAL, resolution=resolution,
             num_bands=num_bands, k_min=K_MIN, k_max=K_MAX,
             k_interp=k_interp, parity='yeven')
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def _cache_path(a_nm, h_spine, W_rib, h_rib, resolution, num_bands, k_interp):
    h = _cache_key(a_nm, h_spine, W_rib, h_rib, resolution, num_bands, k_interp)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, 'cache_3d',
                        f"opt_ng67_res{resolution}_nb{num_bands}_{h}.npz")


def compute_ng(f_band, k_x):
    df_dk = np.gradient(f_band, k_x)
    df_dk = np.where(np.abs(df_dk) < 1e-12, np.nan, df_dk)
    return 1.0 / df_dk


def find_flat_plateau(all_freqs, k_x, a_nm, band_idx, k_plateau_min=0.46):
    """Find the flat slow-light plateau near the band edge.

    The flat region is BEFORE the steep divergence at the zone boundary.
    Near k=0.50, ng shoots up steeply. The plateau is the relatively
    constant-ng region at slightly lower k (typically k ~ 0.46–0.49).

    Strategy:
      1. Focus on k > k_plateau_min
      2. Discard the divergent tail: cut where ng > 2× the minimum ng
         in the band-edge region (the steep rise)
      3. The remaining points form the flat plateau

    Returns dict with plateau metrics, or None.
    """
    if band_idx >= all_freqs.shape[1]:
        return None
    f_b = all_freqs[:, band_idx]
    if np.any(f_b <= 1e-6):
        return None

    ng = compute_ng(f_b, k_x)
    with np.errstate(divide='ignore', invalid='ignore'):
        wl = np.where(f_b > 0, a_nm / f_b, np.nan)

    # Focus on k near band edge
    k_mask = k_x >= k_plateau_min
    valid = k_mask & np.isfinite(ng) & (ng > 1.0) & (ng < 500.0)
    if np.sum(valid) < 3:
        return None

    ng_edge = ng[valid]
    k_edge  = k_x[valid]
    wl_edge = wl[valid]

    # Sort by k ascending
    order = np.argsort(k_edge)
    ng_edge = ng_edge[order]
    k_edge  = k_edge[order]
    wl_edge = wl_edge[order]

    # The flat region is before the steep divergence.
    # Find where ng starts climbing steeply: use the minimum ng in this
    # region as the "flat" reference, then keep points within 1.5× of it.
    ng_min = ng_edge.min()
    cutoff = ng_min * 1.5  # anything above this is the divergent tail

    flat_mask = ng_edge <= cutoff
    if np.sum(flat_mask) < 2:
        # Very few flat points — just take the 3 lowest-ng points
        idx_sorted = np.argsort(ng_edge)[:min(4, len(ng_edge))]
        flat_mask = np.zeros(len(ng_edge), dtype=bool)
        flat_mask[idx_sorted] = True

    ng_plat = ng_edge[flat_mask]
    k_plat  = k_edge[flat_mask]
    wl_plat = wl_edge[flat_mask]

    if len(ng_plat) < 2:
        return None

    ng_mean = float(np.mean(ng_plat))
    ng_std  = float(np.std(ng_plat))
    bw_nm   = float(wl_plat.max() - wl_plat.min())
    wl_c    = float((wl_plat.max() + wl_plat.min()) / 2.0)

    # Full-array plateau mask for plotting
    plateau_mask = np.zeros(len(k_x), dtype=bool)
    for i in range(len(k_x)):
        if k_x[i] >= k_plat.min() and k_x[i] <= k_plat.max():
            if ng[i] <= cutoff and np.isfinite(ng[i]):
                plateau_mask[i] = True

    return dict(
        ng_plateau=ng_mean, ng_std=ng_std,
        bw_nm=bw_nm, wl_center=wl_c,
        n_pts=len(ng_plat),
        k_range=(float(k_plat.min()), float(k_plat.max())),
        wl_arr=wl, ng_arr=ng,
        plateau_mask=plateau_mask,
    )


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)

    combos = list(itertools.product(H_SPINE_VALS, H_RIB_VALS, W_RIB_VALS))

    # ------------------------------------------------------------------
    # Load all cached results
    # ------------------------------------------------------------------
    print(f"\n{'='*75}")
    print(f"  Loading {len(combos)} cached sims (a_nm={A_NM_REF}, band={TARGET_BAND})")
    print(f"{'='*75}")

    results = []
    loaded = 0

    for hs, hr, wr in combos:
        cpath = _cache_path(A_NM_REF, hs, wr, hr, RESOLUTION, NUM_BANDS, K_INTERP)
        if not os.path.exists(cpath):
            continue

        z = np.load(cpath, allow_pickle=False)
        all_freqs, k_x = z['all_freqs'], z['k_x']
        loaded += 1

        m = find_flat_plateau(all_freqs, k_x, A_NM_REF, TARGET_BAND)

        results.append(dict(
            h_spine=hs, h_rib=hr, W_rib=wr,
            all_freqs=all_freqs, k_x=k_x,
            metrics=m,
        ))

    print(f"  Loaded: {loaded}/{len(combos)}")

    valid = [r for r in results if r['metrics'] is not None]
    print(f"  Configs with valid plateau: {len(valid)}")

    # ------------------------------------------------------------------
    # Rank: configs where plateau ng is closest to [6, 7] target
    # Priority: ng_plateau in [6, 7] → rank by BW desc, then ng_std asc
    # If ng_plateau not in [6,7], rank by distance to target range
    # ------------------------------------------------------------------
    def sort_key(r):
        m = r['metrics']
        ng = m['ng_plateau']
        in_target = NG_LO <= ng <= NG_HI
        if in_target:
            # In target: rank by BW desc, then flatness (ng_std asc)
            return (0, -m['bw_nm'], m['ng_std'])
        else:
            # Out of target: rank by distance to nearest target boundary
            dist = min(abs(ng - NG_LO), abs(ng - NG_HI))
            return (1, dist, -m['bw_nm'])

    valid.sort(key=sort_key)

    # ------------------------------------------------------------------
    # Print ranking
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  PLATEAU ANALYSIS — band {TARGET_BAND}, a_nm={A_NM_REF}")
    print(f"  Target: flat ng in [{NG_LO}, {NG_HI}]")
    print(f"{'='*80}")
    hdr = (f"  {'#':>3}  {'h_spine':>8}  {'h_rib':>7}  {'W_rib':>7}  "
           f"{'ng_plat':>8}  {'ng_std':>7}  {'BW(nm)':>7}  {'lam_c':>7}  "
           f"{'k_range':>13}  {'n_pts':>5}  {'in[6,7]':>8}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for i, r in enumerate(valid[:25], 1):
        m = r['metrics']
        ng = m['ng_plateau']
        in_tgt = "YES" if NG_LO <= ng <= NG_HI else ""
        klo, khi = m['k_range']
        print(f"  {i:>3}  {r['h_spine']:>8.3f}  {r['h_rib']:>7.3f}  {r['W_rib']:>7.3f}  "
              f"{ng:>8.2f}  {m['ng_std']:>7.3f}  {m['bw_nm']:>7.1f}  {m['wl_center']:>7.1f}  "
              f"  {klo:.3f}-{khi:.3f}  {m['n_pts']:>5}  {in_tgt:>8}")
    print(f"{'='*80}")

    # Count how many are in target
    in_target = [r for r in valid if NG_LO <= r['metrics']['ng_plateau'] <= NG_HI]
    print(f"\n  Configs with plateau ng in [{NG_LO}, {NG_HI}]: {len(in_target)}")

    if not in_target:
        print(f"\n  No configs achieved plateau ng in [{NG_LO}, {NG_HI}].")
        if valid:
            print(f"  Closest configs (showing top-5 nearest to target):")
            for i, r in enumerate(valid[:5], 1):
                m = r['metrics']
                print(f"    #{i}  ng_plateau={m['ng_plateau']:.2f}  "
                      f"hs={r['h_spine']:.3f}  hr={r['h_rib']:.3f}  Wr={r['W_rib']:.3f}  "
                      f"(need {'more' if m['ng_plateau'] < NG_LO else 'less'} corrugation)")
            print(f"\n  RECOMMENDATION: The current sweep range may need expanding.")
            best_ng = valid[0]['metrics']['ng_plateau']
            print(f"  Best plateau ng = {best_ng:.2f}")
            if best_ng < NG_LO:
                print(f"  → Try larger h_rib (>{H_RIB_VALS[-1]}) or smaller h_spine (<{H_SPINE_VALS[0]})")
        else:
            print(f"  No valid plateaus found at all.")

    # ------------------------------------------------------------------
    # Analytic alignment to 1550nm
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  ALIGNED TO 1550nm (analytic rescaling)")
    print(f"{'='*80}")

    aligned = []
    for r in valid:
        m = r['metrics']
        wl_c = m['wl_center']
        if wl_c < 100 or wl_c > 5000:
            continue
        a_new = A_NM_REF * (WL_TARGET / wl_c)
        scale = a_new / A_NM_REF

        aligned.append(dict(
            h_spine=r['h_spine'], h_rib=r['h_rib'], W_rib=r['W_rib'],
            a_nm=round(a_new, 1),
            band=TARGET_BAND,
            bw_nm=round(m['bw_nm'] * scale, 2),
            wl_center=round(wl_c * scale, 1),
            ng_plateau=m['ng_plateau'],
            ng_std=m['ng_std'],
            n_pts=m['n_pts'],
            k_range=m['k_range'],
            all_freqs=r['all_freqs'],
            k_x=r['k_x'],
        ))

    # Same sort order
    def sort_key_aligned(a):
        ng = a['ng_plateau']
        in_target = NG_LO <= ng <= NG_HI
        if in_target:
            return (0, -a['bw_nm'], a['ng_std'])
        else:
            dist = min(abs(ng - NG_LO), abs(ng - NG_HI))
            return (1, dist, -a['bw_nm'])

    aligned.sort(key=sort_key_aligned)

    hdr2 = (f"  {'#':>3}  {'h_spine':>8}  {'h_rib':>7}  {'W_rib':>7}  "
            f"{'a_nm':>7}  {'ng_plat':>8}  {'ng_std':>7}  {'BW(nm)':>7}  "
            f"{'lam_c':>7}  {'in[6,7]':>8}")
    print(hdr2)
    print('  ' + '-' * (len(hdr2) - 2))
    for i, a in enumerate(aligned[:25], 1):
        ng = a['ng_plateau']
        in_tgt = "YES" if NG_LO <= ng <= NG_HI else ""
        print(f"  {i:>3}  {a['h_spine']:>8.3f}  {a['h_rib']:>7.3f}  {a['W_rib']:>7.3f}  "
              f"{a['a_nm']:>7.1f}  {ng:>8.2f}  {a['ng_std']:>7.3f}  {a['bw_nm']:>7.1f}  "
              f"{a['wl_center']:>7.1f}  {in_tgt:>8}")
    print(f"{'='*80}")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    save_data = {
        'h_spine_arr':    np.array([a['h_spine'] for a in aligned]),
        'h_rib_arr':      np.array([a['h_rib'] for a in aligned]),
        'W_rib_arr':      np.array([a['W_rib'] for a in aligned]),
        'a_nm_arr':       np.array([a['a_nm'] for a in aligned]),
        'band_arr':       np.array([a['band'] for a in aligned]),
        'bw_arr':         np.array([a['bw_nm'] for a in aligned]),
        'ng_plateau_arr': np.array([a['ng_plateau'] for a in aligned]),
        'ng_std_arr':     np.array([a['ng_std'] for a in aligned]),
        'wl_center_arr':  np.array([a['wl_center'] for a in aligned]),
    }
    for i, a in enumerate(aligned[:5]):
        save_data[f'top{i}_all_freqs'] = a['all_freqs']
        save_data[f'top{i}_k_x']       = a['k_x']
        save_data[f'top{i}_a_nm']      = a['a_nm']

    npz_path = os.path.join(out_dir, 'optimize_ng6to7_results.npz')
    np.savez_compressed(npz_path, **save_data)
    print(f"\n  Saved: {npz_path}")

    json_data = []
    for a in aligned:
        json_data.append(dict(
            rank=len(json_data) + 1,
            h_spine=a['h_spine'], h_rib=a['h_rib'], W_rib=a['W_rib'],
            a_nm=a['a_nm'], band=a['band'],
            bw_nm=a['bw_nm'],
            ng_plateau=round(a['ng_plateau'], 3),
            ng_std=round(a['ng_std'], 4),
            wl_center=a['wl_center'],
        ))
    json_path = os.path.join(out_dir, 'optimize_ng6to7_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_path}")

    # Winner
    if aligned:
        w = aligned[0]
        print(f"\n{'='*75}")
        print(f"  BEST CONFIGURATION")
        print(f"{'='*75}")
        print(f"    a_nm      = {w['a_nm']:.1f} nm")
        print(f"    h_spine   = {w['h_spine']:.4f}  ({w['h_spine']*w['a_nm']:.1f} nm)")
        print(f"    W_rib     = {w['W_rib']:.4f}  ({w['W_rib']*w['a_nm']:.1f} nm)  [Wt=Wb]")
        print(f"    h_rib     = {w['h_rib']:.4f}  ({w['h_rib']*w['a_nm']:.1f} nm)  [ht=hb]")
        print(f"    t_PE      = 71.0 nm")
        print(f"    band      = {w['band']}  (y-even / TE0)")
        print(f"    ng_plateau = {w['ng_plateau']:.3f}")
        print(f"    ng_std    = {w['ng_std']:.4f}")
        print(f"    BW        = {w['bw_nm']:.1f} nm  (plateau)")
        print(f"    lam_c     = {w['wl_center']:.1f} nm")
        ng = w['ng_plateau']
        if NG_LO <= ng <= NG_HI:
            print(f"    STATUS    = IN TARGET  ng in [{NG_LO}, {NG_HI}]")
        else:
            print(f"    STATUS    = OUTSIDE TARGET (ng={ng:.2f}, need [{NG_LO}, {NG_HI}])")
        print(f"{'='*75}\n")


if __name__ == '__main__':
    main()
