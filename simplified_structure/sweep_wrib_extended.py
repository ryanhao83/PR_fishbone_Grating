"""
sweep_wrib_extended.py — Quick targeted sweep: push W_rib higher to reach
                         flat ng=6-7 on band 6.

Previous sweep found: plateau ng maxes at ~5.5 with W_rib=0.550 (the limit).
W_rib is the dominant lever. h_spine/h_rib have weak effect → narrow range.

Fab constraint at a=496nm: gap = (1-W_rib)*a > 160nm → W_rib < 0.677

Sweep:
  W_rib:   0.55, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67  (7 values)
  h_spine: 0.51, 0.53, 0.55                             (3 values)
  h_rib:   0.52, 0.54, 0.56                             (3 values)
  Total: 63 combos, num_bands=8 for speed

Usage:
  python sweep_wrib_extended.py
"""

import hashlib
import itertools
import json
import os
import time

import numpy as np

# Constants
N_POLY_SI  = 3.48
N_SIO2     = 1.44
T_SLAB_NM  = 211.0
PAD_Y      = 2.0
PAD_Z      = 1.5
K_MIN      = 0.35
K_MAX      = 0.50
T_PARTIAL  = 71.0
A_NM       = 496.0
WL_TARGET  = 1550.0
TARGET_BAND = 6

RESOLUTION = 16
NUM_BANDS  = 8
K_INTERP   = 19   # 21 k-points — enough for ng estimation

# Sweep grids
W_RIB_VALS   = [0.55, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67]
H_SPINE_VALS = [0.51, 0.53, 0.55]
H_RIB_VALS   = [0.52, 0.54, 0.56]


def _build_geometry(a_nm, h_spine, W_rib, h_rib):
    import meep as mp
    t_slab    = T_SLAB_NM / a_nm
    t_partial = T_PARTIAL / a_nm
    sy = 2.0 * (h_spine + h_rib) + 2.0 * PAD_Y
    sz = t_slab + 2.0 * PAD_Z
    Si   = mp.Medium(index=N_POLY_SI)
    SiO2 = mp.Medium(index=N_SIO2)
    lattice = mp.Lattice(size=mp.Vector3(1, sy, sz))
    geometry = [
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, PAD_Z),
                 center=mp.Vector3(0, 0, -(t_slab/2 + PAD_Z/2)), material=SiO2),
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, PAD_Z),
                 center=mp.Vector3(0, 0, t_slab/2 + PAD_Z/2), material=SiO2),
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, t_partial),
                 center=mp.Vector3(0, 0, -t_slab/2 + t_partial/2), material=Si),
        mp.Block(size=mp.Vector3(mp.inf, 2.0*h_spine, t_slab),
                 center=mp.Vector3(0, 0, 0), material=Si),
        mp.Block(size=mp.Vector3(W_rib, h_rib, t_slab),
                 center=mp.Vector3(0, h_spine + h_rib/2, 0), material=Si),
        mp.Block(size=mp.Vector3(W_rib, h_rib, t_slab),
                 center=mp.Vector3(0, -(h_spine + h_rib/2), 0), material=Si),
    ]
    return lattice, geometry


def run_mpb(a_nm, h_spine, W_rib, h_rib):
    import meep as mp
    import meep.mpb as mpb
    lattice, geometry = _build_geometry(a_nm, h_spine, W_rib, h_rib)
    k_points = mp.interpolate(K_INTERP, [mp.Vector3(K_MIN), mp.Vector3(K_MAX)])
    ms = mpb.ModeSolver(geometry_lattice=lattice, geometry=geometry,
                        k_points=k_points, resolution=RESOLUTION,
                        num_bands=NUM_BANDS)
    ms.run_yeven()
    return np.array(ms.all_freqs), np.array([k.x for k in k_points])


def _cache_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_3d')
    os.makedirs(d, exist_ok=True)
    return d


def _cache_path(hs, wr, hr):
    d = dict(a_nm=A_NM, h_spine=hs, W_rib=wr, h_rib=hr, t_partial=T_PARTIAL,
             resolution=RESOLUTION, num_bands=NUM_BANDS,
             k_min=K_MIN, k_max=K_MAX, k_interp=K_INTERP, parity='yeven')
    h = hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]
    return os.path.join(_cache_dir(), f"wrib_ext_res{RESOLUTION}_nb{NUM_BANDS}_{h}.npz")


def compute_ng(fb, kx):
    df = np.gradient(fb, kx)
    df = np.where(np.abs(df) < 1e-12, np.nan, df)
    return 1.0 / df


def analyze_plateau(all_freqs, k_x, band=TARGET_BAND):
    """Find flat ng region on given band before the divergence.

    Strategy: walk from HIGH-k (band edge) backward.
    1. Select points with k >= 0.44, finite ng > 1.
    2. Starting from the high-k end, skip the divergent tail (ng > 8).
    3. Walk backward (toward lower k), collecting points where ng stays
       within 1.5× the minimum ng seen so far (i.e. the flat plateau).
    4. Stop when ng jumps below 60% or above 150% of running min, or
       when we hit a band-crossing spike (ng > 8).
    """
    if band >= all_freqs.shape[1]:
        return None
    fb = all_freqs[:, band]
    if np.any(fb <= 1e-6):
        return None

    ng = compute_ng(fb, k_x)
    wl = A_NM / fb

    idx = np.where((k_x >= 0.44) & np.isfinite(ng) & (ng > 1) & (ng < 500))[0]
    if len(idx) < 3:
        return None

    ng_sel = ng[idx]
    kx_sel = k_x[idx]
    wl_sel = wl[idx]

    # Walk from high-k end backward, skip divergent tail (ng > 8)
    n = len(ng_sel)
    start = n - 1
    while start >= 0 and ng_sel[start] > 8:
        start -= 1
    if start < 2:
        return None

    # Collect flat plateau points walking backward from start
    flat_indices = [start]
    ng_min = ng_sel[start]
    for j in range(start - 1, -1, -1):
        v = ng_sel[j]
        # Stop if we hit a spike (band crossing) or divergence
        if v > 8 or v < 1:
            break
        # Stop if ng drops below 60% or rises above 150% of plateau min
        if v < 0.6 * ng_min or v > 1.5 * ng_min:
            break
        ng_min = min(ng_min, v)
        flat_indices.append(j)

    flat_indices = sorted(flat_indices)
    if len(flat_indices) < 2:
        return None

    ng_flat = ng_sel[flat_indices]
    wl_flat = wl_sel[flat_indices]
    kx_flat = kx_sel[flat_indices]

    return dict(
        ng_mean=float(np.mean(ng_flat)),
        ng_std=float(np.std(ng_flat)),
        n_pts=len(ng_flat),
        bw_nm=float(wl_flat.max() - wl_flat.min()),
        wl_c=float((wl_flat.max() + wl_flat.min()) / 2.0),
        k_lo=float(kx_flat[0]), k_hi=float(kx_flat[-1]),
        ng_profile=[float(x) for x in ng_flat],
        ng_all=ng, wl_all=wl,
    )


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)

    combos = list(itertools.product(H_SPINE_VALS, H_RIB_VALS, W_RIB_VALS))
    total = len(combos)

    print(f"\n{'='*75}")
    print(f"  Extended W_rib sweep — band {TARGET_BAND}, y-even, nb={NUM_BANDS}")
    print(f"  {total} combos | W_rib: {W_RIB_VALS}")
    print(f"  Fab check: gap=(1-Wr)*{A_NM:.0f}nm > 160nm")
    print(f"{'='*75}\n")

    results = []
    t0 = time.time()

    for idx, (hs, hr, wr) in enumerate(combos):
        gap_nm = (1.0 - wr) * A_NM
        rib_nm = wr * A_NM
        tag = f"[{idx+1:3d}/{total}] hs={hs:.2f} hr={hr:.2f} Wr={wr:.2f}"

        if gap_nm < 160 or rib_nm < 160:
            print(f"  {tag}  SKIP (gap={gap_nm:.0f}nm)")
            continue

        cpath = _cache_path(hs, wr, hr)
        if os.path.exists(cpath):
            z = np.load(cpath, allow_pickle=False)
            all_freqs, k_x = z['all_freqs'], z['k_x']
            src = "cache"
        else:
            all_freqs, k_x = run_mpb(A_NM, hs, wr, hr)
            np.savez_compressed(cpath, all_freqs=all_freqs, k_x=k_x)
            src = "  MPB"

        m = analyze_plateau(all_freqs, k_x)
        if m:
            prof = ' '.join([f'{x:.1f}' for x in m['ng_profile']])
            print(f"  {tag}  [{src}]  ng={m['ng_mean']:.2f}  BW={m['bw_nm']:.1f}nm  [{prof}]")
        else:
            print(f"  {tag}  [{src}]  no flat region")

        results.append(dict(hs=hs, hr=hr, wr=wr,
                            all_freqs=all_freqs, k_x=k_x, metrics=m))

    dt = time.time() - t0
    print(f"\n  Done in {dt:.0f}s")

    # Rank
    valid = [r for r in results if r['metrics'] is not None]
    valid.sort(key=lambda r: -r['metrics']['ng_mean'])

    print(f"\n{'='*80}")
    print(f"  TOP 10 — band {TARGET_BAND}, ranked by highest flat-plateau ng")
    print(f"{'='*80}")
    print(f"{'Rk':>3} {'hs':>5} {'hr':>5} {'Wr':>5} {'gap':>5} {'ng':>6} {'std':>6} {'pts':>3} "
          f"{'BW':>5} {'wl_c':>6} {'in[6,7]':>7}  ng profile")
    print('-'*95)

    for i, r in enumerate(valid[:10], 1):
        m = r['metrics']
        gap = (1 - r['wr']) * A_NM
        in_tgt = " YES" if 6.0 <= m['ng_mean'] <= 7.0 else ""
        prof = ' '.join([f'{x:.1f}' for x in m['ng_profile']])
        print(f"{i:>3} {r['hs']:>5.2f} {r['hr']:>5.2f} {r['wr']:>5.2f} {gap:>5.0f} "
              f"{m['ng_mean']:>6.2f} {m['ng_std']:>6.3f} {m['n_pts']:>3} "
              f"{m['bw_nm']:>5.1f} {m['wl_c']:>6.1f} {in_tgt:>7}  [{prof}]")

    # Save
    save_results = []
    for r in valid:
        m = r['metrics']
        a_aligned = round(A_NM * (WL_TARGET / m['wl_c']), 1) if m['wl_c'] > 100 else A_NM
        save_results.append(dict(
            h_spine=r['hs'], h_rib=r['hr'], W_rib=r['wr'],
            a_nm_ref=A_NM, a_nm_aligned=a_aligned,
            gap_nm=round((1-r['wr'])*A_NM, 1),
            ng_plateau=round(m['ng_mean'], 3),
            ng_std=round(m['ng_std'], 4),
            bw_nm=round(m['bw_nm'], 1),
            wl_center=round(m['wl_c'], 1),
            n_pts=m['n_pts'],
        ))

    json_path = os.path.join(out_dir, 'sweep_wrib_extended.json')
    with open(json_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # Save top-5 band data for notebook
    save_data = {}
    for i, r in enumerate(valid[:5]):
        save_data[f'top{i}_all_freqs'] = r['all_freqs']
        save_data[f'top{i}_k_x'] = r['k_x']
        m = r['metrics']
        a_al = round(A_NM * (WL_TARGET / m['wl_c']), 1)
        save_data[f'top{i}_a_nm'] = a_al
        save_data[f'top{i}_hs'] = r['hs']
        save_data[f'top{i}_hr'] = r['hr']
        save_data[f'top{i}_wr'] = r['wr']
    npz_path = os.path.join(out_dir, 'sweep_wrib_extended.npz')
    np.savez_compressed(npz_path, **save_data)
    print(f"  Saved: {npz_path}")

    # Winner
    if valid:
        w = valid[0]
        m = w['metrics']
        a_al = round(A_NM * (WL_TARGET / m['wl_c']), 1)
        gap = (1 - w['wr']) * A_NM
        print(f"\n{'='*75}")
        print(f"  BEST CONFIG")
        print(f"{'='*75}")
        print(f"    a_nm (aligned) = {a_al} nm")
        print(f"    h_spine = {w['hs']:.3f}  ({w['hs']*a_al:.1f} nm)")
        print(f"    W_rib   = {w['wr']:.3f}  ({w['wr']*a_al:.1f} nm)  gap={(1-w['wr'])*a_al:.1f}nm")
        print(f"    h_rib   = {w['hr']:.3f}  ({w['hr']*a_al:.1f} nm)")
        print(f"    ng_plateau = {m['ng_mean']:.3f} +/- {m['ng_std']:.3f}")
        print(f"    BW = {m['bw_nm']:.1f} nm")
        ng = m['ng_mean']
        if 6.0 <= ng <= 7.0:
            print(f"    STATUS: IN TARGET [6, 7]")
        else:
            print(f"    STATUS: ng={ng:.2f}, need [6, 7]  (gap={6.0-ng:+.2f})")
        print(f"{'='*75}\n")


if __name__ == '__main__':
    main()
