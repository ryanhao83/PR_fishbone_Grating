"""
optimize_ng6to7.py — Systematic geometry sweep to achieve flat ng in [6, 7]
                     for the 71nm partial-etch y-symmetric fishbone grating.

Strategy:
  1. Decrease h_spine (0.500–0.540) → pushes mode outward, increases grating coupling
  2. Increase h_rib   (0.520–0.580) → deeper corrugation, stronger slow-light effect
  3. Sweep W_rib      (0.400–0.550) → duty-cycle tuning for dispersion compensation
  4. Auto-align a_nm so the flat ng=[6,7] window centres on 1550nm
  5. Rank by bandwidth where 6 < ng < 7

Uses run_yeven() from run_71nm_yeven.py for TE0 parity filtering.
All results cached as .npz in cache_3d/.

Usage:
  python optimize_ng6to7.py
  python optimize_ng6to7.py --resolution 16 --num-bands 30 --k-interp 25
"""

import hashlib
import itertools
import json
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# Constants (same as sweep_ysym_partial_etch.py)
# ---------------------------------------------------------------------------
N_POLY_SI  = 3.48
N_SIO2     = 1.44
T_SLAB_NM  = 211.0
PAD_Y      = 2.0
PAD_Z      = 1.5
K_MIN      = 0.35
K_MAX      = 0.50
T_PARTIAL  = 71.0    # nm, fixed

WL_TARGET  = 1550.0  # nm
NG_LO      = 6.0
NG_HI      = 7.0

# ---------------------------------------------------------------------------
# Sweep grids
# ---------------------------------------------------------------------------
H_SPINE_VALS = np.round(np.linspace(0.500, 0.540, 5), 4).tolist()
H_RIB_VALS   = np.round(np.linspace(0.520, 0.580, 4), 4).tolist()
W_RIB_VALS   = np.round(np.linspace(0.400, 0.550, 7), 4).tolist()

# Defaults
RESOLUTION = 16
NUM_BANDS  = 30
K_INTERP   = 25


# ---------------------------------------------------------------------------
# Geometry builder  (identical to run_71nm_yeven.py but avoids top-level meep)
# ---------------------------------------------------------------------------
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
                 center=mp.Vector3(0, 0, -(t_slab / 2 + PAD_Z / 2)),
                 material=SiO2),
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, PAD_Z),
                 center=mp.Vector3(0, 0, t_slab / 2 + PAD_Z / 2),
                 material=SiO2),
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, t_partial),
                 center=mp.Vector3(0, 0, -t_slab / 2 + t_partial / 2),
                 material=Si),
        mp.Block(size=mp.Vector3(mp.inf, 2.0 * h_spine, t_slab),
                 center=mp.Vector3(0, 0, 0),
                 material=Si),
        mp.Block(size=mp.Vector3(W_rib, h_rib, t_slab),
                 center=mp.Vector3(0, h_spine + h_rib / 2.0, 0),
                 material=Si),
        mp.Block(size=mp.Vector3(W_rib, h_rib, t_slab),
                 center=mp.Vector3(0, -(h_spine + h_rib / 2.0), 0),
                 material=Si),
    ]
    return lattice, geometry


# ---------------------------------------------------------------------------
# MPB runner — y-even parity (TE0 sector)
# ---------------------------------------------------------------------------
def run_mpb(a_nm, h_spine, W_rib, h_rib, resolution, num_bands, k_interp):
    import meep as mp
    import meep.mpb as mpb

    lattice, geometry = _build_geometry(a_nm, h_spine, W_rib, h_rib)
    k_points = mp.interpolate(k_interp, [mp.Vector3(K_MIN), mp.Vector3(K_MAX)])

    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=k_points,
        resolution=resolution,
        num_bands=num_bands,
    )
    ms.run_yeven()

    all_freqs = np.array(ms.all_freqs)
    k_x = np.array([k.x for k in k_points])
    return all_freqs, k_x


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def _cache_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_3d')
    os.makedirs(d, exist_ok=True)
    return d


def _cache_key(a_nm, h_spine, W_rib, h_rib, resolution, num_bands, k_interp):
    d = dict(a_nm=a_nm, h_spine=h_spine, W_rib=W_rib, h_rib=h_rib,
             t_partial=T_PARTIAL, resolution=resolution,
             num_bands=num_bands, k_min=K_MIN, k_max=K_MAX,
             k_interp=k_interp, parity='yeven')
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def _cache_path(a_nm, h_spine, W_rib, h_rib, resolution, num_bands, k_interp):
    h = _cache_key(a_nm, h_spine, W_rib, h_rib, resolution, num_bands, k_interp)
    return os.path.join(_cache_dir(),
                        f"opt_ng67_res{resolution}_nb{num_bands}_{h}.npz")


def _save(path, all_freqs, k_x, a_nm, h_spine, W_rib, h_rib, resolution, num_bands):
    np.savez_compressed(path, all_freqs=all_freqs, k_x=k_x,
                        a_nm=a_nm, h_spine=h_spine, W_rib=W_rib, h_rib=h_rib,
                        resolution=resolution, num_bands=num_bands)


def _load(path):
    z = np.load(path, allow_pickle=False)
    return z['all_freqs'], z['k_x']


# ---------------------------------------------------------------------------
# Analysis: compute ng for a single band, measure flat-ng BW
# ---------------------------------------------------------------------------
def compute_ng(f_band, k_x):
    df_dk = np.gradient(f_band, k_x)
    df_dk = np.where(np.abs(df_dk) < 1e-12, np.nan, df_dk)
    return 1.0 / df_dk


def measure_flat_ng(all_freqs, k_x, a_nm, band_idx, ng_lo=NG_LO, ng_hi=NG_HI):
    """Measure bandwidth where ng ∈ [ng_lo, ng_hi] for a given band.

    Returns dict with: bw_nm, wl_center, ng_mean, ng_std, n_pts, wl_arr, ng_arr
    or None if fewer than 2 points qualify.
    """
    if band_idx >= all_freqs.shape[1]:
        return None
    f_b = all_freqs[:, band_idx]
    if np.any(f_b <= 1e-6):
        return None

    ng = compute_ng(f_b, k_x)
    with np.errstate(divide='ignore', invalid='ignore'):
        wl = np.where(f_b > 0, a_nm / f_b, np.nan)

    mask = (np.isfinite(ng) & (ng >= ng_lo) & (ng <= ng_hi)
            & np.isfinite(wl) & (wl > 1300) & (wl < 1800))

    if np.sum(mask) < 2:
        return None

    bw = wl[mask].max() - wl[mask].min()
    wl_c = (wl[mask].max() + wl[mask].min()) / 2.0
    ng_mean = float(np.nanmean(ng[mask]))
    ng_std  = float(np.nanstd(ng[mask]))

    return dict(bw_nm=bw, wl_center=wl_c, ng_mean=ng_mean, ng_std=ng_std,
                n_pts=int(np.sum(mask)), wl_arr=wl, ng_arr=ng)


def auto_align_a(all_freqs, k_x, a_nm_init, band_idx):
    """Calculate a_nm that centres the flat-ng window on 1550nm.

    Logic: find the slow-light window at the initial a_nm, compute its
    centre wavelength, then rescale a_nm proportionally.
    """
    m = measure_flat_ng(all_freqs, k_x, a_nm_init, band_idx)
    if m is None:
        return a_nm_init  # can't align — return unchanged
    wl_c = m['wl_center']
    if wl_c < 100 or wl_c > 5000:
        return a_nm_init
    # f_norm = a / wl is invariant; new_a = old_a * (target_wl / old_wl)
    return round(a_nm_init * (WL_TARGET / wl_c), 1)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=RESOLUTION)
    parser.add_argument('--num-bands',  type=int, default=NUM_BANDS)
    parser.add_argument('--k-interp',   type=int, default=K_INTERP)
    parser.add_argument('--band-lo',    type=int, default=5,
                        help='lowest band index to scan for ng window')
    parser.add_argument('--band-hi',    type=int, default=9,
                        help='highest band index to scan')
    args = parser.parse_args()

    res = args.resolution
    nb  = args.num_bands
    ki  = args.k_interp
    band_range = range(args.band_lo, args.band_hi + 1)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)

    combos = list(itertools.product(H_SPINE_VALS, H_RIB_VALS, W_RIB_VALS))
    total  = len(combos)

    print(f"\n{'='*75}")
    print(f"  OPTIMIZE ng ∈ [{NG_LO}, {NG_HI}] — 71nm PE, y-even parity")
    print(f"  {total} geometry combos × bands {list(band_range)}")
    print(f"  h_spine: {H_SPINE_VALS}")
    print(f"  h_rib  : {H_RIB_VALS}")
    print(f"  W_rib  : {W_RIB_VALS}")
    print(f"  res={res}  nb={nb}  k-interp={ki}")
    print(f"{'='*75}\n")

    # Use a reference a_nm for the initial sweep (will auto-align later)
    a_nm_ref = 496.0

    results = []
    t0 = time.time()

    for idx, (hs, hr, wr) in enumerate(combos):
        tag = f"[{idx+1:3d}/{total}] hs={hs:.3f} hr={hr:.3f} Wr={wr:.3f}"
        cpath = _cache_path(a_nm_ref, hs, wr, hr, res, nb, ki)

        if os.path.exists(cpath):
            all_freqs, k_x = _load(cpath)
            cached = True
        else:
            all_freqs, k_x = run_mpb(a_nm_ref, hs, wr, hr, res, nb, ki)
            _save(cpath, all_freqs, k_x, a_nm_ref, hs, wr, hr, res, nb)
            cached = False

        # Scan bands for best ng=[6,7] window
        best_m = None
        best_b = None
        for b in band_range:
            m = measure_flat_ng(all_freqs, k_x, a_nm_ref, b)
            if m is not None:
                if best_m is None or m['bw_nm'] > best_m['bw_nm']:
                    best_m = m
                    best_b = b

        if best_m is not None:
            bw_str = f"BW={best_m['bw_nm']:5.1f}nm  ng={best_m['ng_mean']:.2f}  band={best_b}"
        else:
            bw_str = "no ng window"

        src = "cache" if cached else "  MPB"
        print(f"  {tag}  [{src}]  {bw_str}")

        results.append(dict(
            h_spine=hs, h_rib=hr, W_rib=wr,
            a_nm_ref=a_nm_ref,
            all_freqs=all_freqs, k_x=k_x,
            best_band=best_b, metrics=best_m,
        ))

    elapsed = time.time() - t0
    print(f"\n  Sweep completed in {elapsed:.0f}s")

    # ------------------------------------------------------------------
    # Rank by BW descending, ng_std ascending
    # ------------------------------------------------------------------
    valid = [r for r in results if r['metrics'] is not None]
    valid.sort(key=lambda r: (-r['metrics']['bw_nm'], r['metrics']['ng_std']))

    # ------------------------------------------------------------------
    # Phase 2: auto-align top-20 to 1550nm and re-run
    # ------------------------------------------------------------------
    top_n = min(20, len(valid))
    print(f"\n{'='*75}")
    print(f"  PHASE 2: Auto-align top {top_n} configs to 1550nm")
    print(f"{'='*75}\n")

    aligned = []
    for rank, r in enumerate(valid[:top_n], 1):
        hs, hr, wr = r['h_spine'], r['h_rib'], r['W_rib']
        b = r['best_band']

        # Compute aligned period
        a_new = auto_align_a(r['all_freqs'], r['k_x'], a_nm_ref, b)

        # Re-run at aligned period
        cpath2 = _cache_path(a_new, hs, wr, hr, res, nb, ki)
        if os.path.exists(cpath2):
            all_freqs2, k_x2 = _load(cpath2)
        else:
            all_freqs2, k_x2 = run_mpb(a_new, hs, wr, hr, res, nb, ki)
            _save(cpath2, all_freqs2, k_x2, a_new, hs, wr, hr, res, nb)

        m2 = measure_flat_ng(all_freqs2, k_x2, a_new, b)

        if m2 is not None:
            print(f"  #{rank:2d}  hs={hs:.3f} hr={hr:.3f} Wr={wr:.3f}  "
                  f"a={a_new:.1f}nm  band={b}  "
                  f"BW={m2['bw_nm']:.1f}nm  ng={m2['ng_mean']:.2f}±{m2['ng_std']:.3f}  "
                  f"λc={m2['wl_center']:.1f}nm")
        else:
            print(f"  #{rank:2d}  hs={hs:.3f} hr={hr:.3f} Wr={wr:.3f}  "
                  f"a={a_new:.1f}nm  band={b}  -- lost ng window after align --")

        aligned.append(dict(
            rank=rank, h_spine=hs, h_rib=hr, W_rib=wr,
            a_nm=a_new, band=b,
            all_freqs=all_freqs2, k_x=k_x2,
            metrics=m2,
        ))

    # ------------------------------------------------------------------
    # Final ranking of aligned results
    # ------------------------------------------------------------------
    aligned_valid = [a for a in aligned if a['metrics'] is not None]
    aligned_valid.sort(key=lambda a: (-a['metrics']['bw_nm'], a['metrics']['ng_std']))

    print(f"\n{'='*75}")
    print(f"  FINAL RANKING (aligned to 1550nm)")
    print(f"{'='*75}")
    hdr = (f"  {'#':>3}  {'h_spine':>8}  {'h_rib':>7}  {'W_rib':>7}  "
           f"{'a_nm':>7}  {'band':>5}  {'λc(nm)':>8}  {'ng':>6}  {'σ(ng)':>7}  {'BW(nm)':>7}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))

    for i, a in enumerate(aligned_valid, 1):
        m = a['metrics']
        print(f"  {i:>3}  {a['h_spine']:>8.3f}  {a['h_rib']:>7.3f}  {a['W_rib']:>7.3f}  "
              f"{a['a_nm']:>7.1f}  {a['band']:>5}  {m['wl_center']:>8.1f}  "
              f"{m['ng_mean']:>6.2f}  {m['ng_std']:>7.3f}  {m['bw_nm']:>7.1f}")
        if i >= 15:
            break

    print(f"{'='*75}")

    # ------------------------------------------------------------------
    # Save full sweep data for later analysis
    # ------------------------------------------------------------------
    save_path = os.path.join(out_dir, 'optimize_ng6to7_results.npz')

    # Flatten aligned results for saving
    save_data = {
        'h_spine_arr':  np.array([a['h_spine'] for a in aligned_valid]),
        'h_rib_arr':    np.array([a['h_rib'] for a in aligned_valid]),
        'W_rib_arr':    np.array([a['W_rib'] for a in aligned_valid]),
        'a_nm_arr':     np.array([a['a_nm'] for a in aligned_valid]),
        'band_arr':     np.array([a['band'] for a in aligned_valid]),
        'bw_arr':       np.array([a['metrics']['bw_nm'] for a in aligned_valid]),
        'ng_mean_arr':  np.array([a['metrics']['ng_mean'] for a in aligned_valid]),
        'ng_std_arr':   np.array([a['metrics']['ng_std'] for a in aligned_valid]),
        'wl_center_arr': np.array([a['metrics']['wl_center'] for a in aligned_valid]),
    }

    # Save the top-5 full band data for the notebook
    for i, a in enumerate(aligned_valid[:5]):
        save_data[f'top{i}_all_freqs'] = a['all_freqs']
        save_data[f'top{i}_k_x']       = a['k_x']

    np.savez_compressed(save_path, **save_data)
    print(f"\n  Saved results to: {save_path}")

    # Also save as JSON for easy reading
    json_path = os.path.join(out_dir, 'optimize_ng6to7_results.json')
    json_data = []
    for a in aligned_valid:
        m = a['metrics']
        json_data.append(dict(
            rank=len(json_data) + 1,
            h_spine=a['h_spine'], h_rib=a['h_rib'], W_rib=a['W_rib'],
            a_nm=a['a_nm'], band=a['band'],
            bw_nm=round(m['bw_nm'], 2),
            ng_mean=round(m['ng_mean'], 3),
            ng_std=round(m['ng_std'], 4),
            wl_center=round(m['wl_center'], 1),
        ))
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved JSON to:    {json_path}")

    # ------------------------------------------------------------------
    # Print the winner
    # ------------------------------------------------------------------
    if aligned_valid:
        w = aligned_valid[0]
        m = w['metrics']
        print(f"\n{'='*75}")
        print(f"  BEST CONFIGURATION")
        print(f"{'='*75}")
        print(f"    a_nm    = {w['a_nm']:.1f} nm")
        print(f"    h_spine = {w['h_spine']:.4f}  ({w['h_spine']*w['a_nm']:.1f} nm)")
        print(f"    W_rib   = {w['W_rib']:.4f}  ({w['W_rib']*w['a_nm']:.1f} nm)")
        print(f"    h_rib   = {w['h_rib']:.4f}  ({w['h_rib']*w['a_nm']:.1f} nm)")
        print(f"    band    = {w['band']}  (y-even sector)")
        print(f"    ng_mean = {m['ng_mean']:.3f}")
        print(f"    ng_std  = {m['ng_std']:.4f}")
        print(f"    BW(nm)  = {m['bw_nm']:.1f} nm  (ng ∈ [{NG_LO}, {NG_HI}])")
        print(f"    λ_center = {m['wl_center']:.1f} nm")
        print(f"{'='*75}\n")
    else:
        print("\n  WARNING: No configuration achieved ng ∈ [6, 7].\n")


if __name__ == '__main__':
    main()
