"""
analyze_and_align.py — Load cached Phase-1 results from optimize_ng6to7.py,
                       rank them, auto-align top-5 to 1550nm, save outputs.

Usage:
  python analyze_and_align.py
  python analyze_and_align.py --top-n 10 --timeout 300
"""

import hashlib
import itertools
import json
import os
import signal
import time

import numpy as np

# Import sweep parameters and helpers from optimize_ng6to7
from optimize_ng6to7 import (
    N_POLY_SI, N_SIO2, T_SLAB_NM, PAD_Y, PAD_Z,
    K_MIN, K_MAX, T_PARTIAL, WL_TARGET, NG_LO, NG_HI,
    H_SPINE_VALS, H_RIB_VALS, W_RIB_VALS,
    RESOLUTION, NUM_BANDS, K_INTERP,
    _cache_path, _load, _save,
    run_mpb, compute_ng, measure_flat_ng, auto_align_a,
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=RESOLUTION)
    parser.add_argument('--num-bands',  type=int, default=NUM_BANDS)
    parser.add_argument('--k-interp',   type=int, default=K_INTERP)
    parser.add_argument('--top-n',      type=int, default=5,
                        help='Number of top configs to re-run with aligned a_nm')
    parser.add_argument('--timeout',    type=int, default=600,
                        help='Max seconds per Phase-2 MPB run before skipping')
    parser.add_argument('--band-lo',    type=int, default=5)
    parser.add_argument('--band-hi',    type=int, default=9)
    args = parser.parse_args()

    res = args.resolution
    nb  = args.num_bands
    ki  = args.k_interp
    band_range = range(args.band_lo, args.band_hi + 1)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)

    a_nm_ref = 496.0
    combos = list(itertools.product(H_SPINE_VALS, H_RIB_VALS, W_RIB_VALS))
    total = len(combos)

    # ------------------------------------------------------------------
    # Phase 1: Load all cached results
    # ------------------------------------------------------------------
    print(f"\n{'='*75}")
    print(f"  PHASE 1: Loading {total} cached results")
    print(f"{'='*75}")

    results = []
    loaded = 0
    missing = 0

    for idx, (hs, hr, wr) in enumerate(combos):
        cpath = _cache_path(a_nm_ref, hs, wr, hr, res, nb, ki)

        if not os.path.exists(cpath):
            missing += 1
            continue

        all_freqs, k_x = _load(cpath)
        loaded += 1

        best_m = None
        best_b = None
        for b in band_range:
            m = measure_flat_ng(all_freqs, k_x, a_nm_ref, b)
            if m is not None:
                if best_m is None or m['bw_nm'] > best_m['bw_nm']:
                    best_m = m
                    best_b = b

        results.append(dict(
            h_spine=hs, h_rib=hr, W_rib=wr,
            a_nm_ref=a_nm_ref,
            all_freqs=all_freqs, k_x=k_x,
            best_band=best_b, metrics=best_m,
        ))

    print(f"  Loaded: {loaded}/{total}  (missing: {missing})")

    # Rank by BW
    valid = [r for r in results if r['metrics'] is not None]
    valid.sort(key=lambda r: (-r['metrics']['bw_nm'], r['metrics']['ng_std']))

    print(f"  Configs with ng ∈ [{NG_LO}, {NG_HI}] window: {len(valid)}")

    # Print Phase-1 top-15
    print(f"\n{'='*75}")
    print(f"  PHASE 1 RANKING (at a_nm={a_nm_ref:.0f}nm, before alignment)")
    print(f"{'='*75}")
    hdr = (f"  {'#':>3}  {'h_spine':>8}  {'h_rib':>7}  {'W_rib':>7}  "
           f"{'band':>5}  {'λc(nm)':>8}  {'ng':>6}  {'σ(ng)':>7}  {'BW(nm)':>7}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for i, r in enumerate(valid[:15], 1):
        m = r['metrics']
        print(f"  {i:>3}  {r['h_spine']:>8.3f}  {r['h_rib']:>7.3f}  {r['W_rib']:>7.3f}  "
              f"{r['best_band']:>5}  {m['wl_center']:>8.1f}  "
              f"{m['ng_mean']:>6.2f}  {m['ng_std']:>7.3f}  {m['bw_nm']:>7.1f}")
    print(f"{'='*75}")

    # ------------------------------------------------------------------
    # Phase 2: Align top-N to 1550nm
    # ------------------------------------------------------------------
    top_n = min(args.top_n, len(valid))
    print(f"\n{'='*75}")
    print(f"  PHASE 2: Auto-align top {top_n} to 1550nm (timeout={args.timeout}s each)")
    print(f"{'='*75}\n")

    aligned = []
    for rank, r in enumerate(valid[:top_n], 1):
        hs, hr, wr = r['h_spine'], r['h_rib'], r['W_rib']
        b = r['best_band']

        a_new = auto_align_a(r['all_freqs'], r['k_x'], a_nm_ref, b)

        # Check if same as ref (no re-run needed)
        if abs(a_new - a_nm_ref) < 0.5:
            all_freqs2, k_x2 = r['all_freqs'], r['k_x']
            print(f"  #{rank:2d}  a_new={a_new:.1f}nm ≈ ref, using cached Phase-1 data")
        else:
            cpath2 = _cache_path(a_new, hs, wr, hr, res, nb, ki)
            if os.path.exists(cpath2):
                all_freqs2, k_x2 = _load(cpath2)
                print(f"  #{rank:2d}  a_new={a_new:.1f}nm [cache hit]")
            else:
                print(f"  #{rank:2d}  a_new={a_new:.1f}nm — running MPB (timeout {args.timeout}s) ...")
                t0 = time.time()

                # Use alarm-based timeout
                def _timeout_handler(signum, frame):
                    raise TimeoutError()
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(args.timeout)
                try:
                    all_freqs2, k_x2 = run_mpb(a_new, hs, wr, hr, res, nb, ki)
                    signal.alarm(0)
                    _save(cpath2, all_freqs2, k_x2, a_new, hs, wr, hr, res, nb)
                    dt = time.time() - t0
                    print(f"         done in {dt:.0f}s")
                except (TimeoutError, Exception) as e:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                    print(f"         SKIPPED ({type(e).__name__})")
                    # Fall back to Phase-1 data with scaled a_nm approximation
                    all_freqs2, k_x2 = r['all_freqs'], r['k_x']
                finally:
                    signal.signal(signal.SIGALRM, old_handler)

        m2 = measure_flat_ng(all_freqs2, k_x2, a_new, b)

        if m2 is not None:
            print(f"         hs={hs:.3f} hr={hr:.3f} Wr={wr:.3f}  "
                  f"band={b}  BW={m2['bw_nm']:.1f}nm  "
                  f"ng={m2['ng_mean']:.2f}±{m2['ng_std']:.3f}  λc={m2['wl_center']:.1f}nm")
        else:
            print(f"         -- no ng window after alignment --")

        aligned.append(dict(
            rank=rank, h_spine=hs, h_rib=hr, W_rib=wr,
            a_nm=a_new, band=b,
            all_freqs=all_freqs2, k_x=k_x2,
            metrics=m2,
        ))

    # ------------------------------------------------------------------
    # Final ranking
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
    print(f"{'='*75}")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    save_data = {
        'h_spine_arr':   np.array([a['h_spine'] for a in aligned_valid]),
        'h_rib_arr':     np.array([a['h_rib'] for a in aligned_valid]),
        'W_rib_arr':     np.array([a['W_rib'] for a in aligned_valid]),
        'a_nm_arr':      np.array([a['a_nm'] for a in aligned_valid]),
        'band_arr':      np.array([a['band'] for a in aligned_valid]),
        'bw_arr':        np.array([a['metrics']['bw_nm'] for a in aligned_valid]),
        'ng_mean_arr':   np.array([a['metrics']['ng_mean'] for a in aligned_valid]),
        'ng_std_arr':    np.array([a['metrics']['ng_std'] for a in aligned_valid]),
        'wl_center_arr': np.array([a['metrics']['wl_center'] for a in aligned_valid]),
    }
    for i, a in enumerate(aligned_valid[:5]):
        save_data[f'top{i}_all_freqs'] = a['all_freqs']
        save_data[f'top{i}_k_x']       = a['k_x']

    npz_path = os.path.join(out_dir, 'optimize_ng6to7_results.npz')
    np.savez_compressed(npz_path, **save_data)
    print(f"\n  Saved: {npz_path}")

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
    json_path = os.path.join(out_dir, 'optimize_ng6to7_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_path}")

    # Winner
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
        print(f"    BW(nm)  = {m['bw_nm']:.1f} nm  (ng in [{NG_LO}, {NG_HI}])")
        print(f"    lam_c   = {m['wl_center']:.1f} nm")
        print(f"{'='*75}\n")


if __name__ == '__main__':
    main()
