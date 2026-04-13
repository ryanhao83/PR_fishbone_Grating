"""
sweep_ysym_partial_etch.py — Y-symmetric 3D grating: partial-etch depth sweep
              and period alignment to 1550nm.

Steps:
  1. Build a y-symmetric fishbone geometry (Wt=Wb, ht=hb, delta_s=0).
  2. Run MPB k-by-k for t_partial = 71nm and 140nm, extracting te_frac per
     (k, band) via compute_field_energy.
  3. Plot colormapped band diagrams (continuous gray lines + scatter colored
     by te_frac, red=TE, blue=TM).
  4. Evaluate which t_partial gives a larger TM-free gap around the TE
     slow-light band near 1550nm.
  5. For the better t_partial, adjust a_nm to align the slow-light band
     center to 1550nm and re-run to confirm.

Usage:
  python sweep_ysym_partial_etch.py
  python sweep_ysym_partial_etch.py --resolution 32 --num-bands 10
"""

import hashlib
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_POLY_SI   = 3.48
N_SIO2      = 1.44
T_SLAB_NM   = 211.0

RESOLUTION  = 16
NUM_BANDS   = 18
K_MIN       = 0.35
K_MAX       = 0.50
K_INTERP    = 45
PAD_Y       = 2.0
PAD_Z       = 1.5

NG_LOW      = 6.0
NG_HIGH     = 8.0
WL_TARGET   = 1550.0

# Base structure (from 2D optimisation, rank-1 geometry made y-symmetric)
BASE_PARAMS = dict(
    a_nm=389.0, h_spine=0.550,
    W_rib=0.484, h_rib=0.514,   # single rib size (both sides identical)
)

T_PARTIAL_CANDIDATES = [71.0, 140.0]   # nm, PDK allowed etch depths


# ---------------------------------------------------------------------------
# Geometry builder (y-symmetric)
# ---------------------------------------------------------------------------
def build_geometry_ysym(params, t_partial_nm):
    """Y-symmetric fishbone: identical ribs on +y and -y, delta_s=0."""
    import meep as mp

    a_nm    = params['a_nm']
    t_slab  = T_SLAB_NM / a_nm
    h_spine = params['h_spine']
    W_rib   = params['W_rib']
    h_rib   = params['h_rib']

    sy = 2.0 * (h_spine + h_rib) + 2.0 * PAD_Y
    sz = t_slab + 2.0 * PAD_Z

    lattice = mp.Lattice(size=mp.Vector3(1, sy, sz))
    Si   = mp.Medium(index=N_POLY_SI)
    SiO2 = mp.Medium(index=N_SIO2)

    geometry = []

    # SiO2 substrate (z < -t_slab/2)
    geometry.append(mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, PAD_Z),
        center=mp.Vector3(0, 0, -(t_slab / 2 + PAD_Z / 2)),
        material=SiO2))

    # SiO2 top cladding (z > t_slab/2)
    geometry.append(mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, PAD_Z),
        center=mp.Vector3(0, 0, t_slab / 2 + PAD_Z / 2),
        material=SiO2))

    # Partial etch Si layer (continuous, at slab bottom)
    t_partial = t_partial_nm / a_nm
    geometry.append(mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, t_partial),
        center=mp.Vector3(0, 0, -t_slab / 2.0 + t_partial / 2.0),
        material=Si))

    # Central spine (full slab)
    geometry.append(mp.Block(
        size=mp.Vector3(mp.inf, 2.0 * h_spine, t_slab),
        center=mp.Vector3(0, 0, 0),
        material=Si))

    # +y rib (full slab)
    geometry.append(mp.Block(
        size=mp.Vector3(W_rib, h_rib, t_slab),
        center=mp.Vector3(0, h_spine + h_rib / 2.0, 0),
        material=Si))

    # -y rib (mirror, full slab)
    geometry.append(mp.Block(
        size=mp.Vector3(W_rib, h_rib, t_slab),
        center=mp.Vector3(0, -(h_spine + h_rib / 2.0), 0),
        material=Si))

    return lattice, geometry, sy, sz, t_slab


# ---------------------------------------------------------------------------
# MPB runner
# ---------------------------------------------------------------------------
def run_mpb(params, t_partial_nm, k_min=K_MIN, k_max=K_MAX,
            k_interp=K_INTERP, num_bands=NUM_BANDS, resolution=RESOLUTION):
    """Run 3D MPB k-by-k, return all_freqs, te_frac, k_x, epsilon."""
    import meep as mp
    import meep.mpb as mpb

    lattice, geometry, sy, sz, t_slab = build_geometry_ysym(params, t_partial_nm)
    k_points = mp.interpolate(k_interp, [mp.Vector3(k_min), mp.Vector3(k_max)])

    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=k_points,
        resolution=resolution,
        num_bands=num_bands,
    )

    a = params['a_nm']
    num_k = len(k_points)
    nb = num_bands
    k_x = np.array([k.x for k in k_points])

    print(f"\n{'='*70}")
    print(f"3D MPB (y-sym): a={a:.0f}nm  t_partial={t_partial_nm:.0f}nm  "
          f"res={resolution}  bands={nb}  k-pts={num_k}")
    print(f"{'='*70}")

    all_freqs = np.zeros((num_k, nb))
    te_frac   = np.zeros((num_k, nb))

    for ik, k in enumerate(k_points):
        ms.k_points = [k]
        ms.run()
        all_freqs[ik, :] = ms.all_freqs[0]

        for ib in range(nb):
            ms.get_dfield(ib + 1)
            energy = ms.compute_field_energy()
            total = energy[0]
            if total > 0:
                te_frac[ik, ib] = 1.0 - energy[3] / total
            else:
                te_frac[ik, ib] = np.nan

        if (ik + 1) % 8 == 0 or ik == num_k - 1:
            print(f"    k-point {ik+1}/{num_k}")

    eps = np.array(ms.get_epsilon())
    return dict(all_freqs=all_freqs, te_frac=te_frac, k_x=k_x,
                epsilon=eps, sy=sy, sz=sz, t_slab=t_slab,
                params=dict(params), t_partial_nm=t_partial_nm,
                resolution=resolution, num_bands=num_bands)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def compute_ng(f_band, k_x):
    df_dk = np.gradient(f_band, k_x)
    df_dk = np.where(df_dk == 0.0, np.nan, df_dk)
    return 1.0 / df_dk


def find_te_slow_light_band(data):
    """Find the TE-like band with best slow-light FOM near 1550nm.

    Returns (band_index, wl_center, ng_mean, bw_nm) or None.
    """
    all_freqs = data['all_freqs']
    te_frac   = data['te_frac']
    k_x       = data['k_x']
    a         = data['params']['a_nm']
    nb        = all_freqs.shape[1]

    best = None
    for b in range(nb):
        f_b  = all_freqs[:, b]
        tf_b = te_frac[:, b]

        # Require band to be mostly TE (median > 0.7)
        if np.nanmedian(tf_b) < 0.7:
            continue
        if np.any(f_b <= 1e-6):
            continue

        ng = compute_ng(f_b, k_x)
        with np.errstate(divide='ignore', invalid='ignore'):
            wl = np.where(f_b > 0, a / f_b, np.nan)

        mask = (np.isfinite(ng) & (ng > 0) & (ng < 200)
                & (wl > 1400) & (wl < 1700))
        if np.sum(mask) < 3:
            continue

        sl = mask & (ng >= NG_LOW) & (ng <= NG_HIGH)
        if not np.any(sl):
            continue

        wl_lo, wl_hi = wl[sl].min(), wl[sl].max()
        bw = wl_hi - wl_lo
        wl_c = (wl_lo + wl_hi) / 2.0
        ng_mean = float(np.nanmean(ng[sl]))

        if best is None or bw > best[3]:
            best = (b, wl_c, ng_mean, bw)

    return best


def find_tm_gap(data, f_lo_te, f_hi_te, k_lo=0.40):
    """Compute the minimum TE-to-TM frequency gap in the slow-light k-range.

    Only considers k-points with k >= k_lo (where slow light operates).
    Returns the minimum gap across those k-points (larger = better).
    """
    all_freqs = data['all_freqs']
    te_frac   = data['te_frac']
    k_x       = data['k_x']
    num_k, nb = all_freqs.shape

    min_gap = np.inf
    for ik in range(num_k):
        if k_x[ik] < k_lo:
            continue
        for b in range(nb):
            if te_frac[ik, b] > 0.7:
                continue  # skip TE bands
            f_tm = all_freqs[ik, b]
            if f_tm < 1e-6:
                continue
            if f_lo_te <= f_tm <= f_hi_te:
                return 0.0  # TM inside TE window
            gap = min(abs(f_tm - f_lo_te), abs(f_tm - f_hi_te))
            min_gap = min(min_gap, gap)

    return min_gap if np.isfinite(min_gap) else -1.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_band_colormapped(data, title_extra="", save_path=None):
    """Band diagram: gray lines + scatter colored by te_frac (red=TE, blue=TM)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    all_freqs = data['all_freqs']
    te_frac   = data['te_frac']
    k_x       = data['k_x']
    a         = data['params']['a_nm']
    nb        = all_freqs.shape[1]

    f_lo, f_hi = 0.20, 0.38
    f_1550 = a / WL_TARGET

    fig, ax = plt.subplots(figsize=(7, 5))

    # Light cone
    k_lc = np.linspace(k_x[0], k_x[-1], 100)
    f_lc = k_lc / N_SIO2
    ax.fill_between(k_lc, np.maximum(f_lc, f_lo), f_hi,
                    alpha=0.08, color='gray')
    ax.plot(k_lc, f_lc, color='gray', lw=0.6)

    # All bands as continuous gray lines
    for b in range(nb):
        ax.plot(k_x, all_freqs[:, b], color='lightgray', lw=1.0, zorder=1)

    # Scatter overlay colored by te_frac
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1.0)
    for b in range(nb):
        sc = ax.scatter(k_x, all_freqs[:, b], c=te_frac[:, b],
                        cmap='coolwarm', norm=norm, s=6, zorder=2,
                        edgecolors='none')

    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('TE fraction (red=TE, blue=TM)')

    ax.axhline(f_1550, color='green', ls=':', lw=1.0, label='1550nm')

    ax.set_xlim(k_x[0], k_x[-1])
    ax.set_ylim(f_lo, f_hi)
    ax.set_xlabel('Wave vector k (2\u03c0/a)')
    ax.set_ylabel('Frequency (a/\u03bb)')
    ax.legend(fontsize=8, loc='upper left')

    t_p = data['t_partial_nm']
    ax.set_title(f'Y-symmetric 3D bands: a={a:.0f}nm, t_partial={t_p:.0f}nm'
                 f'{title_extra}')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close(fig)


def plot_comparison(results, save_path=None):
    """Side-by-side colormapped band diagrams for all t_partial values."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1.0)

    for ax, data in zip(axes, results):
        all_freqs = data['all_freqs']
        te_frac   = data['te_frac']
        k_x       = data['k_x']
        a         = data['params']['a_nm']
        nb        = all_freqs.shape[1]
        t_p       = data['t_partial_nm']

        f_lo, f_hi = 0.20, 0.38
        f_1550 = a / WL_TARGET

        k_lc = np.linspace(k_x[0], k_x[-1], 100)
        f_lc = k_lc / N_SIO2
        ax.fill_between(k_lc, np.maximum(f_lc, f_lo), f_hi,
                        alpha=0.08, color='gray')
        ax.plot(k_lc, f_lc, color='gray', lw=0.6)

        for b in range(nb):
            ax.plot(k_x, all_freqs[:, b], color='lightgray', lw=1.0, zorder=1)

        sc = None
        for b in range(nb):
            sc = ax.scatter(k_x, all_freqs[:, b], c=te_frac[:, b],
                            cmap='coolwarm', norm=norm, s=6, zorder=2,
                            edgecolors='none')

        ax.axhline(f_1550, color='green', ls=':', lw=1.0, label='1550nm')
        ax.set_xlim(k_x[0], k_x[-1])
        ax.set_ylim(f_lo, f_hi)
        ax.set_xlabel('k (2\u03c0/a)')
        ax.set_title(f't_partial = {t_p:.0f} nm')
        ax.legend(fontsize=7, loc='upper left')

    axes[0].set_ylabel('Frequency (a/\u03bb)')

    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, shrink=0.8)
        cb.set_label('TE fraction (red=TE, blue=TM)')

    fig.suptitle(f'Partial-etch sweep (y-symmetric, a={results[0]["params"]["a_nm"]:.0f}nm)',
                 fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=RESOLUTION)
    parser.add_argument('--num-bands', type=int, default=NUM_BANDS)
    parser.add_argument('--k-interp', type=int, default=K_INTERP)
    parser.add_argument('--save-plots', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out, exist_ok=True)

    # ---- Step 1-3: Run both t_partial values ----
    sweep_results = []
    for t_p in T_PARTIAL_CANDIDATES:
        params = dict(BASE_PARAMS)
        data = run_mpb(params, t_p,
                       k_interp=args.k_interp,
                       num_bands=args.num_bands,
                       resolution=args.resolution)
        sweep_results.append(data)

    # ---- Side-by-side comparison plot ----
    sp = os.path.join(out, 'ysym_partial_etch_comparison.png') if args.save_plots else None
    plot_comparison(sweep_results, save_path=sp)

    # ---- Step 4: Evaluate TM gap & pick best t_partial ----
    print(f"\n{'='*70}")
    print("PARTIAL ETCH EVALUATION")
    print(f"{'='*70}")

    candidates = []

    for i, data in enumerate(sweep_results):
        t_p = data['t_partial_nm']
        sl = find_te_slow_light_band(data)
        a = data['params']['a_nm']

        if sl is None:
            print(f"  t_partial={t_p:.0f}nm: NO TE slow-light band found")
            continue

        band, wl_c, ng_mean, bw = sl
        f_band = data['all_freqs'][:, band]
        k_x = data['k_x']
        ng = compute_ng(f_band, k_x)
        with np.errstate(divide='ignore', invalid='ignore'):
            wl = np.where(f_band > 0, a / f_band, np.nan)
        sl_mask = (np.isfinite(ng) & (ng >= NG_LOW) & (ng <= NG_HIGH)
                   & (wl > 1400) & (wl < 1700))
        f_sl_lo = f_band[sl_mask].min()
        f_sl_hi = f_band[sl_mask].max()

        tm_gap = find_tm_gap(data, f_sl_lo, f_sl_hi)

        print(f"  t_partial={t_p:.0f}nm: TE band {band}, "
              f"wl_center={wl_c:.1f}nm, ng={ng_mean:.2f}, BW={bw:.1f}nm, "
              f"TM gap={tm_gap:.5f} (norm freq)")

        candidates.append(dict(idx=i, tm_gap=tm_gap, bw=bw,
                                sl=(band, wl_c, ng_mean, bw)))

    if not candidates:
        print("\nNo suitable configuration found.")
        return

    # Pick best: largest TM gap first, then largest BW as tiebreaker
    candidates.sort(key=lambda c: (c['tm_gap'], c['bw']), reverse=True)
    winner = candidates[0]
    best_idx = winner['idx']
    best_gap = winner['tm_gap']
    best_sl  = winner['sl']

    if best_idx is None:
        print("\nNo suitable configuration found.")
        return

    chosen = sweep_results[best_idx]
    t_partial_best = chosen['t_partial_nm']
    band_best, wl_c_best, ng_best, bw_best = best_sl
    a_orig = chosen['params']['a_nm']

    print(f"\n  >> Best: t_partial = {t_partial_best:.0f} nm  (TM gap = {best_gap:.5f})")

    # ---- Step 5: Adjust a_nm to align slow-light center to 1550nm ----
    # The slow-light band center is at wl_c_best with a_orig.
    # Normalized frequency f_norm = a / wl is invariant under scaling.
    # New a = a_orig * (1550 / wl_c_best)
    a_new = a_orig * (WL_TARGET / wl_c_best)
    print(f"\n  Period adjustment: a_orig={a_orig:.1f}nm, "
          f"wl_center={wl_c_best:.1f}nm")
    print(f"  -> a_new = {a_orig:.1f} * (1550 / {wl_c_best:.1f}) = {a_new:.1f} nm")

    # Re-run with adjusted a_nm
    params_new = dict(BASE_PARAMS)
    params_new['a_nm'] = round(a_new, 1)
    print(f"\n  Re-running with a_nm = {params_new['a_nm']:.1f} nm, "
          f"t_partial = {t_partial_best:.0f} nm ...")

    data_final = run_mpb(params_new, t_partial_best,
                         k_interp=args.k_interp,
                         num_bands=args.num_bands,
                         resolution=args.resolution)

    sp = os.path.join(out, 'ysym_final_aligned.png') if args.save_plots else None
    plot_band_colormapped(data_final, title_extra=" [ALIGNED]", save_path=sp)

    # Final verification
    sl_final = find_te_slow_light_band(data_final)
    if sl_final:
        band_f, wl_f, ng_f, bw_f = sl_final
        f_band = data_final['all_freqs'][:, band_f]
        k_x = data_final['k_x']
        ng = compute_ng(f_band, k_x)
        a_f = data_final['params']['a_nm']
        with np.errstate(divide='ignore', invalid='ignore'):
            wl = np.where(f_band > 0, a_f / f_band, np.nan)
        sl_mask = (np.isfinite(ng) & (ng >= NG_LOW) & (ng <= NG_HIGH)
                   & (wl > 1400) & (wl < 1700))
        f_sl_lo = f_band[sl_mask].min()
        f_sl_hi = f_band[sl_mask].max()
        tm_gap_f = find_tm_gap(data_final, f_sl_lo, f_sl_hi)

        print(f"\n{'='*70}")
        print("FINAL RESULT")
        print(f"{'='*70}")
        print(f"  t_partial   = {t_partial_best:.0f} nm")
        print(f"  a_nm        = {params_new['a_nm']:.1f} nm")
        print(f"  TE band     = {band_f}")
        print(f"  wl_center   = {wl_f:.1f} nm")
        print(f"  ng_mean     = {ng_f:.2f}")
        print(f"  BW (ng {NG_LOW}-{NG_HIGH}) = {bw_f:.1f} nm")
        print(f"  TM gap      = {tm_gap_f:.5f} (norm freq)")
        print(f"{'='*70}")
    else:
        print("  WARNING: No slow-light band found after alignment!")


if __name__ == '__main__':
    main()
