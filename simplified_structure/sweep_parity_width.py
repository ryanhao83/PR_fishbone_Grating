"""
sweep_parity_width.py — Y-symmetric 3D grating: parity-filtered TE0 band diagrams
                        with transverse width reduction sweep.

Improvements over sweep_ysym_partial_etch.py:
  1. Parity filtering: uses ms.run_yeven() to compute ONLY y-even modes.
     In this geometry (x: periodic, y: transverse, z: vertical slab), the
     fundamental TE-like mode has its dominant E_y component even about y=0
     (consistent with fishbone_gc_2d.py and fishbone_gc_3d.py conventions).
     run_yeven() halves the k-space and eliminates all y-odd (TM-like) modes,
     giving a clean single-parity band diagram that only shows modes that can
     couple to a TE0 input waveguide.

  2. Transverse width sweep: h_spine and h_rib (the y-direction dimensions
     controlling lateral mode confinement) are scaled by 100%, 95%, 90%, 85%
     of their baseline values. Wt = Wb = W_rib is strictly maintained (single
     W_rib parameter, same on both +y and -y sides) to preserve y-symmetry.
     Reducing the transverse width blue-shifts higher-order lateral modes,
     potentially clearing the bandgap of competing modes.

  3. Priority: t_partial=140nm is run first (prior sweeps showed it gives
     better slow-light performance). Falls back to t_partial=70nm only if
     no width variant at 140nm meets the slow-light target.

Usage:
  python sweep_parity_width.py
  python sweep_parity_width.py --resolution 32 --num-bands 8 --save-plots --no-show
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
NUM_BANDS   = 30
K_MIN       = 0.35
K_MAX       = 0.50
K_INTERP    = 45
PAD_Y       = 2.0
PAD_Z       = 1.5

NG_LOW      = 6.0
NG_HIGH     = 8.0
WL_TARGET   = 1550.0

# Base structure (y-symmetric: W_rib used for both +y and -y ribs)
BASE_PARAMS = dict(
    a_nm=389.0, h_spine=0.550,
    W_rib=0.484, h_rib=0.514,
)

# Transverse width scale factors: 0%, 5%, 10%, 15% reduction
WIDTH_SCALES = [1.0, 0.95, 0.90, 0.85]

# Partial-etch depths: 140nm is priority-1, 70nm is fallback
T_PARTIAL_PRIMARY  = 140.0   # nm
T_PARTIAL_FALLBACK =  70.0   # nm


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def _cache_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(here, 'cache_3d')
    os.makedirs(d, exist_ok=True)
    return d


def _cache_key(params, t_partial_nm, resolution, num_bands, k_min, k_max, k_interp):
    d = dict(params)
    d.update(
        t_partial_nm=float(t_partial_nm),
        resolution=resolution,
        num_bands=num_bands,
        k_min=k_min, k_max=k_max, k_interp=k_interp,
        parity='yeven',
    )
    s = json.dumps(d, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def cache_path(params, t_partial_nm, resolution, num_bands, k_min, k_max, k_interp):
    h = _cache_key(params, t_partial_nm, resolution, num_bands, k_min, k_max, k_interp)
    fname = f"parity_yeven_res{resolution}_nb{num_bands}_{h}.npz"
    return os.path.join(_cache_dir(), fname)


def save_results(data, path):
    flat = {k: v for k, v in data.items()
            if isinstance(v, (np.ndarray, float, int))}
    flat['params_json']      = json.dumps(data['params'])
    flat['t_partial_nm']     = data['t_partial_nm']
    flat['resolution']       = data['resolution']
    flat['num_bands']        = data['num_bands']
    flat['width_scale']      = data['width_scale']
    np.savez_compressed(path, **flat)
    print(f"  [cache] Saved: {os.path.basename(path)}")


def load_results(path):
    npz = np.load(path, allow_pickle=False)
    data = {k: npz[k] for k in npz.files}
    data['params']      = json.loads(str(data.pop('params_json')))
    data['t_partial_nm'] = float(data['t_partial_nm'])
    data['resolution']   = int(data['resolution'])
    data['num_bands']    = int(data['num_bands'])
    data['width_scale']  = float(data['width_scale'])
    return data


# ---------------------------------------------------------------------------
# Geometry builder (y-symmetric)
# ---------------------------------------------------------------------------
def build_geometry_ysym(params, t_partial_nm, width_scale=1.0):
    """Y-symmetric fishbone with optional transverse width scaling.

    width_scale < 1.0 shrinks h_spine and h_rib (y-direction) to push
    higher-order lateral modes to higher frequencies.
    Wt = Wb = W_rib is always maintained (y-symmetry enforced).
    """
    import meep as mp

    a_nm    = params['a_nm']
    t_slab  = T_SLAB_NM / a_nm
    h_spine = params['h_spine'] * width_scale   # scale transverse half-width
    W_rib   = params['W_rib']                   # x-extent (fill factor), unchanged
    h_rib   = params['h_rib']   * width_scale   # scale rib transverse height

    sy = 2.0 * (h_spine + h_rib) + 2.0 * PAD_Y
    sz = t_slab + 2.0 * PAD_Z

    lattice = mp.Lattice(size=mp.Vector3(1, sy, sz))
    Si   = mp.Medium(index=N_POLY_SI)
    SiO2 = mp.Medium(index=N_SIO2)

    geometry = []

    # SiO2 substrate (below slab)
    geometry.append(mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, PAD_Z),
        center=mp.Vector3(0, 0, -(t_slab / 2 + PAD_Z / 2)),
        material=SiO2))

    # SiO2 top cladding
    geometry.append(mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, PAD_Z),
        center=mp.Vector3(0, 0, t_slab / 2 + PAD_Z / 2),
        material=SiO2))

    # Partial-etch Si layer (continuous, bottom of slab)
    t_partial = t_partial_nm / a_nm
    geometry.append(mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, t_partial),
        center=mp.Vector3(0, 0, -t_slab / 2.0 + t_partial / 2.0),
        material=Si))

    # Central spine (full slab height)
    geometry.append(mp.Block(
        size=mp.Vector3(mp.inf, 2.0 * h_spine, t_slab),
        center=mp.Vector3(0, 0, 0),
        material=Si))

    # +y rib  (Wt = W_rib, full slab height)
    geometry.append(mp.Block(
        size=mp.Vector3(W_rib, h_rib, t_slab),
        center=mp.Vector3(0, h_spine + h_rib / 2.0, 0),
        material=Si))

    # -y rib  (Wb = W_rib, enforcing Wt == Wb for y-symmetry)
    geometry.append(mp.Block(
        size=mp.Vector3(W_rib, h_rib, t_slab),
        center=mp.Vector3(0, -(h_spine + h_rib / 2.0), 0),
        material=Si))

    return lattice, geometry, sy, sz, t_slab


# ---------------------------------------------------------------------------
# MPB runner — parity filtered (y-even only)
# ---------------------------------------------------------------------------
def run_mpb_yeven(params, t_partial_nm, width_scale=1.0,
                  k_min=K_MIN, k_max=K_MAX, k_interp=K_INTERP,
                  num_bands=NUM_BANDS, resolution=RESOLUTION):
    """Run 3D MPB k-by-k with run_yeven() for TE0 parity selection.

    The fundamental TE-like mode is y-even in this geometry:
      • E_y (dominant) is even about y=0
      • run_yeven() restricts the solve to the y-even eigenspace,
        eliminating all y-odd (TM-like / higher-order lateral) modes.

    Returns dict with keys: all_freqs, te_frac, k_x, epsilon, params,
                            t_partial_nm, width_scale, resolution, num_bands.
    """
    import meep as mp
    import meep.mpb as mpb

    lattice, geometry, sy, sz, t_slab = build_geometry_ysym(
        params, t_partial_nm, width_scale)
    k_points = mp.interpolate(k_interp, [mp.Vector3(k_min), mp.Vector3(k_max)])

    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=k_points,
        resolution=resolution,
        num_bands=num_bands,
    )

    a      = params['a_nm']
    num_k  = len(k_points)
    nb     = num_bands
    k_x    = np.array([k.x for k in k_points])

    ws_pct = int(round((1.0 - width_scale) * 100))
    print(f"\n{'='*70}")
    print(f"3D MPB (y-even only): a={a:.0f}nm  t_partial={t_partial_nm:.0f}nm  "
          f"width_scale={width_scale:.2f} (-{ws_pct}%)")
    print(f"  res={resolution}  bands={nb}  k-pts={num_k}")
    print(f"  h_spine={params['h_spine']*width_scale:.3f}  "
          f"h_rib={params['h_rib']*width_scale:.3f}  "
          f"W_rib={params['W_rib']:.3f}  (Wt=Wb enforced)")
    print(f"{'='*70}")

    all_freqs = np.zeros((num_k, nb))
    te_frac   = np.zeros((num_k, nb))

    for ik, k in enumerate(k_points):
        ms.k_points = [k]
        # *** Parity filter: only y-even modes (TE0-compatible sector) ***
        ms.run_yeven()
        all_freqs[ik, :] = ms.all_freqs[0]

        for ib in range(nb):
            ms.get_dfield(ib + 1)
            energy = ms.compute_field_energy()
            total  = energy[0]
            if total > 0:
                te_frac[ik, ib] = 1.0 - energy[3] / total
            else:
                te_frac[ik, ib] = np.nan

        if (ik + 1) % 8 == 0 or ik == num_k - 1:
            print(f"    k-point {ik+1}/{num_k}")

    eps = np.array(ms.get_epsilon())
    return dict(
        all_freqs=all_freqs, te_frac=te_frac, k_x=k_x,
        epsilon=eps, sy=sy, sz=sz, t_slab=t_slab,
        params=dict(params), t_partial_nm=float(t_partial_nm),
        width_scale=float(width_scale),
        resolution=resolution, num_bands=nb,
    )


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def compute_ng(f_band, k_x):
    df_dk = np.gradient(f_band, k_x)
    df_dk = np.where(df_dk == 0.0, np.nan, df_dk)
    return 1.0 / df_dk


def find_te_slow_light_band(data):
    """Return (band_idx, wl_center, ng_mean, bw_nm) for the best TE slow-light
    band, or None if no band meets the criteria."""
    all_freqs = data['all_freqs']
    te_frac   = data['te_frac']
    k_x       = data['k_x']
    a         = data['params']['a_nm']
    nb        = all_freqs.shape[1]

    best = None
    for b in range(nb):
        f_b  = all_freqs[:, b]
        tf_b = te_frac[:, b]

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
        bw     = wl_hi - wl_lo
        wl_c   = (wl_lo + wl_hi) / 2.0
        ng_mean = float(np.nanmean(ng[sl]))

        if best is None or bw > best[3]:
            best = (b, wl_c, ng_mean, bw)

    return best


def meets_target(data, min_bw=5.0, wl_tolerance=100.0):
    """Return True if data contains a TE slow-light band near 1550nm."""
    sl = find_te_slow_light_band(data)
    if sl is None:
        return False
    _, wl_c, _, bw = sl
    return bw >= min_bw and abs(wl_c - WL_TARGET) < wl_tolerance


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_band_colormapped(data, title_extra="", save_path=None, show=True):
    """Band diagram: gray lines + scatter colored by te_frac (red=TE, blue=TM).

    Only y-even modes are plotted (as returned by run_yeven).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    all_freqs = data['all_freqs']
    te_frac   = data['te_frac']
    k_x       = data['k_x']
    a         = data['params']['a_nm']
    nb        = all_freqs.shape[1]
    t_p       = data['t_partial_nm']
    ws        = data['width_scale']
    ws_pct    = int(round((1.0 - ws) * 100))

    f_lo, f_hi = 0.20, 0.38
    f_1550 = a / WL_TARGET

    fig, ax = plt.subplots(figsize=(7, 5))

    k_lc = np.linspace(k_x[0], k_x[-1], 100)
    f_lc = k_lc / N_SIO2
    ax.fill_between(k_lc, np.maximum(f_lc, f_lo), f_hi,
                    alpha=0.08, color='gray', label='Light cone')
    ax.plot(k_lc, f_lc, color='gray', lw=0.6)

    # Continuous gray lines for all y-even bands
    for b in range(nb):
        ax.plot(k_x, all_freqs[:, b], color='lightgray', lw=1.0, zorder=1)

    # Scatter overlay colored by te_frac
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1.0)
    sc = None
    for b in range(nb):
        sc = ax.scatter(k_x, all_freqs[:, b], c=te_frac[:, b],
                        cmap='coolwarm', norm=norm, s=6, zorder=2,
                        edgecolors='none')

    if sc is not None:
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label('TE fraction (red=TE, blue=TM)')

    ax.axhline(f_1550, color='green', ls=':', lw=1.0, label='1550 nm')

    # Mark slow-light window if a target band exists
    sl = find_te_slow_light_band(data)
    if sl is not None:
        b, wl_c, ng_m, bw = sl
        f_b = all_freqs[:, b]
        ng  = compute_ng(f_b, k_x)
        with np.errstate(divide='ignore', invalid='ignore'):
            wl = np.where(f_b > 0, a / f_b, np.nan)
        sl_mask = (np.isfinite(ng) & (ng >= NG_LOW) & (ng <= NG_HIGH)
                   & (wl > 1400) & (wl < 1700))
        if np.any(sl_mask):
            f_sl_lo = f_b[sl_mask].min()
            f_sl_hi = f_b[sl_mask].max()
            ax.axhspan(f_sl_lo, f_sl_hi, color='lime', alpha=0.15,
                       label=f'SL window: ng={ng_m:.1f}, BW={bw:.0f}nm')

    ax.set_xlim(k_x[0], k_x[-1])
    ax.set_ylim(f_lo, f_hi)
    ax.set_xlabel('Wave vector k (2\u03c0/a)')
    ax.set_ylabel('Frequency (a/\u03bb)')
    ax.legend(fontsize=8, loc='upper left')

    ws_label = f"  [-{ws_pct}% width]" if ws_pct > 0 else ""
    ax.set_title(
        f'Y-even modes only: a={a:.0f}nm, t_partial={t_p:.0f}nm'
        f'{ws_label}{title_extra}')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_width_sweep(results, t_partial_nm, save_path=None, show=True):
    """2×2 grid of colormapped band diagrams for the four width scales."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    n = len(results)
    ncols = min(n, 2)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 5 * nrows), sharey=True)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1.0)

    for ax, data in zip(axes_flat, results):
        all_freqs = data['all_freqs']
        te_frac   = data['te_frac']
        k_x       = data['k_x']
        a         = data['params']['a_nm']
        nb        = all_freqs.shape[1]
        ws        = data['width_scale']
        ws_pct    = int(round((1.0 - ws) * 100))

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

        ax.axhline(f_1550, color='green', ls=':', lw=1.0, label='1550 nm')

        sl = find_te_slow_light_band(data)
        if sl is not None:
            b_sl, wl_c, ng_m, bw = sl
            f_b = all_freqs[:, b_sl]
            ng  = compute_ng(f_b, k_x)
            with np.errstate(divide='ignore', invalid='ignore'):
                wl = np.where(f_b > 0, a / f_b, np.nan)
            sl_mask = (np.isfinite(ng) & (ng >= NG_LOW) & (ng <= NG_HIGH)
                       & (wl > 1400) & (wl < 1700))
            if np.any(sl_mask):
                f_sl_lo = f_b[sl_mask].min()
                f_sl_hi = f_b[sl_mask].max()
                ax.axhspan(f_sl_lo, f_sl_hi, color='lime', alpha=0.15,
                           label=f'ng={ng_m:.1f}, BW={bw:.0f}nm')

        ax.set_xlim(k_x[0], k_x[-1])
        ax.set_ylim(f_lo, f_hi)
        ax.set_xlabel('k (2\u03c0/a)')
        ax.set_title(
            f't_partial={t_partial_nm:.0f}nm  |  '
            f'width \u00d7{ws:.2f}  (-{ws_pct}%)')
        ax.legend(fontsize=7, loc='upper left')

    axes_flat[0].set_ylabel('Frequency (a/\u03bb)')
    if nrows > 1:
        axes_flat[ncols].set_ylabel('Frequency (a/\u03bb)')

    # Hide any unused subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    if sc is not None:
        cb = fig.colorbar(sc, ax=list(axes_flat[:n]), shrink=0.6)
        cb.set_label('TE fraction (red=TE, blue=TM)')

    a_base = results[0]['params']['a_nm']
    fig.suptitle(
        f'Y-even parity filter — width sweep  '
        f'(a={a_base:.0f}nm, t_partial={t_partial_nm:.0f}nm)',
        fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run one t_partial with all width scales (with caching)
# ---------------------------------------------------------------------------
def run_width_sweep(t_partial_nm, k_interp, num_bands, resolution,
                    width_scales=WIDTH_SCALES):
    """Run (or load from cache) all width-scale variants for one t_partial."""
    results = []
    for ws in width_scales:
        params = dict(BASE_PARAMS)
        cpath  = cache_path(
            {**params, 'width_scale': ws},
            t_partial_nm, resolution, num_bands,
            K_MIN, K_MAX, k_interp)

        if os.path.exists(cpath):
            print(f"  [cache] Loading: {os.path.basename(cpath)}")
            data = load_results(cpath)
        else:
            data = run_mpb_yeven(
                params, t_partial_nm, width_scale=ws,
                k_interp=k_interp, num_bands=num_bands,
                resolution=resolution)
            save_results(data, cpath)

        results.append(data)
    return results


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
def print_summary(results, t_partial_nm):
    print(f"\n{'='*70}")
    print(f"WIDTH SWEEP SUMMARY  (t_partial={t_partial_nm:.0f}nm, y-even parity)")
    print(f"{'='*70}")
    print(f"  {'Scale':>8}  {'h_spine':>8}  {'h_rib':>8}  "
          f"{'Band':>5}  {'wl_c(nm)':>9}  {'ng':>6}  {'BW(nm)':>7}  {'Target?':>8}")
    print(f"  {'-'*72}")
    for data in results:
        ws    = data['width_scale']
        a     = data['params']['a_nm']
        hs    = data['params']['h_spine'] * ws
        hr    = data['params']['h_rib']   * ws
        sl    = find_te_slow_light_band(data)
        ok    = meets_target(data)
        if sl is not None:
            b, wl_c, ng_m, bw = sl
            print(f"  {ws:>8.2f}  {hs:>8.3f}  {hr:>8.3f}  "
                  f"{b:>5d}  {wl_c:>9.1f}  {ng_m:>6.2f}  {bw:>7.1f}  "
                  f"{'YES ***' if ok else 'no':>8}")
        else:
            print(f"  {ws:>8.2f}  {hs:>8.3f}  {hr:>8.3f}  "
                  f"{'—':>5}  {'—':>9}  {'—':>6}  {'—':>7}  {'no':>8}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution',  type=int,  default=RESOLUTION)
    parser.add_argument('--num-bands',   type=int,  default=NUM_BANDS)
    parser.add_argument('--k-interp',    type=int,  default=K_INTERP)
    parser.add_argument('--save-plots',  action='store_true')
    parser.add_argument('--no-show',     action='store_true')
    args = parser.parse_args()

    show = not args.no_show
    out  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Step 1: Run / load t_partial = 140nm (priority 1)                  #
    # ------------------------------------------------------------------ #
    print(f"\n{'#'*70}")
    print(f"#  PRIORITY 1: t_partial = {T_PARTIAL_PRIMARY:.0f} nm")
    print(f"{'#'*70}")

    res_140 = run_width_sweep(
        T_PARTIAL_PRIMARY, args.k_interp, args.num_bands, args.resolution)
    print_summary(res_140, T_PARTIAL_PRIMARY)

    any_pass_140 = any(meets_target(d) for d in res_140)

    sp_140 = (os.path.join(out, f'parity_width_sweep_PE140nm.png')
              if args.save_plots else None)
    plot_width_sweep(res_140, T_PARTIAL_PRIMARY,
                     save_path=sp_140, show=show)

    # ------------------------------------------------------------------ #
    #  Step 2: Fall back to t_partial = 70nm only if 140nm fails target   #
    # ------------------------------------------------------------------ #
    res_70 = None
    if not any_pass_140:
        print(f"\n  >> t_partial={T_PARTIAL_PRIMARY:.0f}nm: NO variant met target.")
        print(f"  >> Falling back to t_partial={T_PARTIAL_FALLBACK:.0f}nm ...\n")

        res_70 = run_width_sweep(
            T_PARTIAL_FALLBACK, args.k_interp, args.num_bands, args.resolution)
        print_summary(res_70, T_PARTIAL_FALLBACK)

        sp_70 = (os.path.join(out, f'parity_width_sweep_PE70nm.png')
                 if args.save_plots else None)
        plot_width_sweep(res_70, T_PARTIAL_FALLBACK,
                         save_path=sp_70, show=show)
    else:
        print(f"\n  >> t_partial={T_PARTIAL_PRIMARY:.0f}nm met target — "
              f"skipping {T_PARTIAL_FALLBACK:.0f}nm fallback.")

    # ------------------------------------------------------------------ #
    #  Step 3: Individual per-width plots for the chosen t_partial         #
    # ------------------------------------------------------------------ #
    chosen_results   = res_140 if any_pass_140 else (res_70 or [])
    chosen_t_partial = T_PARTIAL_PRIMARY if any_pass_140 else T_PARTIAL_FALLBACK

    print(f"\n{'='*70}")
    print(f"SELECTED: t_partial = {chosen_t_partial:.0f} nm")
    print(f"  Generating individual band diagrams per width scale ...")
    print(f"{'='*70}")

    best_data = None
    best_bw   = -1.0

    for data in chosen_results:
        ws     = data['width_scale']
        ws_pct = int(round((1.0 - ws) * 100))
        label  = f"baseline" if ws_pct == 0 else f"-{ws_pct}%_width"
        sp     = (os.path.join(out,
                    f'parity_yeven_PE{chosen_t_partial:.0f}nm_{label}.png')
                  if args.save_plots else None)
        plot_band_colormapped(data, save_path=sp, show=show)

        sl = find_te_slow_light_band(data)
        if sl is not None and sl[3] > best_bw:
            best_bw   = sl[3]
            best_data = data

    # ------------------------------------------------------------------ #
    #  Step 4: Final summary                                               #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print(f"FINAL RESULT  (y-even parity filter, t_partial={chosen_t_partial:.0f}nm)")
    print(f"{'='*70}")

    if best_data is None:
        print("  WARNING: No TE slow-light band found in any configuration.")
        return

    ws  = best_data['width_scale']
    a   = best_data['params']['a_nm']
    sl  = find_te_slow_light_band(best_data)
    b_f, wl_f, ng_f, bw_f = sl

    print(f"  Best width scale : {ws:.2f}  ({int(round((1-ws)*100))}% reduction)")
    print(f"  a_nm             : {a:.1f} nm")
    print(f"  h_spine (scaled) : {best_data['params']['h_spine']*ws:.3f} a")
    print(f"  h_rib   (scaled) : {best_data['params']['h_rib']*ws:.3f} a")
    print(f"  W_rib (Wt=Wb)    : {best_data['params']['W_rib']:.3f} a  (unchanged)")
    print(f"  t_partial        : {chosen_t_partial:.0f} nm")
    print(f"  TE band index    : {b_f}  (y-even sector)")
    print(f"  wl_center        : {wl_f:.1f} nm")
    print(f"  ng_mean          : {ng_f:.2f}")
    print(f"  BW (ng {NG_LOW:.0f}-{NG_HIGH:.0f})  : {bw_f:.1f} nm")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
