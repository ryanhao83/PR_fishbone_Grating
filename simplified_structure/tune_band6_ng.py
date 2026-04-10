"""
tune_band6_ng.py — Sweep h_spine / W_rib / h_rib to maximise flat-ng bandwidth
                   on the confirmed single-mode slow-light band (band 6, y-even)
                   of the 71nm partial-etch y-symmetric fishbone.

Fixed:
  a_nm      = 496.0 nm   (1550nm-aligned, do not sweep)
  t_partial = 71.0  nm   (PDK partial-etch depth, do not sweep)
  parity    = y-even      (run_yeven — TE0 sector only)
  TARGET_BAND = 6         (0-indexed y-even band, confirmed single-mode)

Sweep: ±10% around baseline in 5 steps for each of h_spine, W_rib, h_rib.
  5 × 5 × 5 = 125 combinations; MD5-keyed NPZ caching in cache_3d/.

Metric:
  BW  — wavelength range where ng ∈ [NG_LO, NG_HI] on band 6  (primary, maximise)
  ng_std — std(ng) in that window                               (secondary, minimise)

Output:
  Console  — ranked summary table (top 15 + baseline)
  output/tune_band6_top5.png — band diagram + ng(λ) panel for top-5 configs
  output/tune_band6_best.png — same for single best config

Usage:
  python tune_band6_ng.py --resolution 16 --num-bands 30 --k-interp 20 --save-plots --no-show
"""

import hashlib
import itertools
import json
import os

import numpy as np

# ---------------------------------------------------------------------------
# Constants (mirrors sweep_ysym_partial_etch.py — no top-level meep import)
# ---------------------------------------------------------------------------
N_POLY_SI  = 3.48
N_SIO2     = 1.44
T_SLAB_NM  = 211.0
PAD_Y      = 2.0
PAD_Z      = 1.5
K_MIN      = 0.35
K_MAX      = 0.50

# ---------------------------------------------------------------------------
# Fixed simulation parameters
# ---------------------------------------------------------------------------
A_NM        = 496.0   # nm — do not sweep
T_PARTIAL   = 71.0    # nm — do not sweep
TARGET_BAND = 6       # 0-indexed in y-even eigenspace
NG_TARGET   = 7.0
NG_LO       = 6.5
NG_HI       = 7.5
WL_TARGET   = 1550.0

RESOLUTION  = 16
NUM_BANDS   = 30
K_INTERP_DEFAULT = 20

# Baseline (confirmed single-mode slow-light)
BASELINE = dict(h_spine=0.550, W_rib=0.484, h_rib=0.514)

# Sweep grids — ±10% of baseline in 5 equal steps
H_SPINE_VALS = [round(0.550 * s, 4) for s in [0.90, 0.95, 1.00, 1.05, 1.10]]
W_RIB_VALS   = [round(0.484 * s, 4) for s in [0.90, 0.95, 1.00, 1.05, 1.10]]
H_RIB_VALS   = [round(0.514 * s, 4) for s in [0.90, 0.95, 1.00, 1.05, 1.10]]


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def _cache_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(here, 'cache_3d')
    os.makedirs(d, exist_ok=True)
    return d


def _cache_key(h_spine, W_rib, h_rib, resolution, num_bands, k_min, k_max, k_interp):
    d = dict(
        a_nm=A_NM, t_partial_nm=T_PARTIAL,
        h_spine=h_spine, W_rib=W_rib, h_rib=h_rib,
        resolution=resolution, num_bands=num_bands,
        k_min=k_min, k_max=k_max, k_interp=k_interp,
        parity='yeven', target_band=TARGET_BAND,
    )
    s = json.dumps(d, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def _cache_path(h_spine, W_rib, h_rib, resolution, num_bands, k_min, k_max, k_interp):
    h = _cache_key(h_spine, W_rib, h_rib, resolution, num_bands, k_min, k_max, k_interp)
    fname = f"tune_band6_res{resolution}_nb{num_bands}_{h}.npz"
    return os.path.join(_cache_dir(), fname)


def _save(data, path):
    flat = {
        'all_freqs':  data['all_freqs'],
        'k_x':        data['k_x'],
        'h_spine':    data['h_spine'],
        'W_rib':      data['W_rib'],
        'h_rib':      data['h_rib'],
        'resolution': data['resolution'],
        'num_bands':  data['num_bands'],
    }
    np.savez_compressed(path, **flat)
    print(f"    [cache] saved {os.path.basename(path)}")


def _load(path):
    npz = np.load(path, allow_pickle=False)
    return {
        'all_freqs':  npz['all_freqs'],
        'k_x':        npz['k_x'],
        'h_spine':    float(npz['h_spine']),
        'W_rib':      float(npz['W_rib']),
        'h_rib':      float(npz['h_rib']),
        'resolution': int(npz['resolution']),
        'num_bands':  int(npz['num_bands']),
    }


# ---------------------------------------------------------------------------
# Geometry builder (inlined from run_71nm_yeven.py)
# ---------------------------------------------------------------------------
def _build_geometry(h_spine, W_rib, h_rib):
    """Y-symmetric fishbone with t_partial=71nm.  Returns (lattice, geometry)."""
    import meep as mp

    t_slab   = T_SLAB_NM / A_NM
    t_partial = T_PARTIAL / A_NM
    sy = 2.0 * (h_spine + h_rib) + 2.0 * PAD_Y
    sz = t_slab + 2.0 * PAD_Z

    Si   = mp.Medium(index=N_POLY_SI)
    SiO2 = mp.Medium(index=N_SIO2)
    lattice = mp.Lattice(size=mp.Vector3(1, sy, sz))

    geometry = [
        # SiO2 substrate
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, PAD_Z),
                 center=mp.Vector3(0, 0, -(t_slab / 2 + PAD_Z / 2)),
                 material=SiO2),
        # SiO2 top cladding
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, PAD_Z),
                 center=mp.Vector3(0, 0, t_slab / 2 + PAD_Z / 2),
                 material=SiO2),
        # Partial-etch Si layer (bottom of slab)
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, t_partial),
                 center=mp.Vector3(0, 0, -t_slab / 2.0 + t_partial / 2.0),
                 material=Si),
        # Central spine (full slab)
        mp.Block(size=mp.Vector3(mp.inf, 2.0 * h_spine, t_slab),
                 center=mp.Vector3(0, 0, 0),
                 material=Si),
        # +y rib
        mp.Block(size=mp.Vector3(W_rib, h_rib, t_slab),
                 center=mp.Vector3(0, h_spine + h_rib / 2.0, 0),
                 material=Si),
        # -y rib (Wt = Wb enforced)
        mp.Block(size=mp.Vector3(W_rib, h_rib, t_slab),
                 center=mp.Vector3(0, -(h_spine + h_rib / 2.0), 0),
                 material=Si),
    ]
    return lattice, geometry


# ---------------------------------------------------------------------------
# MPB runner (y-even parity, 71nm partial etch)
# ---------------------------------------------------------------------------
def run_one(h_spine, W_rib, h_rib, resolution, num_bands, k_interp):
    """Run MPB run_yeven() for a single (h_spine, W_rib, h_rib) combination."""
    import meep as mp
    import meep.mpb as mpb

    lattice, geometry = _build_geometry(h_spine, W_rib, h_rib)
    k_points = mp.interpolate(k_interp, [mp.Vector3(K_MIN), mp.Vector3(K_MAX)])

    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=k_points,
        resolution=resolution,
        num_bands=num_bands,
    )

    # Single run_yeven() call — MPB computes all k-points at once
    ms.run_yeven()

    all_freqs = np.array(ms.all_freqs)   # shape (num_k, num_bands)
    k_x = np.array([k.x for k in k_points])

    return {
        'all_freqs': all_freqs,
        'k_x':       k_x,
        'h_spine':   h_spine,
        'W_rib':     W_rib,
        'h_rib':     h_rib,
        'resolution': resolution,
        'num_bands':  num_bands,
    }


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def compute_ng(f_band, k_x):
    df_dk = np.gradient(f_band, k_x)
    df_dk = np.where(df_dk == 0.0, np.nan, df_dk)
    return 1.0 / df_dk


def analyze_band6(data):
    """Extract slow-light metrics for TARGET_BAND from data.

    Returns dict with keys:
      bw_nm, ng_mean, ng_std, wl_center, ng_arr, wl_arr, sl_mask
    or None if band 6 does not show slow light.
    """
    all_freqs = data['all_freqs']
    k_x       = data['k_x']
    nb        = all_freqs.shape[1]

    if TARGET_BAND >= nb:
        return None

    f_b = all_freqs[:, TARGET_BAND]
    if np.any(f_b <= 1e-6):
        return None

    ng = compute_ng(f_b, k_x)
    with np.errstate(divide='ignore', invalid='ignore'):
        wl = np.where(f_b > 0, A_NM / f_b, np.nan)

    # Slow-light mask: ng in target window, physical wavelength in C-band
    sl_mask = (np.isfinite(ng) & (ng >= NG_LO) & (ng <= NG_HI)
               & (wl > 1400.0) & (wl < 1700.0))

    if np.sum(sl_mask) < 2:
        return None

    bw_nm    = wl[sl_mask].max() - wl[sl_mask].min()
    ng_mean  = float(np.nanmean(ng[sl_mask]))
    ng_std   = float(np.nanstd(ng[sl_mask]))
    wl_c     = (wl[sl_mask].max() + wl[sl_mask].min()) / 2.0

    return dict(bw_nm=bw_nm, ng_mean=ng_mean, ng_std=ng_std,
                wl_center=wl_c, ng_arr=ng, wl_arr=wl, sl_mask=sl_mask)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_config(ax_band, ax_ng, data, metrics, label=""):
    """Fill one (band diagram, ng) axes pair for a single config."""
    import matplotlib.pyplot as plt

    all_freqs = data['all_freqs']
    k_x       = data['k_x']
    nb        = all_freqs.shape[1]
    f_1550    = A_NM / WL_TARGET

    f_lo, f_hi = 0.25, 0.40

    # Light cone
    k_lc = np.linspace(k_x[0], k_x[-1], 100)
    f_lc = k_lc / N_SIO2
    ax_band.fill_between(k_lc, np.maximum(f_lc, f_lo), f_hi,
                         alpha=0.08, color='gray')
    ax_band.plot(k_lc, f_lc, color='gray', lw=0.6)

    # All y-even bands in light gray
    for b in range(nb):
        f_b = all_freqs[:, b]
        ax_band.plot(k_x, f_b, color='lightgray', lw=0.8, zorder=1)

    # Target band (band 6) highlighted in red
    f_tgt = all_freqs[:, TARGET_BAND]
    ax_band.plot(k_x, f_tgt, color='red', lw=1.8, zorder=3,
                 label=f'Band {TARGET_BAND} (target)')

    # 1550nm reference line
    ax_band.axhline(f_1550, color='green', ls=':', lw=1.0, label='1550 nm')

    # Slow-light frequency shading
    if metrics is not None and np.any(metrics['sl_mask']):
        f_sl = f_tgt[metrics['sl_mask']]
        ax_band.axhspan(f_sl.min(), f_sl.max(), color='lime',
                        alpha=0.20, zorder=2,
                        label=f"ng∈[{NG_LO},{NG_HI}], BW={metrics['bw_nm']:.1f}nm")

    ax_band.set_xlim(k_x[0], k_x[-1])
    ax_band.set_ylim(f_lo, f_hi)
    ax_band.set_ylabel('Freq (a/λ)', fontsize=8)
    ax_band.legend(fontsize=7, loc='upper left')
    ax_band.set_title(label, fontsize=8)
    ax_band.tick_params(labelbottom=False)

    # ng(λ) panel
    if metrics is not None:
        wl_arr = metrics['wl_arr']
        ng_arr = metrics['ng_arr']
        mask   = np.isfinite(ng_arr) & (ng_arr > 0) & (ng_arr < 50) & (wl_arr > 1400) & (wl_arr < 1700)
        if np.any(mask):
            ax_ng.plot(wl_arr[mask], ng_arr[mask], 'r-', lw=1.5)
            ax_ng.axhline(NG_LO, color='lime',  ls='--', lw=0.8)
            ax_ng.axhline(NG_HI, color='lime',  ls='--', lw=0.8)
            ax_ng.axvline(WL_TARGET, color='green', ls=':', lw=1.0)
            ax_ng.set_ylim(4, 12)
            ax_ng.set_xlim(1400, 1700)
            ax_ng.set_ylabel('ng', fontsize=8)
            ax_ng.set_xlabel('λ (nm)', fontsize=8)
    else:
        ax_ng.text(0.5, 0.5, 'No slow-light', ha='center', va='center',
                   transform=ax_ng.transAxes, fontsize=8)
        ax_ng.set_xlim(1400, 1700)
        ax_ng.set_xlabel('λ (nm)', fontsize=8)


def plot_top_n(top_entries, n=5, save_path=None, show=True):
    """n-config grid: each column = (band diagram, ng panel)."""
    import matplotlib.pyplot as plt

    nc = min(n, len(top_entries))
    fig, axes = plt.subplots(2, nc, figsize=(5 * nc, 7),
                             gridspec_kw={'height_ratios': [2, 1]})
    if nc == 1:
        axes = axes.reshape(2, 1)

    for col, entry in enumerate(top_entries[:nc]):
        data    = entry['data']
        metrics = entry['metrics']
        rank    = entry['rank']
        hs, wr, hr = entry['h_spine'], entry['W_rib'], entry['h_rib']
        bw  = metrics['bw_nm'] if metrics else 0.0
        ng  = metrics['ng_mean'] if metrics else float('nan')
        lbl = (f"#{rank}  hs={hs:.3f} Wr={wr:.3f} hr={hr:.3f}\n"
               f"ng={ng:.2f}  BW={bw:.1f}nm")
        _plot_config(axes[0, col], axes[1, col], data, metrics, label=lbl)

    fig.suptitle(
        f'Top-{nc} y-even band-6 slow-light configs  '
        f'(a={A_NM:.0f}nm, t_partial={T_PARTIAL:.0f}nm, ng∈[{NG_LO},{NG_HI}])',
        fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_single(entry, save_path=None, show=True):
    """Single config: band diagram + ng panel stacked."""
    import matplotlib.pyplot as plt

    fig, (ax_band, ax_ng) = plt.subplots(2, 1, figsize=(7, 7),
                                          gridspec_kw={'height_ratios': [2, 1]})
    data    = entry['data']
    metrics = entry['metrics']
    rank    = entry['rank']
    hs, wr, hr = entry['h_spine'], entry['W_rib'], entry['h_rib']
    bw = metrics['bw_nm'] if metrics else 0.0
    ng = metrics['ng_mean'] if metrics else float('nan')
    lbl = (f"#{rank} BEST  h_spine={hs:.3f}  W_rib={wr:.3f}  h_rib={hr:.3f}  "
           f"ng={ng:.2f}  BW={bw:.1f}nm")
    _plot_config(ax_band, ax_ng, data, metrics, label=lbl)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_table(ranked, baseline_bw):
    hdr = (f"  {'#':>3}  {'h_spine':>8}  {'W_rib':>7}  {'h_rib':>7}  "
           f"{'wl_c(nm)':>9}  {'ng':>6}  {'ng_std':>7}  {'BW(nm)':>7}  {'ΔBW':>7}")
    sep = '  ' + '-' * (len(hdr) - 2)
    print(f"\n{'='*75}")
    print(f"  SLOW-LIGHT SWEEP RESULTS  "
          f"(band {TARGET_BAND}, y-even, ng∈[{NG_LO},{NG_HI}])")
    print(f"  a_nm={A_NM:.0f}  t_partial={T_PARTIAL:.0f}nm")
    print(f"{'='*75}")
    print(hdr)
    print(sep)

    for e in ranked:
        m = e['metrics']
        if m is None:
            print(f"  {e['rank']:>3}  {e['h_spine']:>8.3f}  {e['W_rib']:>7.3f}  "
                  f"{e['h_rib']:>7.3f}  {'—':>9}  {'—':>6}  {'—':>7}  {'—':>7}  {'—':>7}")
        else:
            delta = m['bw_nm'] - baseline_bw
            flag  = ' ***' if m['bw_nm'] >= 1.5 * baseline_bw else ''
            print(f"  {e['rank']:>3}  {e['h_spine']:>8.3f}  {e['W_rib']:>7.3f}  "
                  f"{e['h_rib']:>7.3f}  {m['wl_center']:>9.1f}  "
                  f"{m['ng_mean']:>6.2f}  {m['ng_std']:>7.3f}  "
                  f"{m['bw_nm']:>7.1f}  {delta:>+7.1f}{flag}")

    print(f"{'='*75}")
    print(f"  Baseline BW = {baseline_bw:.1f} nm  "
          f"(h_spine={BASELINE['h_spine']}, W_rib={BASELINE['W_rib']}, h_rib={BASELINE['h_rib']})")
    print(f"{'='*75}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int,   default=RESOLUTION)
    parser.add_argument('--num-bands',  type=int,   default=NUM_BANDS)
    parser.add_argument('--k-interp',   type=int,   default=K_INTERP_DEFAULT)
    parser.add_argument('--save-plots', action='store_true')
    parser.add_argument('--no-show',    action='store_true')
    args = parser.parse_args()

    show = not args.no_show
    out  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out, exist_ok=True)

    res   = args.resolution
    nb    = args.num_bands
    ki    = args.k_interp

    combos = list(itertools.product(H_SPINE_VALS, W_RIB_VALS, H_RIB_VALS))
    total  = len(combos)

    print(f"\n{'='*75}")
    print(f"  Geometry sweep: band {TARGET_BAND}, y-even parity")
    print(f"  {total} combinations  |  res={res}  nb={nb}  k-interp={ki}")
    print(f"{'='*75}")

    results = []

    for i, (hs, wr, hr) in enumerate(combos):
        cpath = _cache_path(hs, wr, hr, res, nb, K_MIN, K_MAX, ki)
        tag   = f"[{i+1:3d}/{total}]  hs={hs:.3f} Wr={wr:.3f} hr={hr:.3f}"

        if os.path.exists(cpath):
            print(f"  {tag}  [cache]")
            data = _load(cpath)
        else:
            print(f"  {tag}  running MPB ...")
            data = run_one(hs, wr, hr, res, nb, ki)
            _save(data, cpath)

        metrics = analyze_band6(data)
        results.append(dict(
            h_spine=hs, W_rib=wr, h_rib=hr,
            data=data, metrics=metrics, rank=0))

    # --- Rank: sort by BW desc, then ng_std asc ---
    def sort_key(e):
        m = e['metrics']
        if m is None:
            return (-1.0, 1e9)
        return (-m['bw_nm'], m['ng_std'])

    results.sort(key=sort_key)
    for i, e in enumerate(results):
        e['rank'] = i + 1

    # Baseline entry
    bl_entry = next(
        (e for e in results
         if abs(e['h_spine'] - BASELINE['h_spine']) < 1e-6
         and abs(e['W_rib']  - BASELINE['W_rib'])   < 1e-6
         and abs(e['h_rib']  - BASELINE['h_rib'])   < 1e-6),
        None)
    bl_bw = bl_entry['metrics']['bw_nm'] if (bl_entry and bl_entry['metrics']) else 0.0

    # Show top 15 + baseline
    top15 = results[:15]
    if bl_entry and bl_entry not in top15:
        bl_entry['rank'] = results.index(bl_entry) + 1
        top15.append(bl_entry)

    print_table(top15, bl_bw)

    # --- Plots ---
    top5 = [e for e in results if e['metrics'] is not None][:5]

    if top5:
        sp5 = os.path.join(out, 'tune_band6_top5.png') if args.save_plots else None
        plot_top_n(top5, n=5, save_path=sp5, show=show)

        sp1 = os.path.join(out, 'tune_band6_best.png') if args.save_plots else None
        plot_single(top5[0], save_path=sp1, show=show)
    else:
        print("  WARNING: No config showed slow-light on band 6.")


if __name__ == '__main__':
    main()
