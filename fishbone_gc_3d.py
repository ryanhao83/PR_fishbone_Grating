"""
3D MPB simulation of fishbone grating waveguide.

Full 3D simulation with SiO2 substrate, 220nm Si slab, and air cladding.
Uses Y+Z mirror symmetry (方案E) to reduce computation to 1/4.
Results are cached to .npz files.

Run this AFTER identifying the correct h_spine and target band from the
fast 2D scan (fishbone_gc_2d.py).

Usage:
    python fishbone_gc_3d.py --run                           # Run with defaults
    python fishbone_gc_3d.py --run --h-spine 0.40 --resolution 32
    python fishbone_gc_3d.py --plot-only                     # Plot from cache
    python fishbone_gc_3d.py --run --sweep-spine             # Sweep h_spine
    python fishbone_gc_3d.py --plot-only --sweep-spine
"""

import argparse
import os
import hashlib
import json

import meep as mp
import meep.mpb as mpb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# ==========================================
# Geometry builder
# ==========================================

def get_default_params():
    """Return default fishbone geometry parameters (all in units of a)."""
    return dict(
        a_nm=420.0,
        w1=0.52, w2=0.80, w3=0.68,
        h1=1.70, h2=0.85, h3=1.75,
        h_spine=0.40,        # Tunable — determine from 2D scan
        t_slab=220.0 / 420.0,  # 220nm Si slab thickness in units of a
        n_Si=3.48,
        n_SiO2=1.44,
        pad_y=3.0,
        pad_z=2.0,
    )


def build_geometry_3d(params):
    """Build 3D MPB geometry + lattice for the fishbone structure.

    Coordinate system:
      - x: periodic direction (period = 1a)
      - y: transverse (in-plane)
      - z: vertical (out-of-plane)
    Slab centered at z=0. SiO2 substrate below (z < -t_slab/2).
    """
    p = params
    t_slab = p['t_slab']
    sy = 2 * (p['h_spine'] + p['h1'] + p['h2'] + p['h3']) + 2 * p['pad_y']
    sz = t_slab + 2 * p['pad_z']

    lattice = mp.Lattice(size=mp.Vector3(1, sy, sz))

    Si = mp.Medium(index=p['n_Si'])
    SiO2 = mp.Medium(index=p['n_SiO2'])

    geometry = []

    # SiO2 substrate: fills z < -t_slab/2
    geometry.append(
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, sz / 2),
                 center=mp.Vector3(0, 0, -t_slab / 2 - sz / 4),
                 material=SiO2)
    )

    # Central continuous spine waveguide
    geometry.append(
        mp.Block(size=mp.Vector3(mp.inf, 2 * p['h_spine'], t_slab),
                 center=mp.Vector3(0, 0, 0),
                 material=Si)
    )

    # Periodic ribs on both sides
    for sign in [+1, -1]:
        y0 = sign * (p['h_spine'] + p['h3'] / 2)
        geometry.append(
            mp.Block(size=mp.Vector3(p['w3'], p['h3'], t_slab),
                     center=mp.Vector3(0, y0, 0), material=Si))

        y1 = sign * (p['h_spine'] + p['h3'] + p['h2'] / 2)
        geometry.append(
            mp.Block(size=mp.Vector3(p['w2'], p['h2'], t_slab),
                     center=mp.Vector3(0, y1, 0), material=Si))

        y2 = sign * (p['h_spine'] + p['h3'] + p['h2'] + p['h1'] / 2)
        geometry.append(
            mp.Block(size=mp.Vector3(p['w1'], p['h1'], t_slab),
                     center=mp.Vector3(0, y2, 0), material=Si))

    return lattice, geometry, sy, sz


# ==========================================
# Simulation runner
# ==========================================

def params_hash(params):
    s = json.dumps(params, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def run_mpb_3d(params, k_min=0.36, k_max=0.50, k_interp=25,
               num_bands=10, resolution=16):
    """Run 3D MPB simulation with Y and Z symmetry exploitation."""
    lattice, geometry, sy, sz = build_geometry_3d(params)
    k_points = mp.interpolate(k_interp,
                              [mp.Vector3(k_min), mp.Vector3(k_max)])

    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=k_points,
        resolution=resolution,
        num_bands=num_bands,
    )

    print(f"Running 3D MPB: resolution={resolution}, k_points={len(k_points)}, "
          f"num_bands={num_bands}, h_spine={params['h_spine']:.3f}")
    print(f"  Supercell: 1 x {sy:.2f} x {sz:.2f} (a units)")

    # TE-like guided modes: E is mostly in-plane.
    # For this symmetric structure:
    #   Y-even + Z-odd => TE-like fundamental mode
    #   Y-even + Z-even => TM-like
    # Run the TE-like symmetry sector (most relevant for photonic waveguides)
    ms.run_yeven_zodd()
    freqs_te = np.array(ms.all_freqs)

    # Also run TM-like for completeness
    ms.run_yeven_zeven()
    freqs_tm = np.array(ms.all_freqs)

    k_x = np.array([k.x for k in k_points])

    # Get epsilon at z=0 slice for visualization
    eps = np.array(ms.get_epsilon())

    return dict(
        freqs_te=freqs_te,
        freqs_tm=freqs_tm,
        k_x=k_x,
        epsilon=eps,
        sy=sy,
        sz=sz,
        params=params,
        resolution=resolution,
        num_bands=num_bands,
    )


# ==========================================
# Cache I/O
# ==========================================

def get_cache_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_3d')
    os.makedirs(d, exist_ok=True)
    return d


def format_float_for_filename(value):
    return f"{value:.3f}".replace('.', 'p')


def cache_path(params, tag=''):
    h = params_hash(params)
    h_spine_str = format_float_for_filename(params['h_spine'])
    if tag:
        name = f"fishbone3d_hspine{h_spine_str}_{tag}_{h}.npz"
    else:
        name = f"fishbone3d_hspine{h_spine_str}_{h}.npz"
    return os.path.join(get_cache_dir(), name)


def save_results(data, path):
    save_dict = {}
    for k, v in data.items():
        if k == 'params':
            save_dict['params_json'] = json.dumps(v)
        else:
            save_dict[k] = v
    np.savez(path, **save_dict)
    print(f"Results saved to {path}")


def load_results(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file not found: {path}")
    d = dict(np.load(path, allow_pickle=True))
    if 'params_json' in d:
        d['params'] = json.loads(str(d['params_json']))
        del d['params_json']
    for k in ['sy', 'sz', 'resolution', 'num_bands']:
        if k in d:
            d[k] = float(d[k])
    print(f"Results loaded from {path}")
    return d


# ==========================================
# Post-processing
# ==========================================

def compute_ng(f_band, k_x, a_nm):
    df_dk = np.gradient(f_band, k_x)
    df_dk[df_dk == 0] = np.nan
    ng = 1.0 / df_dk
    wavelengths = a_nm / f_band
    return ng, wavelengths


# ==========================================
# Plotting
# ==========================================

def plot_bands_and_ng(data, target_band=0, save_prefix='fishbone3d'):
    params = data['params']
    a_nm = params['a_nm']
    n_SiO2 = params['n_SiO2']
    k_x = data['k_x']
    freqs_te = data['freqs_te']
    freqs_tm = data['freqs_tm']
    eps = data['epsilon']
    sy = float(data['sy'])
    sz = float(data['sz'])
    num_bands = int(data['num_bands'])

    f_band = freqs_te[:, target_band]
    ng, wavelengths = compute_ng(f_band, k_x, a_nm)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax_geo, ax_band, ax_ng = axes

    # (0) Geometry — z=0 slice
    slide_z = eps.shape[2] // 2
    eps_slice = eps[:, :, slide_z].T
    ax_geo.imshow(eps_slice, extent=[-0.5, 0.5, -sy / 2, sy / 2],
                  cmap='gray', origin='lower')
    ax_geo.set_xlabel('x (a)')
    ax_geo.set_ylabel('y (a)')
    ax_geo.set_title(f'Geometry z=0 (h_spine={params["h_spine"]:.2f}a)')

    # (a) Band diagram
    light_line = k_x / n_SiO2
    ax_band.fill_between(k_x, light_line, 1.0, color='gray', alpha=0.3,
                         label='Light cone')

    for b in range(min(num_bands, freqs_te.shape[1])):
        lbl_te = f'TE band {b}' if b == target_band else None
        lbl_tm = f'TM band {b}' if b == target_band else None
        ax_band.plot(k_x, freqs_te[:, b], 'r.-', markersize=2, linewidth=0.8,
                     alpha=0.7, label=lbl_te)
        ax_band.plot(k_x, freqs_tm[:, b], 'b.--', markersize=2, linewidth=0.8,
                     alpha=0.5, label=lbl_tm)

    ax_band.set_xlabel(r'Wave vector ($2\pi/a$)')
    ax_band.set_ylabel(r'Normalized frequency ($a/\lambda$)')
    ax_band.set_title('(a) Band Diagram (3D)')
    ax_band.set_xlim([k_x.min(), k_x.max()])
    ax_band.set_ylim([0.25, 0.28])
    ax_band.grid(True, alpha=0.3)
    ax_band.legend(fontsize=7, loc='upper left')

    # (b) Group index
    valid = (ng > 0) & (ng < 160) & np.isfinite(ng)
    ax_ng.plot(wavelengths[valid], ng[valid], 'ro-', markersize=3)
    ax_ng.set_xlabel('Wavelength (nm)')
    ax_ng.set_ylabel(r'Group index $n_g$')
    ax_ng.set_title('(b) Group Index vs Wavelength (3D)')
    ax_ng.set_xlim([1550, 1630])
    ax_ng.set_ylim([0, 160])
    ax_ng.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'{save_prefix}_hspine{params["h_spine"]:.2f}.png'
    plt.savefig(fname, dpi=200)
    print(f"Figure saved: {fname}")
    plt.show()


def plot_sweep_ng(all_data, target_band=0, save_prefix='fishbone3d_sweep'):
    fig, (ax_band, ax_ng) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_data)))

    for i, data in enumerate(all_data):
        params = data['params']
        a_nm = params['a_nm']
        k_x = data['k_x']
        freqs_te = data['freqs_te']

        f_band = freqs_te[:, target_band]
        ng, wavelengths = compute_ng(f_band, k_x, a_nm)

        label = f'h_spine={params["h_spine"]:.2f}a'
        ax_band.plot(k_x, f_band, '.-', color=colors[i], markersize=2,
                     label=label)

        valid = (ng > 0) & (ng < 160) & np.isfinite(ng)
        ax_ng.plot(wavelengths[valid], ng[valid], 'o-', color=colors[i],
                   markersize=3, label=label)

    n_SiO2 = all_data[0]['params']['n_SiO2']
    k_x = all_data[0]['k_x']
    light_line = k_x / n_SiO2
    ax_band.fill_between(k_x, light_line, 1.0, color='gray', alpha=0.2,
                         label='Light cone')

    ax_band.set_xlabel(r'Wave vector ($2\pi/a$)')
    ax_band.set_ylabel(r'Normalized frequency ($a/\lambda$)')
    ax_band.set_title('(a) 3D Band — h_spine sweep')
    ax_band.set_xlim([k_x.min(), k_x.max()])
    ax_band.set_ylim([0.24, 0.29])
    ax_band.grid(True, alpha=0.3)
    ax_band.legend(fontsize=7)

    ax_ng.set_xlabel('Wavelength (nm)')
    ax_ng.set_ylabel(r'Group index $n_g$')
    ax_ng.set_title('(b) 3D Group Index — h_spine sweep')
    ax_ng.set_xlim([1550, 1630])
    ax_ng.set_ylim([0, 160])
    ax_ng.grid(True, alpha=0.3)
    ax_ng.legend(fontsize=7)

    plt.tight_layout()
    fname = f'{save_prefix}.png'
    plt.savefig(fname, dpi=200)
    print(f"Sweep figure saved: {fname}")
    plt.show()


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='3D Fishbone Grating MPB Simulation')
    parser.add_argument('--run', action='store_true',
                        help='Run MPB simulation')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only plot from cached results')
    parser.add_argument('--sweep-spine', action='store_true',
                        help='Sweep h_spine values')
    parser.add_argument('--h-spine', type=float, default=None,
                        help='Override h_spine value (units of a)')
    parser.add_argument('--target-band', type=int, default=0,
                        help='Band index to analyze (default: 0)')
    parser.add_argument('--resolution', type=int, default=16,
                        help='MPB resolution (default: 16, use 32 for publication)')
    parser.add_argument('--k-interp', type=int, default=25,
                        help='Number of interpolated k-points (default: 25)')
    parser.add_argument('--num-bands', type=int, default=10,
                        help='Number of bands to compute (default: 10)')
    args = parser.parse_args()

    if args.sweep_spine:
        spine_values = [0.20, 0.30, 0.40, 0.50, 0.60]
    else:
        spine_val = args.h_spine if args.h_spine is not None else 0.40
        spine_values = [spine_val]

    all_data = []

    for h_sp in spine_values:
        params = get_default_params()
        params['h_spine'] = h_sp

        cpath = cache_path(params, tag=f'res{args.resolution}')

        # Auto-run if: --run flag set, OR no cache exists and --plot-only not set
        need_run = (args.run and not args.plot_only) or \
                   (not args.plot_only and not os.path.exists(cpath))

        if need_run:
            if not args.run:
                print(f"No cache found for h_spine={h_sp:.2f} — running simulation automatically.")
            data = run_mpb_3d(params, k_interp=args.k_interp,
                              num_bands=args.num_bands,
                              resolution=args.resolution)
            save_results(data, cpath)
        else:
            try:
                data = load_results(cpath)
            except FileNotFoundError:
                print(f"No cache for h_spine={h_sp:.2f}. Run with --run first.")
                continue

        all_data.append(data)

    if not all_data:
        print("No data available. Run with --run to generate data first.")
        return

    if args.sweep_spine and len(all_data) > 1:
        plot_sweep_ng(all_data, target_band=args.target_band)
    else:
        for data in all_data:
            plot_bands_and_ng(data, target_band=args.target_band)


if __name__ == '__main__':
    main()
