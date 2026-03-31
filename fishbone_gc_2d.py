"""
2D MPB simulation of fishbone grating waveguide.

Uses effective index method (方案F) to reduce 3D -> 2D for fast parameter sweeps.
Uses Y-mirror symmetry (方案E) to halve the computation.
Results are cached to .npz files so plotting doesn't require re-running MPB.

Usage:
    python fishbone_gc_2d.py --run                  # Run simulation + plot
    python fishbone_gc_2d.py --plot-only             # Plot from cached data
    python fishbone_gc_2d.py --run --sweep-spine     # Sweep h_spine values
    python fishbone_gc_2d.py --plot-only --sweep-spine
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


# =====================================================================
# >>>  ALL CONTROL PARAMETERS — modify ONLY this section  <<<
# =====================================================================

# --- Geometry parameters (all in units of a, except a_nm) ---
A_NM        = 420.0       # lattice constant in nm
W1          = 0.52        # rib width 1 (x-direction)
W2          = 0.80        # rib width 2
W3          = 0.68        # rib width 3
H1          = 1.70        # rib height 1 (y-direction, transverse extent)
H2          = 0.85        # rib height 2
H3          = 1.75        # rib height 3
H_SPINE     = 0.40        # half-width of central spine (tunable)
N_EFF       = 2.85        # effective index of 220nm Si slab TE mode at 1550nm
N_SIO2      = 1.44        # cladding / substrate index
PAD_Y       = 3.0         # supercell padding in y

# --- Simulation control ---
RESOLUTION  = 32           # MPB spatial resolution
NUM_BANDS   = 18           # number of bands to compute
K_MIN       = 0.4         # k sweep start
K_MAX       = 0.50         # k sweep end
K_INTERP    = 30           # number of interpolated k-points

# --- Analysis / plotting ---
TARGET_BAND = 11            # band index to analyze (0-based)

# --- Spine sweep values ---
SPINE_SWEEP_VALUES = [0.36, 0.38, 0.40, 0.42, 0.44, 0.46]

# --- Field export ---
FIELD_K_POINT = 0.48        # k-point at which to extract mode fields

# =====================================================================


# ==========================================
# Geometry builder
# ==========================================

def get_default_params():
    """Return default fishbone geometry parameters (all in units of a)."""
    return dict(
        a_nm=A_NM,
        w1=W1, w2=W2, w3=W3,
        h1=H1, h2=H2, h3=H3,
        h_spine=H_SPINE,
        n_eff=N_EFF,
        n_SiO2=N_SIO2,
        pad_y=PAD_Y,
    )


def build_geometry_2d(params):
    """Build 2D MPB geometry + lattice for the fishbone structure.

    In 2D effective-index model:
      - x is the periodic direction (period = 1a)
      - y is the transverse direction
      - no z dimension
      - Si ribs use Medium(index=n_eff), background is air (index=1)
    """
    p = params
    sy = 2 * (p['h_spine'] + p['h1'] + p['h2'] + p['h3']) + 2 * p['pad_y']

    lattice = mp.Lattice(size=mp.Vector3(1, sy))

    mat_si = mp.Medium(index=p['n_eff'])

    geometry = []

    # Central continuous spine waveguide (extends infinitely in x)
    geometry.append(
        mp.Block(size=mp.Vector3(mp.inf, 2 * p['h_spine']),
                 center=mp.Vector3(0, 0),
                 material=mat_si)
    )

    # Periodic ribs on both sides (symmetric about y=0)
    for sign in [+1, -1]:
        y0 = sign * (p['h_spine'] + p['h3'] / 2)
        geometry.append(
            mp.Block(size=mp.Vector3(p['w3'], p['h3']),
                     center=mp.Vector3(0, y0), material=mat_si))

        y1 = sign * (p['h_spine'] + p['h3'] + p['h2'] / 2)
        geometry.append(
            mp.Block(size=mp.Vector3(p['w2'], p['h2']),
                     center=mp.Vector3(0, y1), material=mat_si))

        y2 = sign * (p['h_spine'] + p['h3'] + p['h2'] + p['h1'] / 2)
        geometry.append(
            mp.Block(size=mp.Vector3(p['w1'], p['h1']),
                     center=mp.Vector3(0, y2), material=mat_si))

    return lattice, geometry, sy


# ==========================================
# Simulation runner
# ==========================================

def params_hash(params):
    """Create a short deterministic hash from the parameter dict for cache naming."""
    s = json.dumps(params, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def run_mpb_2d(params, k_min=K_MIN, k_max=K_MAX, k_interp=K_INTERP,
               num_bands=NUM_BANDS, resolution=RESOLUTION):
    """Run 2D MPB simulation and return results dict."""
    lattice, geometry, sy = build_geometry_2d(params)
    k_points = mp.interpolate(k_interp,
                              [mp.Vector3(k_min), mp.Vector3(k_max)])

    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=k_points,
        resolution=resolution,
        num_bands=num_bands,
    )

    print(f"Running 2D MPB: resolution={resolution}, k_points={len(k_points)}, "
          f"num_bands={num_bands}, h_spine={params['h_spine']:.3f}")

    # Use Y-even symmetry (方案E): the TE guided mode of this symmetric
    # waveguide has E_y even about y=0 (i.e. the dominant Ey component is
    # symmetric). This halves the computational domain.
    ms.run_yeven()

    freqs_even = np.array(ms.all_freqs)
    k_x = np.array([k.x for k in k_points])

    # Also get epsilon for geometry visualization
    eps = np.array(ms.get_epsilon())

    # Run Y-odd as well to capture all guided modes
    # ms.run_yodd()
    # freqs_odd = np.array(ms.all_freqs)

    return dict(
        freqs_even=freqs_even,
        # freqs_odd=freqs_odd,
        k_x=k_x,
        epsilon=eps,
        sy=sy,
        params=params,
        resolution=resolution,
        num_bands=num_bands,
    )


def run_mpb_fields_2d(params, k_at=FIELD_K_POINT,
                      num_bands=NUM_BANDS, resolution=RESOLUTION):
    """Run 2D MPB at a single k-point and extract E/H fields for all bands.

    Returns a dict with:
        epsilon : 2D array (nx, ny)
        freqs   : 1D array (num_bands,)
        efield_b{i} : complex array (nx, ny, 3) for each band i
        hfield_b{i} : complex array (nx, ny, 3) for each band i
        k_at, sy, params, resolution, num_bands
    """
    lattice, geometry, sy = build_geometry_2d(params)

    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=[mp.Vector3(k_at)],
        resolution=resolution,
        num_bands=num_bands,
    )

    print(f"Running 2D MPB fields: resolution={resolution}, k={k_at}, "
          f"num_bands={num_bands}, h_spine={params['h_spine']:.3f}")
    ms.run_yeven()

    freqs = np.array(ms.all_freqs[0])  # shape (num_bands,)
    eps = np.array(ms.get_epsilon())

    result = dict(
        epsilon=eps,
        freqs=freqs,
        k_at=k_at,
        sy=sy,
        params=params,
        resolution=resolution,
        num_bands=num_bands,
    )

    for b in range(1, num_bands + 1):  # MPB bands are 1-indexed
        efield = ms.get_efield(b, bloch_phase=False)
        hfield = ms.get_hfield(b, bloch_phase=False)
        # Convert from MPB array to numpy: shape (nx, ny, 1, 3) -> (nx, ny, 3)
        e_arr = np.squeeze(np.array(efield))
        h_arr = np.squeeze(np.array(hfield))
        result[f'efield_b{b-1}'] = e_arr  # 0-indexed in output
        result[f'hfield_b{b-1}'] = h_arr
        print(f"  Band {b-1}: freq={freqs[b-1]:.6f}, "
              f"|E| shape={e_arr.shape}")

    return result


# ==========================================
# Cache I/O
# ==========================================

def get_cache_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_2d')
    os.makedirs(d, exist_ok=True)
    return d


def format_float_for_filename(value):
    return f"{value:.3f}".replace('.', 'p')


def cache_path(params, tag=''):
    """Generate cache file path based on params hash."""
    h = params_hash(params)
    h_spine_str = format_float_for_filename(params['h_spine'])
    if tag:
        name = f"fishbone2d_hspine{h_spine_str}_{tag}_{h}.npz"
    else:
        name = f"fishbone2d_hspine{h_spine_str}_{h}.npz"
    return os.path.join(get_cache_dir(), name)


def save_results(data, path):
    """Save simulation results to .npz."""
    # Convert params dict to JSON string for storage
    save_dict = {}
    for k, v in data.items():
        if k == 'params':
            save_dict['params_json'] = json.dumps(v)
        else:
            save_dict[k] = v
    np.savez(path, **save_dict)
    print(f"Results saved to {path}")


def load_results(path):
    """Load simulation results from .npz."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file not found: {path}")
    d = dict(np.load(path, allow_pickle=True))
    if 'params_json' in d:
        d['params'] = json.loads(str(d['params_json']))
        del d['params_json']
    # Convert scalar arrays back
    for k in ['sy', 'resolution', 'num_bands']:
        if k in d:
            d[k] = float(d[k])
    print(f"Results loaded from {path}")
    return d


# ==========================================
# Data export
# ==========================================

def get_output_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_2d')
    os.makedirs(d, exist_ok=True)
    return d


def export_band_csv(data, out_path=None):
    """Export full band data to a CSV file.

    Columns: k, band_0, band_1, ..., band_N, wavelength_band_0, ...
    Also saves a companion _info.json with parameters.
    """
    params = data['params']
    a_nm = params['a_nm']
    k_x = data['k_x']
    freqs = data['freqs_even']
    nb = freqs.shape[1]

    h_sp_str = format_float_for_filename(params['h_spine'])
    if out_path is None:
        out_path = os.path.join(get_output_dir(),
                                f'bands_hspine{h_sp_str}_nb{nb}.csv')

    header_cols = ['k']
    header_cols += [f'freq_band{b}' for b in range(nb)]
    header_cols += [f'wl_nm_band{b}' for b in range(nb)]

    rows = []
    for i in range(len(k_x)):
        row = [k_x[i]]
        row += [freqs[i, b] for b in range(nb)]
        row += [a_nm / freqs[i, b] if freqs[i, b] > 0 else np.nan
                for b in range(nb)]
        rows.append(row)

    arr = np.array(rows)
    np.savetxt(out_path, arr, delimiter=',',
               header=','.join(header_cols), comments='')

    # Save companion info
    info_path = out_path.replace('.csv', '_info.json')
    info = dict(params=params, resolution=int(data['resolution']),
                num_bands=nb, k_min=float(k_x[0]), k_max=float(k_x[-1]),
                n_kpoints=len(k_x))
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"Band data exported to: {out_path}")
    print(f"Parameter info saved to: {info_path}")
    return out_path


def export_field_data(field_data, out_path=None):
    """Save mode fields to a single .npz file.

    Contents:
        epsilon:  (nx, ny)
        freqs:    (num_bands,)
        efield_b0 .. efield_bN: (nx, ny, 3) complex
        hfield_b0 .. hfield_bN: (nx, ny, 3) complex
        k_at, sy, resolution, num_bands, params_json
    """
    params = field_data['params']
    h_sp_str = format_float_for_filename(params['h_spine'])
    k_str = format_float_for_filename(field_data['k_at'])
    nb = field_data['num_bands']

    if out_path is None:
        out_path = os.path.join(
            get_output_dir(),
            f'fields_hspine{h_sp_str}_k{k_str}_nb{nb}.npz')

    save_dict = {}
    for key, val in field_data.items():
        if key == 'params':
            save_dict['params_json'] = json.dumps(val)
        else:
            save_dict[key] = val
    np.savez_compressed(out_path, **save_dict)

    print(f"Field data exported to: {out_path}")
    print(f"  Contains: epsilon, freqs, efield_b0..b{nb-1}, hfield_b0..b{nb-1}")
    print(f"  Load with: data = dict(np.load('{out_path}', allow_pickle=True))")
    return out_path


# ==========================================
# Post-processing
# ==========================================

def compute_ng(f_band, k_x, a_nm):
    """Compute group index and wavelength from a single band."""
    df_dk = np.gradient(f_band, k_x)
    # Avoid division by zero
    df_dk[df_dk == 0] = np.nan
    ng = 1.0 / df_dk
    wavelengths = a_nm / f_band
    return ng, wavelengths


# ==========================================
# Plotting
# ==========================================

def plot_bands_and_ng(data, target_band=0, save_prefix='fishbone2d'):
    """Plot band diagram + group index, similar to paper Fig. 3."""
    params = data['params']
    a_nm = params['a_nm']
    n_SiO2 = params['n_SiO2']
    k_x = data['k_x']
    freqs_even = data['freqs_even']
    # freqs_odd = data['freqs_odd']
    eps = data['epsilon']
    sy = float(data['sy'])
    num_bands = int(data['num_bands'])

    # --- Compute ng for target band (from even modes) ---
    f_band = freqs_even[:, target_band]
    ng, wavelengths = compute_ng(f_band, k_x, a_nm)

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax_geo, ax_band, ax_ng = axes

    # (0) Geometry
    ax_geo.imshow(eps.T, extent=[-0.5, 0.5, -sy / 2, sy / 2],
                  cmap='gray', origin='lower')
    ax_geo.set_xlabel('x (a)')
    ax_geo.set_ylabel('y (a)')
    ax_geo.set_title(f'Geometry (h_spine={params["h_spine"]:.2f}a)')

    # (a) Band diagram — all bands
    light_line = k_x / n_SiO2
    ax_band.fill_between(k_x, light_line, 1.0, color='gray', alpha=0.3,
                         label='Light cone')

    for b in range(min(num_bands, freqs_even.shape[1])):
        label_e = f'Even band {b}' if b == target_band else None
        #label_o = f'Odd band {b}' if b == target_band else None
        ax_band.plot(k_x, freqs_even[:, b], 'r.-', markersize=2, linewidth=0.8,
                     alpha=0.7, label=label_e)
        # ax_band.plot(k_x, freqs_odd[:, b], 'b.--', markersize=2, linewidth=0.8,
        #              alpha=0.5, label=label_o)

    ax_band.set_xlabel(r'Wave vector ($2\pi/a$)')
    ax_band.set_ylabel(r'Normalized frequency ($a/\lambda$)')
    ax_band.set_title('(a) Band Diagram')
    ax_band.set_xlim([k_x.min(), k_x.max()])
    #ax_band.set_ylim([0.25, 0.28])
    ax_band.grid(True, alpha=0.3)
    ax_band.legend(fontsize=7, loc='upper left')

    # (b) Group index
    valid = (ng > 0) & (ng < 160) & np.isfinite(ng)
    ax_ng.plot(wavelengths[valid], ng[valid], 'ro-', markersize=3)
    ax_ng.set_xlabel('Wavelength (nm)')
    ax_ng.set_ylabel(r'Group index $n_g$')
    ax_ng.set_title('(b) Group Index vs Wavelength')
    ax_ng.set_xlim([1550, 1630])
    ax_ng.set_ylim([0, 160])
    ax_ng.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'{save_prefix}_hspine{params["h_spine"]:.2f}.png'
    plt.savefig(fname, dpi=200)
    print(f"Figure saved: {fname}")
    plt.show()


def plot_sweep_ng(all_data, target_band=0, save_prefix='fishbone2d_sweep'):
    """Plot group index for multiple h_spine values on one figure."""
    fig, (ax_band, ax_ng) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_data)))

    for i, data in enumerate(all_data):
        params = data['params']
        a_nm = params['a_nm']
        k_x = data['k_x']
        freqs_even = data['freqs_even']

        f_band = freqs_even[:, target_band]
        ng, wavelengths = compute_ng(f_band, k_x, a_nm)

        label = f'h_spine={params["h_spine"]:.2f}a'
        ax_band.plot(k_x, f_band, '.-', color=colors[i], markersize=2,
                     label=label)

        valid = (ng > 0) & (ng < 160) & np.isfinite(ng)
        ax_ng.plot(wavelengths[valid], ng[valid], 'o-', color=colors[i],
                   markersize=3, label=label)

    # Light cone on band plot
    n_SiO2 = all_data[0]['params']['n_SiO2']
    k_x = all_data[0]['k_x']
    light_line = k_x / n_SiO2
    ax_band.fill_between(k_x, light_line, 1.0, color='gray', alpha=0.2,
                         label='Light cone')

    ax_band.set_xlabel(r'Wave vector ($2\pi/a$)')
    ax_band.set_ylabel(r'Normalized frequency ($a/\lambda$)')
    ax_band.set_title('(a) Band Diagram — h_spine sweep')
    ax_band.set_xlim([k_x.min(), k_x.max()])
    ax_band.set_ylim([0.25, 0.28])
    ax_band.grid(True, alpha=0.3)
    ax_band.legend(fontsize=7)

    ax_ng.set_xlabel('Wavelength (nm)')
    ax_ng.set_ylabel(r'Group index $n_g$')
    ax_ng.set_title('(b) Group Index — h_spine sweep')
    ax_ng.set_xlim([1550, 1630])
    ax_ng.set_ylim([0, 160])
    ax_ng.grid(True, alpha=0.3)
    ax_ng.legend(fontsize=7)

    plt.tight_layout()
    fname = f'{save_prefix}.png'
    plt.savefig(fname, dpi=200)
    print(f"Sweep figure is saved: {fname}")
    plt.show()


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='2D Fishbone Grating MPB Simulation')
    parser.add_argument('--run', action='store_true',
                        help='Run MPB simulation (otherwise plot from cache)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only plot from cached results')
    parser.add_argument('--sweep-spine', action='store_true',
                        help='Sweep h_spine values')
    parser.add_argument('--h-spine', type=float, default=None,
                        help='Override h_spine value (units of a)')
    parser.add_argument('--target-band', type=int, default=TARGET_BAND,
                        help=f'Band index to analyze (default: {TARGET_BAND})')
    parser.add_argument('--resolution', type=int, default=RESOLUTION,
                        help=f'MPB resolution (default: {RESOLUTION})')
    parser.add_argument('--k-interp', type=int, default=K_INTERP,
                        help=f'Number of interpolated k-points (default: {K_INTERP})')
    parser.add_argument('--num-bands', type=int, default=NUM_BANDS,
                        help=f'Number of bands to compute (default: {NUM_BANDS})')
    parser.add_argument('--export-bands', action='store_true',
                        help='Export full band data to CSV for a single h_spine')
    parser.add_argument('--export-fields', action='store_true',
                        help='Compute and export mode fields at a single k-point')
    parser.add_argument('--k-point', type=float, default=FIELD_K_POINT,
                        help=f'k-point for field extraction (default: {FIELD_K_POINT})')
    args = parser.parse_args()

    # Validate: target_band must be within num_bands
    if args.target_band >= args.num_bands:
        parser.error(f'--target-band={args.target_band} is out of range for '
                     f'--num-bands={args.num_bands} (valid: 0..{args.num_bands-1})')

    # h_spine values to sweep (0.36a to 0.46a in steps of 0.02a)
    if args.sweep_spine:
        spine_values = SPINE_SWEEP_VALUES
    else:
        spine_val = args.h_spine if args.h_spine is not None else 0.40
        spine_values = [spine_val]

    all_data = []

    for h_sp in spine_values:
        params = get_default_params()
        params['h_spine'] = h_sp

        cpath = cache_path(params, tag=f'res{args.resolution}_nb{args.num_bands}')

        # Auto-run if: --run flag set, OR no cache exists and --plot-only not set
        need_run = (args.run and not args.plot_only) or \
                   (not args.plot_only and not os.path.exists(cpath))

        if not need_run and not args.plot_only:
            # Cache exists — verify it has enough bands
            try:
                cached = load_results(cpath)
                cached_nb = int(cached.get('num_bands', 0))
                if cached_nb < args.num_bands:
                    print(f"Cache for h_spine={h_sp:.2f} has only {cached_nb} bands, "
                          f"need {args.num_bands} — re-running.")
                    need_run = True
            except FileNotFoundError:
                need_run = True

        if need_run:
            if not args.run:
                print(f"No cache found for h_spine={h_sp:.2f} — running simulation automatically.")
            data = run_mpb_2d(params, k_interp=args.k_interp,
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

    # --- Export full band data to CSV ---
    if args.export_bands:
        for data in all_data:
            export_band_csv(data)
        return

    # --- Export mode fields at single k-point ---
    if args.export_fields:
        for h_sp in spine_values:
            params = get_default_params()
            params['h_spine'] = h_sp
            field_data = run_mpb_fields_2d(
                params, k_at=args.k_point,
                num_bands=args.num_bands, resolution=args.resolution)
            export_field_data(field_data)
        return

    # Plot
    if args.sweep_spine and len(all_data) > 1:
        plot_sweep_ng(all_data, target_band=args.target_band)
    else:
        for data in all_data:
            plot_bands_and_ng(data, target_band=args.target_band)


if __name__ == '__main__':
    main()
