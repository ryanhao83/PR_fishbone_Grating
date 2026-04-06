"""
simplified_gc_3d.py — 3D MPB validation of simplified asymmetric fishbone
slow-light grating coupler.

3D structure:
  - 211nm poly-Si slab on SiO2 BOX (SOI substrate)
  - SiO2 top cladding (fully clad)
  - Fishbone pattern etched through the full poly-Si slab thickness
  - x: periodic direction (period = 1a)
  - y: transverse in-plane
  - z: vertical (out-of-plane), slab centered at z=0

Since top and bottom are both SiO2, z-mirror symmetry is preserved.
But y-mirror symmetry is broken (asymmetric ribs), so:
  - TE-like: ms.run_zodd()   (E mostly in xy-plane, odd in z)
  - TM-like: ms.run_zeven()  (E mostly along z, even in z)

Usage:
  python simplified_gc_3d.py --run                 # Run all 3 structures
  python simplified_gc_3d.py --run --id 0          # Run only structure #0
  python simplified_gc_3d.py --plot-only            # Plot from cache
  python simplified_gc_3d.py --run --resolution 32  # Higher resolution
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

# =====================================================================
# Material and layer parameters
# =====================================================================

N_POLY_SI = 3.48    # poly-Si refractive index at 1550nm
N_SIO2    = 1.44    # SiO2 cladding/substrate
T_SLAB_NM = 211.0   # poly-Si slab thickness (nm)
T_PARTIAL_NM = 71.0 # Partial etch background Si layer thickness (nm)

# Simulation defaults
RESOLUTION = 16     # 3D is expensive; 16 for survey, 32 for production
NUM_BANDS  = 6     # increased to 16 since we compute both TE and TM modes without parity
K_MIN      = 0.35   # wider k range for 3D (bands shift)
K_MAX      = 0.50
K_INTERP   = 30
PAD_Y      = 2.0    # supercell padding in y (units of a)
PAD_Z      = 1.0    # supercell padding in z (units of a)

# Slow-light analysis
TARGET_NG  = 6.5
NG_LOW     = 6.0
NG_HIGH    = 7.0
WL_MIN     = 1400.0
WL_MAX     = 1700.0


# =====================================================================
# The 3 structures to validate (from 2D optimization)
# =====================================================================

def get_structures():
    """Return the top 3 distinct single-mode structures from 2D optimization.

    Rank 1-3 share the same normalized geometry (different a only),
    so we pick rank 3 (a=389nm, centered at 1549nm) as representative.
    Rank 4 and 5 are distinct structures at a=380nm.
    """
    return [
        dict(
            label="Rank 1-3 (a=389nm, wl=1549nm, 2D BW=61nm)",
            a_nm=389.0, h_spine=0.550,
            Wt=0.484, ht=0.514, Wb=0.484, hb=0.514,
            delta_s=0.0,
        ),
        dict(
            label="Rank 4 (a=380nm, wl=1514nm, 2D BW=60nm)",
            a_nm=380.0, h_spine=0.550,
            Wt=0.569, ht=0.514, Wb=0.569, hb=0.514,
            delta_s=0.0,
        ),
        dict(
            label="Rank 5 (a=380nm, wl=1510nm, 2D BW=60nm)",
            a_nm=380.0, h_spine=0.550,
            Wt=0.468, ht=0.514, Wb=0.468, hb=0.514,
            delta_s=0.0,
        ),
    ]


# =====================================================================
# Geometry builder
# =====================================================================

def build_geometry_3d(params):
    """Build 3D MPB lattice + geometry for the simplified asymmetric grating.

    Layers (z direction):
      SiO2 substrate:  z < -t_slab/2
      Poly-Si slab:    -t_slab/2 < z < t_slab/2 (fishbone pattern)
      SiO2 top clad:   z > t_slab/2

    In-plane (xy):
      Central spine: Block(inf x 2*h_spine x t_slab) at z=0
      Top rib:       Block(Wt x ht x t_slab) at (delta_s/2, h_spine+ht/2, 0)
      Bottom rib:    Block(Wb x hb x t_slab) at (-delta_s/2, -(h_spine+hb/2), 0)

    Background is air; we explicitly place SiO2 substrate and top cladding.
    """
    import meep as mp

    a_nm = params['a_nm']
    t_slab = T_SLAB_NM / a_nm  # slab thickness in units of a

    sy = 2.0 * (params['h_spine'] + max(params['ht'], params['hb'])) + 2.0 * PAD_Y
    sz = t_slab + 2.0 * PAD_Z

    lattice = mp.Lattice(size=mp.Vector3(1, sy, sz))

    Si   = mp.Medium(index=N_POLY_SI)
    SiO2 = mp.Medium(index=N_SIO2)

    geometry = []

    # SiO2 substrate: fills lower half of supercell (z < -t_slab/2)
    geometry.append(
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, PAD_Z),
                 center=mp.Vector3(0, 0, -(t_slab / 2 + PAD_Z / 2)),
                 material=SiO2)
    )

    # SiO2 top cladding: fills upper half (z > t_slab/2)
    geometry.append(
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, PAD_Z),
                 center=mp.Vector3(0, 0, t_slab / 2 + PAD_Z / 2),
                 material=SiO2)
    )

    # Central continuous spine waveguide (full slab thickness)
    geometry.append(
        mp.Block(size=mp.Vector3(mp.inf, 2.0 * params['h_spine'], t_slab),
                 center=mp.Vector3(0, 0, 0),
                 material=Si)
    )

    # Top rib
    geometry.append(
        mp.Block(size=mp.Vector3(params['Wt'], params['ht'], t_slab),
                 center=mp.Vector3(params['delta_s'] / 2.0,
                                   params['h_spine'] + params['ht'] / 2.0, 0),
                 material=Si)
    )

    # Bottom rib
    geometry.append(
        mp.Block(size=mp.Vector3(params['Wb'], params['hb'], t_slab),
                 center=mp.Vector3(-params['delta_s'] / 2.0,
                                   -(params['h_spine'] + params['hb'] / 2.0), 0),
                 material=Si)
    )

    # Partial etch layer (continuous Si background at the bottom of the slab)
    t_partial = T_PARTIAL_NM / params['a_nm']
    geometry.append(
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, t_partial),
                 center=mp.Vector3(0, 0, -t_slab / 2.0 + t_partial / 2.0),
                 material=Si)
    )

    return lattice, geometry, sy, sz, t_slab


# =====================================================================
# Cache
# =====================================================================

def _geo_hash(params, resolution, num_bands):
    geo = {k: params[k] for k in ('a_nm','h_spine','Wt','ht','Wb','hb','delta_s')}
    geo['res'] = resolution
    geo['nb'] = num_bands
    geo['t_slab'] = T_SLAB_NM
    geo['t_partial'] = T_PARTIAL_NM
    return hashlib.md5(json.dumps(geo, sort_keys=True).encode()).hexdigest()[:8]


def get_cache_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_3d')
    os.makedirs(d, exist_ok=True)
    return d


def get_output_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(d, exist_ok=True)
    return d


def cache_path(params, resolution, num_bands):
    h = _geo_hash(params, resolution, num_bands)
    return os.path.join(get_cache_dir(), f'simplified3d_res{resolution}_nb{num_bands}_{h}.npz')


def save_results(data, path):
    params_json = json.dumps(data['params'], sort_keys=True)
    np.savez(path,
             freqs_te=data['freqs_te'],
             freqs_tm=data['freqs_tm'],
             k_x=data['k_x'],
             epsilon=data['epsilon'],
             sy=data['sy'], sz=data['sz'], t_slab=data['t_slab'],
             params_json=params_json,
             resolution=data['resolution'],
             num_bands=data['num_bands'])
    print(f"  Saved: {path}")


def load_results(path):
    d = dict(np.load(path, allow_pickle=True))
    d['params'] = json.loads(str(d['params_json']))
    for k in ('sy', 'sz', 't_slab', 'resolution', 'num_bands'):
        if k in d:
            d[k] = float(d[k])
    return d


# =====================================================================
# MPB runner
# =====================================================================

def run_mpb_3d(params, k_min=K_MIN, k_max=K_MAX, k_interp=K_INTERP,
               num_bands=NUM_BANDS, resolution=RESOLUTION):
    """Run 3D MPB for the simplified asymmetric grating.

    Partial etch breaks z-mirror symmetry, so we use ms.run() (no parity).
    After solving, each mode is classified as TE-like or TM-like by computing
    the TE fraction from the E-field:
        te_frac = 1 - sum(|Ez|^2) / sum(|Ex|^2 + |Ey|^2 + |Ez|^2)
    Bands with te_frac > 0.7 are TE-like; the rest are TM-like.
    """
    import meep as mp
    import meep.mpb as mpb

    lattice, geometry, sy, sz, t_slab = build_geometry_3d(params)
    k_points = mp.interpolate(k_interp, [mp.Vector3(k_min), mp.Vector3(k_max)])

    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=k_points,
        resolution=resolution,
        num_bands=num_bands,
    )

    a = params['a_nm']
    print(f"\n{'='*70}")
    print(f"3D MPB: a={a:.0f}nm  t_slab={T_SLAB_NM:.0f}nm ({t_slab:.4f}a)")
    print(f"  h_spine={params['h_spine']:.3f}a ({params['h_spine']*a:.1f}nm)")
    print(f"  Wt={params['Wt']:.3f}a ({params['Wt']*a:.1f}nm)  "
          f"ht={params['ht']:.3f}a ({params['ht']*a:.1f}nm)")
    print(f"  Wb={params['Wb']:.3f}a ({params['Wb']*a:.1f}nm)  "
          f"hb={params['hb']:.3f}a ({params['hb']*a:.1f}nm)")
    print(f"  Supercell: 1 x {sy:.2f} x {sz:.2f} (a units)")
    print(f"  Resolution={resolution}, bands={num_bands}, k={len(k_points)} pts")
    print(f"{'='*70}")

    print("  Running with no z-symmetry (partial etch breaks z-mirror)...")
    print("  Running k-point by k-point to extract mode polarization...")
    num_k = len(k_points)
    nb = num_bands

    all_freqs = np.zeros((num_k, nb))
    te_frac = np.zeros((num_k, nb))
    k_x = np.array([k.x for k in k_points])

    for ik, k in enumerate(k_points):
        ms.k_points = [k]
        ms.run()

        all_freqs[ik, :] = ms.all_freqs[0]

        for ib in range(nb):
            # Use MPB's built-in field energy decomposition (robust).
            # get_dfield loads D for band (ib+1), then compute_field_energy
            # returns [total, x_energy, y_energy, z_energy, xy, xz, yz].
            ms.get_dfield(ib + 1)
            energy = ms.compute_field_energy()
            # energy[0] = total, energy[1..3] = Ex, Ey, Ez fractions
            total = energy[0]
            if total > 0:
                ez_frac = energy[3] / total
                te_frac[ik, ib] = 1.0 - ez_frac
            else:
                te_frac[ik, ib] = np.nan

        if (ik + 1) % 10 == 0 or ik == num_k - 1:
            print(f"    k-point {ik+1}/{num_k} done")

    # Split into TE and TM arrays, padding with NaN where band doesn't belong
    freqs_te = np.where(te_frac > 0.7, all_freqs, np.nan)
    freqs_tm = np.where(te_frac <= 0.7, all_freqs, np.nan)

    n_te = np.sum(te_frac > 0.7)
    n_tm = np.sum(te_frac <= 0.7)
    print(f"  TE-like entries: {n_te}, TM-like entries: {n_tm} "
          f"(out of {num_k * nb} total)")

    eps = np.array(ms.get_epsilon())

    return dict(
        freqs_te=freqs_te, freqs_tm=freqs_tm,
        k_x=k_x, epsilon=eps,
        sy=sy, sz=sz, t_slab=t_slab,
        params=dict(params), resolution=resolution, num_bands=num_bands,
    )


# =====================================================================
# Analysis
# =====================================================================

def compute_ng(f_band, k_x, a_nm):
    """Group index and wavelength from band dispersion."""
    df_dk = np.gradient(f_band, k_x)
    df_dk = np.where(df_dk == 0.0, np.nan, df_dk)
    ng = 1.0 / df_dk
    with np.errstate(divide='ignore', invalid='ignore'):
        wl = np.where(f_band > 0, a_nm / f_band, np.nan)
    return ng, wl


def compute_slow_light_bandwidth(ng, wavelengths, ng_lo=NG_LOW, ng_hi=NG_HIGH):
    """Max contiguous bandwidth (nm) where ng_lo <= ng <= ng_hi."""
    order = np.argsort(wavelengths)
    wl = wavelengths[order]
    g  = ng[order]
    g  = np.where(np.isfinite(g), g, np.nan)
    in_range = (g >= ng_lo) & (g <= ng_hi)
    if not np.any(in_range):
        return 0.0

    max_bw = 0.0
    n = len(wl)
    i = 0
    while i < n:
        if not in_range[i]:
            i += 1
            continue
        wl_start = wl[i]
        j = i
        while j < n and in_range[j]:
            j += 1
        wl_end = wl[j - 1]
        max_bw = max(max_bw, abs(wl_end - wl_start))
        i = j
    return max_bw


def guided_freq_range(freqs_band, k_x, n_clad):
    """(f_min, f_max) of guided portion (below light cone) of a band."""
    gf = [freqs_band[i] for i in range(len(k_x))
          if np.isfinite(freqs_band[i]) and freqs_band[i] > 1e-6
          and freqs_band[i] < k_x[i] / n_clad]
    return (min(gf), max(gf)) if gf else None


def check_mode_hybridization(freqs_te, freqs_tm, k_x, a_nm,
                             target_wl=1550.0, f_tol=0.005):
    """Check whether any TM band is dangerously close to the target TE band.

    Parameters
    ----------
    freqs_te, freqs_tm : ndarray, shape (num_k, num_bands)
        Frequency arrays with NaN where a band doesn't belong to that
        polarization (as produced by run_mpb_3d).
    k_x : ndarray, shape (num_k,)
    a_nm : float — lattice constant in nm
    target_wl : float — target operating wavelength in nm
    f_tol : float — frequency proximity threshold (normalised units a/λ)

    Returns
    -------
    is_safe : bool
        True if no TM band comes within f_tol of the target TE band.
    detail : str
        Human-readable description of any detected overlap.
    """
    f_target = a_nm / target_wl
    num_k, nb = freqs_te.shape

    # Identify TE band closest to target_wl at each k-point
    # Use the band with the smallest median |f - f_target| across k
    band_dist = np.full(nb, np.inf)
    for b in range(nb):
        f_b = freqs_te[:, b]
        valid = np.isfinite(f_b)
        if np.sum(valid) < num_k // 2:
            continue  # band is mostly NaN (TM at most k-points)
        band_dist[b] = np.nanmedian(np.abs(f_b - f_target))

    te_band = int(np.argmin(band_dist))
    if not np.isfinite(band_dist[te_band]):
        return False, "No TE band found near target wavelength."

    f_te = freqs_te[:, te_band]

    # Check every TM band for frequency proximity at each k-point
    overlaps = []
    for b in range(freqs_tm.shape[1]):
        f_tm_b = freqs_tm[:, b]
        both_valid = np.isfinite(f_te) & np.isfinite(f_tm_b)
        if np.sum(both_valid) == 0:
            continue
        gap = np.abs(f_te[both_valid] - f_tm_b[both_valid])
        min_gap = float(np.min(gap))
        if min_gap < f_tol:
            k_worst = k_x[both_valid][np.argmin(gap)]
            wl_worst = a_nm / f_te[both_valid][np.argmin(gap)]
            overlaps.append(
                f"TM band {b}: min gap={min_gap:.5f} (< {f_tol}) "
                f"at k={k_worst:.3f}, ~{wl_worst:.0f}nm"
            )

    is_safe = len(overlaps) == 0
    if is_safe:
        detail = (f"TE band {te_band} (f~{f_target:.4f}) is clear of TM modes "
                  f"(tol={f_tol}).")
    else:
        detail = (f"TE band {te_band} has TE/TM anti-crossing risk:\n  "
                  + "\n  ".join(overlaps))

    return is_safe, detail


def analyze_3d(data):
    """Analyze 3D result: find slow-light band, check single-mode, compute BW.

    Searches TE bands for the one with the best slow-light FOM.
    Also checks if any TM band overlaps the slow-light frequency window.
    """
    freqs_te = data['freqs_te']
    freqs_tm = data['freqs_tm']
    k_x      = data['k_x']
    a        = data['params']['a_nm']
    nb_te    = freqs_te.shape[1]

    best_band = None
    best_bw   = 0.0
    best_ng   = np.nan
    best_wlc  = np.nan

    # Find best TE slow-light band
    for b in range(nb_te):
        f_b = freqs_te[:, b]
        valid = np.isfinite(f_b)
        if np.sum(valid) < 3 or np.any(f_b[valid] <= 1e-6):
            continue
        ng, wl = compute_ng(f_b, k_x, a)
        mask = np.isfinite(ng) & (ng > 0) & (ng < 200) & (wl > WL_MIN) & (wl < WL_MAX)
        if np.sum(mask) < 3:
            continue
        bw = compute_slow_light_bandwidth(ng[mask], wl[mask])
        sl = mask & (ng >= NG_LOW) & (ng <= NG_HIGH)
        if bw > best_bw and np.any(sl):
            best_bw = bw
            best_band = b
            best_ng = float(np.nanmean(ng[sl]))
            best_wlc = float((wl[sl].min() + wl[sl].max()) / 2)

    if best_band is None:
        return dict(band=None, bw=0, ng_mean=np.nan, wl_center=np.nan,
                    single_mode_te=False, no_tm_overlap=False, overlap_info="No slow-light band found")

    # Check single-mode within TE bands
    f_b = freqs_te[:, best_band]
    ng, wl = compute_ng(f_b, k_x, a)
    sl = np.isfinite(ng) & (ng >= NG_LOW) & (ng <= NG_HIGH) & (wl > WL_MIN) & (wl < WL_MAX)
    f_sl_lo, f_sl_hi = f_b[sl].min(), f_b[sl].max()

    te_overlap = []
    for b in range(nb_te):
        if b == best_band:
            continue
        gr = guided_freq_range(freqs_te[:, b], k_x, N_SIO2)
        if gr and gr[0] <= f_sl_hi and gr[1] >= f_sl_lo:
            te_overlap.append(f"TE-{b}")

    # Check TM band overlap via check_mode_hybridization
    is_safe, detail = check_mode_hybridization(
        freqs_te, freqs_tm, k_x, a, target_wl=best_wlc, f_tol=0.005)

    single_mode_te = len(te_overlap) == 0
    overlap_info   = f"TE overlap: {te_overlap}; TM hybridization: {detail}"

    return dict(
        band=best_band, bw=best_bw, ng_mean=best_ng, wl_center=best_wlc,
        single_mode_te=single_mode_te, no_tm_overlap=is_safe,
        overlap_info=overlap_info,
        f_sl_range=(float(f_sl_lo), float(f_sl_hi)),
    )


# =====================================================================
# Plotting
# =====================================================================

# --- Example: manual hybridization check with zoomed band diagram ---
# import matplotlib.pyplot as plt
#
# data = load_results('cache_3d/simplified3d_res32_nb16_XXXXXXXX.npz')
# freqs_te, freqs_tm = data['freqs_te'], data['freqs_tm']
# k_x, a_nm = data['k_x'], data['params']['a_nm']
#
# is_safe, detail = check_mode_hybridization(
#     freqs_te, freqs_tm, k_x, a_nm, target_wl=1550.0, f_tol=0.005)
# print(f"Safe: {is_safe}\n{detail}")
#
# # Find worst-case anti-crossing k-point from the detail string
# f_target = a_nm / 1550.0
# nb = freqs_te.shape[1]
# # Identify the operating TE band (closest median freq to f_target)
# band_dist = [np.nanmedian(np.abs(freqs_te[:, b] - f_target))
#              if np.sum(np.isfinite(freqs_te[:, b])) > len(k_x) // 2
#              else np.inf for b in range(nb)]
# te_band = int(np.argmin(band_dist))
# f_te = freqs_te[:, te_band]
#
# # Find the TM band with smallest gap to this TE band
# min_gap, worst_tm, worst_k = np.inf, None, None
# for b in range(freqs_tm.shape[1]):
#     f_tm_b = freqs_tm[:, b]
#     valid = np.isfinite(f_te) & np.isfinite(f_tm_b)
#     if not np.any(valid):
#         continue
#     gaps = np.abs(f_te[valid] - f_tm_b[valid])
#     idx = np.argmin(gaps)
#     if gaps[idx] < min_gap:
#         min_gap, worst_tm = gaps[idx], b
#         worst_k = k_x[valid][idx]
#
# # Plot zoomed band diagram around 1550nm
# fig, ax = plt.subplots(figsize=(6, 4))
# f_margin = 0.01  # narrow frequency window around target
# ax.plot(k_x, f_te, 'b-', lw=1.5, label=f'TE-{te_band}')
# if worst_tm is not None:
#     ax.plot(k_x, freqs_tm[:, worst_tm], 'g--', lw=1.5,
#             label=f'TM-{worst_tm} (gap={min_gap:.5f})')
#     ax.axvline(worst_k, color='red', ls=':', lw=0.8,
#                label=f'worst k={worst_k:.3f}')
# ax.axhline(f_target, color='gray', ls=':', lw=0.8, label='1550nm')
# ax.set_xlim(k_x[0], k_x[-1])
# ax.set_ylim(f_target - f_margin, f_target + f_margin)
# ax.set_xlabel('Wave vector (2pi/a)')
# ax.set_ylabel('Frequency (a/lambda)')
# ax.set_title('Zoomed TE/TM anti-crossing check near 1550nm')
# ax.legend(fontsize=7)
# plt.tight_layout()
# plt.show()

def plot_3d_result(data, analysis, save_path=None, show=True):
    """Four-panel figure: geometry | TE-only bands | TE+TM bands | ng vs wl."""
    import matplotlib.pyplot as plt

    freqs_te = data['freqs_te']
    freqs_tm = data['freqs_tm']
    k_x      = data['k_x']
    eps      = data['epsilon']
    sy       = float(data['sy'])
    a        = data['params']['a_nm']
    band     = analysis['band']

    f_lo, f_hi = 0.20, 0.35
    k_lc = np.linspace(k_x[0], k_x[-1], 100)
    f_lc = k_lc / N_SIO2
    f_1550 = a / 1550.0

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # --- Panel 0: Geometry z=0 slice ---
    ax0 = axes[0]
    nz = eps.shape[2]
    z_mid = nz // 2
    eps_slice = eps[:, :, z_mid].T
    ax0.imshow(eps_slice, origin='lower', cmap='RdBu_r',
               extent=[-0.5, 0.5, -sy/2, sy/2], aspect='auto')
    ax0.set_xlabel('x (a)')
    ax0.set_ylabel('y (a)')
    ax0.set_title('Geometry z=0 slice')

    # --- Panel 1: TE-only band diagram ---
    ax1 = axes[1]
    ax1.fill_between(k_lc, np.maximum(f_lc, f_lo), f_hi,
                     alpha=0.12, color='gray', label='Light cone')
    ax1.plot(k_lc, f_lc, color='gray', lw=0.8)
    for b in range(freqs_te.shape[1]):
        f_b = freqs_te[:, b]
        if not np.any(np.isfinite(f_b) & (f_b >= f_lo - 0.01) & (f_b <= f_hi + 0.01)):
            continue
        if band is not None and b == band:
            ax1.plot(k_x, f_b, 'r-', lw=2.5, zorder=3, label=f'TE-{b} (slow light)')
        else:
            ax1.plot(k_x, f_b, 'b-', lw=0.8, alpha=0.6)
    if f_lo < f_1550 < f_hi:
        ax1.axhline(f_1550, color='blue', ls=':', lw=0.8, label='1550nm')
    ax1.set_xlabel('Wave vector (2\u03c0/a)')
    ax1.set_ylabel('Frequency (a/\u03bb)')
    ax1.set_title('TE-like bands only')
    ax1.set_xlim(k_x[0], k_x[-1])
    ax1.set_ylim(f_lo, f_hi)
    ax1.legend(fontsize=7, loc='upper left')

    # --- Panel 2: TE + TM band diagram (hybridization view) ---
    ax2 = axes[2]
    ax2.fill_between(k_lc, np.maximum(f_lc, f_lo), f_hi,
                     alpha=0.12, color='gray', label='Light cone')
    ax2.plot(k_lc, f_lc, color='gray', lw=0.8)
    for b in range(freqs_te.shape[1]):
        f_b = freqs_te[:, b]
        if not np.any(np.isfinite(f_b) & (f_b >= f_lo - 0.01) & (f_b <= f_hi + 0.01)):
            continue
        if band is not None and b == band:
            ax2.plot(k_x, f_b, 'r-', lw=2.5, zorder=3, label=f'TE-{b} (slow light)')
        else:
            ax2.plot(k_x, f_b, 'b-', lw=0.8, alpha=0.6)
    for b in range(freqs_tm.shape[1]):
        f_b = freqs_tm[:, b]
        if not np.any(np.isfinite(f_b) & (f_b >= f_lo - 0.01) & (f_b <= f_hi + 0.01)):
            continue
        ax2.plot(k_x, f_b, 'g--', lw=1.0, alpha=0.7,
                 label='TM' if b == 0 else None)
    if f_lo < f_1550 < f_hi:
        ax2.axhline(f_1550, color='blue', ls=':', lw=0.8, label='1550nm')
    ax2.set_xlabel('Wave vector (2\u03c0/a)')
    ax2.set_ylabel('Frequency (a/\u03bb)')
    ax2.set_title('TE + TM (hybridization check)')
    ax2.set_xlim(k_x[0], k_x[-1])
    ax2.set_ylim(f_lo, f_hi)
    ax2.legend(fontsize=7, loc='upper left')

    # --- Panel 3: ng vs wavelength ---
    ax3 = axes[3]
    if band is not None:
        f_b = freqs_te[:, band]
        ng, wl = compute_ng(f_b, k_x, a)
        mask = (wl >= WL_MIN) & (wl <= WL_MAX) & np.isfinite(ng) & (ng > 0) & (ng < 30)
        if np.any(mask):
            ax3.plot(wl[mask], ng[mask], 'r-o', ms=3, lw=1.5, label=f'TE-{band}')

    ax3.axhspan(NG_LOW, NG_HIGH, alpha=0.15, color='green',
                label=f'ng\u2208[{NG_LOW},{NG_HIGH}]')
    ax3.axhline(TARGET_NG, color='green', ls='--', lw=1.0)
    ax3.axvline(1550, color='blue', ls=':', lw=0.8, label='1550nm')

    bw = analysis['bw']
    if bw > 0:
        ax3.text(0.05, 0.95,
                 f"BW = {bw:.1f} nm\nng = {analysis['ng_mean']:.2f}\n"
                 f"TE single-mode: {analysis['single_mode_te']}\n"
                 f"No TM overlap: {analysis['no_tm_overlap']}",
                 transform=ax3.transAxes, va='top', fontsize=8,
                 bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_ylabel('Group index ng')
    ax3.set_title('ng vs wavelength')
    ax3.set_xlim(WL_MIN, WL_MAX)
    ax3.set_ylim(0, 20)
    ax3.legend(fontsize=7)

    p = data['params']
    fig.suptitle(
        f"3D: a={a:.0f}nm  t={T_SLAB_NM:.0f}nm poly-Si/SiO2  "
        f"spine={2*p['h_spine']*a:.0f}nm  "
        f"Wt={p['Wt']*a:.0f}nm ht={p['ht']*a:.0f}nm  "
        f"Wb={p['Wb']*a:.0f}nm hb={p['hb']*a:.0f}nm",
        fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='3D MPB validation of simplified fishbone slow-light grating')
    parser.add_argument('--run', action='store_true',
                        help='Run 3D MPB simulations')
    parser.add_argument('--plot-only', action='store_true',
                        help='Plot from cached results')
    parser.add_argument('--id', type=int, default=None,
                        help='Run only structure #id (0-indexed)')
    parser.add_argument('--resolution', type=int, default=RESOLUTION)
    parser.add_argument('--num-bands', type=int, default=NUM_BANDS)
    parser.add_argument('--k-interp', type=int, default=K_INTERP)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--save-plots', action='store_true')
    args = parser.parse_args()

    structures = get_structures()
    if args.id is not None:
        structures = [structures[args.id]]

    show = not args.no_show
    out = get_output_dir()

    all_results = []

    for idx, s in enumerate(structures):
        label = s.pop('label')
        params = dict(s)

        cpath = cache_path(params, args.resolution, args.num_bands)

        if args.run or (not args.plot_only and not os.path.exists(cpath)):
            data = run_mpb_3d(params, k_min=K_MIN, k_max=K_MAX,
                              k_interp=args.k_interp,
                              num_bands=args.num_bands,
                              resolution=args.resolution)
            save_results(data, cpath)
        else:
            if not os.path.exists(cpath):
                print(f"No cache for {label}. Run with --run first.")
                s['label'] = label
                continue
            data = load_results(cpath)
            print(f"  Loaded cache for {label}")

        s['label'] = label

        # Analyze
        ana = analyze_3d(data)
        all_results.append(dict(data=data, analysis=ana, label=label, params=params))

        a = params['a_nm']
        print(f"\n--- {label} ---")
        print(f"  3D result: band={ana['band']}, BW={ana['bw']:.1f}nm, "
              f"ng={ana['ng_mean']:.2f}, wl_center={ana['wl_center']:.1f}nm")
        print(f"  TE single-mode: {ana['single_mode_te']}")
        print(f"  No TM overlap: {ana['no_tm_overlap']}")
        print(f"  {ana['overlap_info']}")

        if args.save_plots:
            sp = os.path.join(out, f'3d_struct{idx}_{_geo_hash(params, args.resolution, args.num_bands)}.png')
            plot_3d_result(data, ana, save_path=sp, show=show)
        elif show:
            plot_3d_result(data, ana, show=show)

    # Summary table
    if all_results:
        print(f"\n{'='*100}")
        print(f"3D VALIDATION SUMMARY")
        print(f"{'='*100}")
        print(f"{'#':>3} {'a':>5} {'BW_3D':>7} {'ng_3D':>7} {'wl_c':>7} "
              f"{'TE-SM':>6} {'TM-ok':>6} {'Label'}")
        print("-"*100)
        for i, r in enumerate(all_results):
            ana = r['analysis']
            a = r['params']['a_nm']
            print(f"{i:>3} {a:>5.0f} {ana['bw']:>7.1f} {ana['ng_mean']:>7.2f} "
                  f"{ana['wl_center']:>7.1f} "
                  f"{'Yes' if ana['single_mode_te'] else 'No':>6} "
                  f"{'Yes' if ana['no_tm_overlap'] else 'No':>6} "
                  f"{r['label']}")
        print(f"{'='*100}")


if __name__ == '__main__':
    main()
