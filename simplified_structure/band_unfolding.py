"""
band_unfolding.py — Post-processing to resolve band crossings in MPB results.

MPB returns eigenvalues sorted by frequency at each k-point. When physical
bands cross, band indices swap, creating artificial discontinuities.
This module tracks physical band identity across k-points using:

  Tier 1: First-order (slope-based) prediction + Hungarian assignment
          Works on cached frequency data without re-running MPB.

  Tier 2: D-field overlap (requires fields saved during MPB run).
          Definitive tracking based on mode identity. [Future]

Usage:
    python band_unfolding.py --analyze-3d --save-plots --no-show
    python band_unfolding.py --analyze-3d           # interactive plots
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Constants (matching simplified_gc_3d.py)
# ---------------------------------------------------------------------------

N_SIO2    = 1.44
TARGET_NG = 6.0
NG_LOW    = 5.0
NG_HIGH   = 7.0
WL_MIN    = 1400.0
WL_MAX    = 1700.0


# ===================================================================
# Tier 1: Anti-crossing detection + first-order prediction
# ===================================================================

def detect_anti_crossings(freqs, k_x, gap_threshold_factor=3.0):
    """Detect anti-crossings between adjacent bands.

    At an anti-crossing between bands i and i+1:
      - Both bands change slope direction simultaneously
      - Band i reaches a local max, Band i+1 reaches a local min
      - The gap between them is locally minimized

    Returns list of (band_lo, band_hi, k_index) for each detected
    anti-crossing.
    """
    Nk, Nb = freqs.shape
    anti_crossings = []

    for b in range(Nb - 1):
        f_lo = freqs[:, b]
        f_hi = freqs[:, b + 1]
        gap = f_hi - f_lo

        # Slopes
        df_lo = np.gradient(f_lo, k_x)
        df_hi = np.gradient(f_hi, k_x)

        for i in range(2, Nk - 2):
            # Conditions for anti-crossing at k-point i:
            # 1. Gap is at a local minimum
            is_gap_min = (gap[i] <= gap[i - 1]) and (gap[i] <= gap[i + 1])

            # 2. Lower band changes from positive to negative slope
            #    (local max in lower band)
            lo_slope_change = (df_lo[i - 1] > 0 and df_lo[i + 1] < 0)

            # 3. Upper band changes from negative to positive slope
            #    (local min in upper band)
            hi_slope_change = (df_hi[i - 1] < 0 and df_hi[i + 1] > 0)

            # 4. Gap is relatively small compared to neighboring gaps
            avg_gap = np.mean(gap)
            is_small_gap = gap[i] < avg_gap / gap_threshold_factor

            if is_gap_min and lo_slope_change and hi_slope_change:
                anti_crossings.append((b, b + 1, i))

    return anti_crossings


def reconnect_anti_crossings(freqs, k_x, anti_crossings):
    """Reconnect bands through anti-crossing points.

    At each anti-crossing between bands (b_lo, b_hi) at k-index k_ac:
    swap the band assignments for k < k_ac (or k > k_ac, depending on
    which gives smoother continuation).

    We swap the k < k_ac portion because we anchor from the zone edge (k_max).
    """
    Nk, Nb = freqs.shape
    reconnected = freqs.copy()

    # Sort anti-crossings by k-index (process from high-k to low-k
    # so swaps don't interfere)
    acs = sorted(anti_crossings, key=lambda x: x[2], reverse=True)

    for b_lo, b_hi, k_ac in acs:
        # Swap bands b_lo and b_hi for all k-points with index <= k_ac
        temp = reconnected[:k_ac + 1, b_lo].copy()
        reconnected[:k_ac + 1, b_lo] = reconnected[:k_ac + 1, b_hi]
        reconnected[:k_ac + 1, b_hi] = temp

    return reconnected


def unfold_bands(freqs, k_x):
    """Track physical bands: first-order prediction + anti-crossing correction.

    Two-pass algorithm:
      Pass 1: Hungarian assignment with first-order (slope) prediction
              to resolve simple band crossings (where indices swap).
      Pass 2: Detect and reconnect anti-crossings (where bands repel
              without swapping — the Hungarian method can't see these).

    Parameters
    ----------
    freqs : ndarray, shape (Nk, Nb)
        Raw MPB frequencies sorted by eigenvalue at each k-point.
    k_x : ndarray, shape (Nk,)
        Wave-vector values (monotonically increasing).

    Returns
    -------
    unfolded : ndarray, shape (Nk, Nb)
        Frequencies with columns representing physically continuous bands.
    anti_crossings : list of (b_lo, b_hi, k_index)
        Detected anti-crossing points.
    """
    Nk, Nb = freqs.shape

    # --- Pass 1: Hungarian assignment ---
    stage1 = freqs.copy()
    for i in range(Nk - 2, -1, -1):
        if i == Nk - 2:
            predicted = stage1[i + 1, :]
        else:
            predicted = 2.0 * stage1[i + 1, :] - stage1[i + 2, :]

        cost = np.abs(freqs[i, :, np.newaxis] - predicted[np.newaxis, :])
        row_ind, col_ind = linear_sum_assignment(cost)
        stage1[i, col_ind] = freqs[i, row_ind]

    # --- Pass 2: Anti-crossing detection and reconnection ---
    anti_crossings = detect_anti_crossings(stage1, k_x)
    if anti_crossings:
        unfolded = reconnect_anti_crossings(stage1, k_x, anti_crossings)
    else:
        unfolded = stage1

    return unfolded, anti_crossings


def trim_above_light_cone(freqs, k_x, n_clad):
    """Replace frequencies above the light cone with NaN.

    After anti-crossing reconnection, some band segments extend above the
    light cone (f > k / n_clad).  These are radiation modes, not guided,
    and should be excluded from all downstream analysis (ng, single-mode
    overlap checks).

    Parameters
    ----------
    freqs : ndarray, shape (Nk, Nb)
        Unfolded+reconnected frequencies.
    k_x : ndarray, shape (Nk,)
    n_clad : float
        Cladding refractive index (light line = k / n_clad).

    Returns
    -------
    trimmed : ndarray, shape (Nk, Nb)
        Copy with above-light-cone entries set to NaN.
    """
    trimmed = freqs.copy()
    f_light = k_x / n_clad
    for b in range(freqs.shape[1]):
        above = trimmed[:, b] >= f_light
        trimmed[above, b] = np.nan
    return trimmed


# ===================================================================
# Tier 2 stub: field-overlap tracking (requires saved D-fields)
# ===================================================================

def get_mode_fields(params, k_points_x, num_bands=6, resolution=16,
                    symmetry='zeven', field_type='d'):
    """Run MPB at given k-points and extract mode fields for all bands.

    This is a live MPB run (not from cache). It solves each k-point
    individually and extracts the D-field (or E-field) as a numpy array.

    Parameters
    ----------
    params : dict
        Geometry parameters (a_nm, h_spine, Wt, ht, Wb, hb, delta_s).
    k_points_x : array-like
        k-vector x-components to solve.
    num_bands : int
        Number of bands.
    resolution : int
        MPB grid resolution.
    symmetry : str
        'zeven' for TE-like, 'zodd' for TM-like, 'none' for no symmetry.
    field_type : str
        'd' for D-field, 'e' for E-field.

    Returns
    -------
    freqs : ndarray, shape (Nk, num_bands)
    fields : list of list of ndarray
        fields[k_idx][band_idx] = complex field array
    k_x : ndarray
    """
    import meep as mp
    import meep.mpb as mpb

    # Import geometry builder from simplified_gc_3d
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from simplified_gc_3d import build_geometry_3d

    lattice, geometry, sy, sz, t_slab = build_geometry_3d(params)

    k_pts = [mp.Vector3(kx) for kx in k_points_x]

    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=k_pts,
        resolution=resolution,
        num_bands=num_bands,
    )

    # Run to get all eigenvalues first
    if symmetry == 'zeven':
        ms.run_zeven()
    elif symmetry == 'zodd':
        ms.run_zodd()
    else:
        ms.run()

    freqs = np.array(ms.all_freqs)
    k_x = np.array([k.x for k in k_pts])

    # Now extract fields at each k-point
    # MPB keeps the last solved k-point's fields in memory.
    # To get fields at each k-point, we re-solve individually.
    fields = []
    get_field = ms.get_dfield if field_type == 'd' else ms.get_efield

    for ki, kpt in enumerate(k_pts):
        print(f"  Extracting fields at k={kpt.x:.4f} ({ki+1}/{len(k_pts)})")
        ms.solve_kpoint(kpt)
        band_fields = []
        for b in range(num_bands):
            get_field(b + 1)  # 1-based band index in MPB
            f_arr = np.array(ms.get_curfield_as_array()).copy()
            band_fields.append(f_arr)
        fields.append(band_fields)

    return freqs, fields, k_x


def plot_mode_fields(params, k_x_value, num_bands=6, resolution=16,
                     save_path=None, show=True):
    """Plot mode field profiles (|Dy|^2 in z=0 plane) at a given k-point.

    Generates a multi-panel figure showing each band's mode profile,
    useful for visually identifying mode character and anti-crossings.

    Parameters
    ----------
    params : dict
        Geometry parameters.
    k_x_value : float
        Single k-point to solve.
    num_bands : int
    resolution : int
    save_path : str or None
    show : bool
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    freqs, fields, k_x = get_mode_fields(
        params, [k_x_value], num_bands=num_bands, resolution=resolution)

    # fields[0] is the list of band fields at this k-point
    band_fields = fields[0]
    f_vals = freqs[0]

    ncols = min(num_bands, 4)
    nrows = (num_bands + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.atleast_2d(axes)

    for b in range(num_bands):
        ax = axes[b // ncols, b % ncols]
        f_arr = band_fields[b]  # shape depends on MPB: (Nx, Ny, Nz, 3) complex

        # Take z=0 slice, compute |D|^2
        if f_arr.ndim == 4:
            # 3D field: (Nx, Ny, Nz, 3)
            nz = f_arr.shape[2]
            z_mid = nz // 2
            d_slice = f_arr[:, :, z_mid, :]  # (Nx, Ny, 3)
        elif f_arr.ndim == 3:
            # 2D field: (Nx, Ny, 3)
            d_slice = f_arr
        else:
            ax.set_title(f'Band {b}: unexpected shape')
            continue

        intensity = np.sum(np.abs(d_slice)**2, axis=-1).T  # (Ny, Nx)

        im = ax.imshow(intensity, origin='lower', cmap='hot',
                       aspect='auto')
        ax.set_title(f'Band {b}, f={f_vals[b]:.5f}', fontsize=9)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Hide unused axes
    for b in range(num_bands, nrows * ncols):
        axes[b // ncols, b % ncols].axis('off')

    a = params['a_nm']
    fig.suptitle(
        f"|D|² at k={k_x_value:.4f}  a={a:.0f}nm  "
        f"spine={2*params['h_spine']*a:.0f}nm",
        fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def unfold_bands_field_overlap(params, k_points_x, num_bands=6,
                               resolution=16, symmetry='zeven'):
    """Tier 2: Track bands using D-field overlap between adjacent k-points.

    This is the definitive band-tracking method. At each pair of adjacent
    k-points, it computes the overlap matrix of mode D-fields and uses
    the Hungarian algorithm to find the optimal assignment.

    Advantages over Tier 1:
      - Detects anti-crossings even when bands don't change slope
      - Works regardless of band spacing or crossing angle
      - Physically grounded: tracks mode identity, not just frequency

    Parameters
    ----------
    params : dict
        Geometry parameters for build_geometry_3d.
    k_points_x : array-like
        k-vector x-components.
    num_bands, resolution, symmetry : see get_mode_fields.

    Returns
    -------
    freqs_raw : ndarray, shape (Nk, Nb)
        Original MPB frequencies.
    freqs_unfolded : ndarray, shape (Nk, Nb)
        Frequencies with columns tracking physical band identity.
    k_x : ndarray
    swap_log : list of (k_index, [(raw_band, physical_band), ...])
        Log of band reassignments at each k-point.
    """
    print("Tier 2: Running MPB and extracting mode fields...")
    freqs_raw, fields_list, k_x = get_mode_fields(
        params, k_points_x, num_bands=num_bands,
        resolution=resolution, symmetry=symmetry)

    Nk, Nb = freqs_raw.shape
    unfolded = freqs_raw.copy()
    swap_log = []

    # Track from k_max (zone edge) backwards to k_min
    # At k_max, physical band order = frequency order (identity)
    # fields_list[i][b] = field array for k-point i, band b

    # Make a mutable copy of fields (we'll permute as we track)
    fields = [[fields_list[i][b].copy() for b in range(Nb)] for i in range(Nk)]

    print("Computing field overlaps...")
    for i in range(Nk - 2, -1, -1):
        # Overlap matrix: O[p,q] = |<D_{i,p} | D_{i+1,q}>|
        # p = raw band index at k_i
        # q = physical band index (already assigned at k_{i+1})
        overlap = np.zeros((Nb, Nb))
        for p in range(Nb):
            for q in range(Nb):
                d_raw = fields_list[i][p]    # raw band p at k_i
                d_phys = fields[i + 1][q]    # physical band q at k_{i+1}
                inner = np.sum(np.conj(d_raw) * d_phys)
                overlap[p, q] = np.abs(inner)

        # Normalize: each row sum to 1
        row_sums = overlap.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        overlap_norm = overlap / row_sums

        cost = 1.0 - overlap_norm
        row_ind, col_ind = linear_sum_assignment(cost)

        # Apply permutation
        swaps = []
        for r, c in zip(row_ind, col_ind):
            if r != c:
                swaps.append((r, c))
        if swaps:
            swap_log.append((i, swaps))

        # Reorder frequencies and fields at k_i
        new_freqs = np.zeros(Nb)
        new_fields = [None] * Nb
        for r, c in zip(row_ind, col_ind):
            new_freqs[c] = freqs_raw[i, r]
            new_fields[c] = fields_list[i][r].copy()

        unfolded[i, :] = new_freqs
        fields[i] = new_fields

    print(f"Field-overlap tracking complete. "
          f"Swaps at {len(swap_log)}/{Nk} k-points.")

    return freqs_raw, unfolded, k_x, swap_log


# ===================================================================
# Analysis helpers
# ===================================================================

def compute_ng(f_band, k_x, a_nm):
    """Group index and wavelength from band dispersion.

    Handles NaN entries (trimmed above light cone): computes gradient only
    on contiguous finite segments, leaving NaN elsewhere.
    """
    ng = np.full_like(f_band, np.nan)
    wl = np.full_like(f_band, np.nan)

    valid = np.isfinite(f_band) & (f_band > 1e-6)
    wl[valid] = a_nm / f_band[valid]

    # Find contiguous runs of valid points
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return ng, wl

    # Split into contiguous segments
    splits = np.where(np.diff(idx) > 1)[0] + 1
    segments = np.split(idx, splits)

    for seg in segments:
        if len(seg) < 2:
            continue
        f_seg = f_band[seg]
        k_seg = k_x[seg]
        df_dk = np.gradient(f_seg, k_seg)
        df_dk = np.where(df_dk == 0.0, np.nan, df_dk)
        ng[seg] = 1.0 / df_dk

    return ng, wl


def find_flat_ng_region(ng, wavelengths, delta_ng=1.0, min_points=3,
                        ng_min_threshold=2.0):
    """Find the widest wavelength region where ng is approximately flat.

    For a U-shaped ng curve (band-edge slow light), the useful slow-light
    bandwidth is the flat bottom of the U, NOT where ng crosses a fixed
    threshold on the steep sides.

    Algorithm (sliding window):
      1. Sort by wavelength, keep only finite positive ng points.
      2. For each starting index i, expand window [i, j] as long as
         max(ng) - min(ng) <= delta_ng within the window.
      3. Track the widest such window.

    Parameters
    ----------
    ng : array-like
        Group index values.
    wavelengths : array-like
        Corresponding wavelengths (nm).
    delta_ng : float
        Maximum allowed ng variation within the flat region (default 1.0).
    min_points : int
        Minimum number of data points for a valid region.
    ng_min_threshold : float
        Minimum ng value to be considered slow light (default 2.0).
        Regions where ng is below this are ignored (too close to bulk).

    Returns
    -------
    dict with keys:
        bw : float - bandwidth in nm (0 if no flat region found)
        ng_mean : float - mean ng in the flat region
        ng_min : float - minimum ng in the flat region
        ng_max : float - maximum ng in the flat region
        wl_center : float - center wavelength of the flat region
        wl_range : (float, float) - (wl_min, wl_max)
        idx_range : (int, int) - indices into the sorted arrays
    """
    ng = np.asarray(ng, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)

    # Sort by wavelength, keep only valid points
    valid = np.isfinite(ng) & np.isfinite(wavelengths) & (ng > ng_min_threshold)
    if np.sum(valid) < min_points:
        return dict(bw=0.0, ng_mean=np.nan, ng_min=np.nan, ng_max=np.nan,
                    wl_center=np.nan, wl_range=(np.nan, np.nan),
                    idx_range=(0, 0))

    order = np.argsort(wavelengths[valid])
    wl = wavelengths[valid][order]
    g = ng[valid][order]
    n = len(wl)

    best_i, best_j = 0, 0
    best_bw = 0.0

    for i in range(n):
        g_min = g[i]
        g_max = g[i]
        for j in range(i + 1, n):
            g_min = min(g_min, g[j])
            g_max = max(g_max, g[j])
            if g_max - g_min > delta_ng:
                break
            bw = wl[j] - wl[i]
            if bw > best_bw and (j - i + 1) >= min_points:
                best_bw = bw
                best_i = i
                best_j = j

    if best_bw <= 0:
        return dict(bw=0.0, ng_mean=np.nan, ng_min=np.nan, ng_max=np.nan,
                    wl_center=np.nan, wl_range=(np.nan, np.nan),
                    idx_range=(0, 0))

    region_ng = g[best_i:best_j + 1]
    region_wl = wl[best_i:best_j + 1]

    return dict(
        bw=float(best_bw),
        ng_mean=float(np.mean(region_ng)),
        ng_min=float(np.min(region_ng)),
        ng_max=float(np.max(region_ng)),
        wl_center=float((region_wl[0] + region_wl[-1]) / 2),
        wl_range=(float(region_wl[0]), float(region_wl[-1])),
        idx_range=(int(best_i), int(best_j)),
    )


def guided_freq_range(f_band, k_x, n_clad):
    """(f_min, f_max) of the guided portion (below light cone) of a band.

    Handles NaN values (trimmed above light cone) gracefully.
    """
    f_light = k_x / n_clad
    fs = [f_band[i] for i in range(len(k_x))
          if np.isfinite(f_band[i]) and f_band[i] > 1e-6
          and f_band[i] < f_light[i]]
    if not fs:
        return None
    return (min(fs), max(fs))


def analyze_unfolded(freqs_unfolded, k_x, a_nm, n_clad=N_SIO2,
                     delta_ng=1.0):
    """Analyze all unfolded bands for slow light and single-mode operation.

    Slow-light detection uses the FLAT REGION method:
      For band-edge (U-shaped) ng curves, the useful slow-light bandwidth
      is the flat bottom/plateau of ng(λ), not where ng crosses a threshold
      on the steep sides. We find the widest wavelength region where
      max(ng) - min(ng) <= delta_ng.

    Single-mode criteria (ALL must be satisfied):
      1. The band is monotonic in the entire guided region (below light cone).
      2. No other band has guided modes at frequencies within the slow-light
         frequency window.

    Returns a list of dicts, one per band with slow light, sorted by bandwidth.
    """
    Nk, Nb = freqs_unfolded.shape
    results = []

    for b in range(Nb):
        f_b = freqs_unfolded[:, b]

        # Only consider finite (guided, below light cone) points
        valid = np.isfinite(f_b) & (f_b > 1e-6)
        n_guided = np.sum(valid)
        if n_guided < 3:
            continue

        # Monotonicity among valid (guided) points
        guided_f = f_b[valid]
        df = np.diff(guided_f)
        is_monotonic = bool(np.all(df >= -1e-6) or np.all(df <= 1e-6))

        # Group index — only on valid (guided) points
        f_for_ng = np.where(valid, f_b, np.nan)
        ng, wl = compute_ng(f_for_ng, k_x, a_nm)
        mask = valid & np.isfinite(wl) & (wl >= WL_MIN) & (wl <= WL_MAX) \
               & np.isfinite(ng) & (ng > 0)
        if np.sum(mask) < 3:
            continue

        # Find flat ng region (the real slow-light bandwidth)
        flat = find_flat_ng_region(ng[mask], wl[mask], delta_ng=delta_ng)
        if flat['bw'] <= 0:
            continue

        # Map the flat-region frequency range back to f_b for overlap check
        wl_lo, wl_hi = flat['wl_range']
        f_sl_hi = a_nm / wl_lo   # short wl → high freq
        f_sl_lo = a_nm / wl_hi   # long wl → low freq

        # Check neighbor band overlap in slow-light frequency window
        neighbors_overlap = []
        for nb in range(Nb):
            if nb == b:
                continue
            gr = guided_freq_range(freqs_unfolded[:, nb], k_x, n_clad)
            if gr and gr[0] <= f_sl_hi and gr[1] >= f_sl_lo:
                neighbors_overlap.append(nb)

        single_mode = is_monotonic and (len(neighbors_overlap) == 0)

        results.append(dict(
            band=b,
            bw=flat['bw'],
            ng_mean=flat['ng_mean'],
            ng_min=flat['ng_min'],
            ng_max=flat['ng_max'],
            wl_center=flat['wl_center'],
            wl_range=flat['wl_range'],
            f_sl_range=(f_sl_lo, f_sl_hi),
            is_monotonic=is_monotonic,
            n_guided=int(n_guided),
            neighbors_overlap=neighbors_overlap,
            single_mode=single_mode,
        ))

    # Sort by bandwidth descending
    results.sort(key=lambda r: r['bw'], reverse=True)
    return results


# ===================================================================
# Cache I/O
# ===================================================================

def load_3d_cache(cache_dir):
    """Load all .npz files from the 3D cache directory."""
    import glob
    files = sorted(glob.glob(os.path.join(cache_dir, '*.npz')))
    all_data = []
    for f in files:
        d = dict(np.load(f, allow_pickle=True))
        if 'params_json' in d:
            d['params'] = json.loads(str(d['params_json']))
            del d['params_json']
        for k in ['sy', 'sz', 't_slab', 'resolution', 'num_bands']:
            if k in d:
                d[k] = float(d[k])
        d['_file'] = os.path.basename(f)
        all_data.append(d)
    return all_data


# ===================================================================
# Plotting
# ===================================================================

def plot_unfolded(data, freqs_unfolded, freqs_trimmed, analysis,
                  anti_crossings, save_path=None, show=True):
    """Three-panel figure comparing raw vs unfolded bands + ng analysis.

    freqs_unfolded: full unfolded data (for band diagram, all points shown)
    freqs_trimmed:  with above-light-cone set to NaN (for ng calculation)
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    freqs_raw = data['freqs_te']
    k_x = data['k_x']
    eps = data['epsilon']
    sy = float(data['sy'])
    a_nm = data['params']['a_nm']

    # Find the best single-mode slow-light band
    best = None
    for r in analysis:
        if r['single_mode'] and r['bw'] > 0:
            best = r
            break
    # If no single-mode, pick best by BW
    if best is None and analysis:
        best = analysis[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel 0: Geometry z=0 slice ---
    ax0 = axes[0]
    nz = eps.shape[2]
    z_mid = nz // 2
    eps_slice = eps[:, :, z_mid].T
    ax0.imshow(eps_slice, origin='lower', cmap='RdBu_r',
               extent=[-0.5, 0.5, -sy / 2, sy / 2], aspect='auto')
    ax0.set_xlabel('x (a)')
    ax0.set_ylabel('y (a)')
    ax0.set_title('Geometry z=0 slice')

    # --- Panel 1: Band diagram (raw dashed + unfolded solid) ---
    ax1 = axes[1]
    f_lo, f_hi = 0.20, 0.35
    Nb = freqs_raw.shape[1]

    # Light cone
    k_lc = np.linspace(k_x[0], k_x[-1], 100)
    f_lc = k_lc / N_SIO2
    ax1.fill_between(k_lc, np.maximum(f_lc, f_lo), f_hi,
                     alpha=0.10, color='gray', label='Light cone')
    ax1.plot(k_lc, f_lc, color='gray', lw=0.8)

    # Raw bands (thin dashed gray)
    for b in range(Nb):
        f_b = freqs_raw[:, b]
        if np.any((f_b >= f_lo - 0.01) & (f_b <= f_hi + 0.01)):
            ax1.plot(k_x, f_b, '--', color='#bbbbbb', lw=0.6, zorder=1)

    # Unfolded bands (solid, distinct colors)
    colors = plt.cm.tab10(np.arange(Nb))
    for b in range(Nb):
        f_b = freqs_unfolded[:, b]
        if not np.any((f_b >= f_lo - 0.01) & (f_b <= f_hi + 0.01)):
            continue
        lw = 2.5 if (best is not None and b == best['band']) else 1.2
        zorder = 3 if (best is not None and b == best['band']) else 2
        label = f'Band {b}'
        if best is not None and b == best['band']:
            label += ' (slow light)'
        ax1.plot(k_x, f_b, '-', color=colors[b], lw=lw, zorder=zorder,
                 label=label)

    # Mark anti-crossing points
    if anti_crossings:
        for b_lo, b_hi, ki in anti_crossings:
            ax1.axvline(k_x[ki], color='orange', ls='--', lw=1.5, alpha=0.7)
            ax1.annotate(f'AC: {b_lo}↔{b_hi}',
                         xy=(k_x[ki], f_lo + 0.005), fontsize=6,
                         color='orange', ha='center')

    # 1550nm line
    f_1550 = a_nm / 1550.0
    if f_lo < f_1550 < f_hi:
        ax1.axhline(f_1550, color='blue', ls=':', lw=0.8, label='1550nm')

    ax1.set_xlabel('Wave vector (2π/a)')
    ax1.set_ylabel('Frequency (a/λ)')
    ax1.set_title('Band diagram (dashed=raw, solid=unfolded)')
    ax1.set_xlim(k_x[0], k_x[-1])
    ax1.set_ylim(f_lo, f_hi)
    ax1.legend(fontsize=6, loc='upper left')

    # --- Panel 2: ng vs wavelength (using trimmed/guided-only data) ---
    ax2 = axes[2]
    if best is not None:
        b = best['band']
        f_b = freqs_trimmed[:, b]
        ng, wl = compute_ng(f_b, k_x, a_nm)
        mask = np.isfinite(wl) & (wl >= WL_MIN) & (wl <= WL_MAX) & np.isfinite(ng) & (ng > 0) & (ng < 30)
        if np.any(mask):
            ax2.plot(wl[mask], ng[mask], 'r-o', ms=3, lw=1.5,
                     label=f'Band {b}')

    ax2.axhspan(NG_LOW, NG_HIGH, alpha=0.15, color='green',
                label=f'ng∈[{NG_LOW},{NG_HIGH}]')
    ax2.axhline(TARGET_NG, color='green', ls='--', lw=1.0)
    ax2.axvline(1550, color='blue', ls=':', lw=0.8, label='1550nm')

    if best is not None and best['bw'] > 0:
        wl_lo, wl_hi = best['wl_range']
        ax2.axvspan(wl_lo, wl_hi, alpha=0.15, color='yellow',
                    label=f'Flat region ({wl_hi-wl_lo:.0f}nm)')
        # Horizontal band showing the flat ng range
        ax2.axhspan(best['ng_min'], best['ng_max'], alpha=0.10, color='yellow')
        info_lines = [
            f"Band {best['band']} (flat region)",
            f"BW = {best['bw']:.1f} nm",
            f"ng = {best['ng_mean']:.2f} [{best['ng_min']:.1f}, {best['ng_max']:.1f}]",
            f"λ = [{wl_lo:.0f}, {wl_hi:.0f}] nm",
            f"Monotonic: {best['is_monotonic']}",
            f"Single-mode: {best['single_mode']}",
        ]
        if best['neighbors_overlap']:
            info_lines.append(f"Overlap: bands {best['neighbors_overlap']}")
        ax2.text(0.05, 0.95, '\n'.join(info_lines),
                 transform=ax2.transAxes, va='top', fontsize=8,
                 bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Group index ng')
    ax2.set_title('ng vs wavelength (unfolded)')
    ax2.set_xlim(WL_MIN, WL_MAX)
    ax2.set_ylim(0, 20)
    ax2.legend(fontsize=7)

    p = data['params']
    a = p['a_nm']
    fig.suptitle(
        f"3D Unfolded: a={a:.0f}nm  "
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


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Band unfolding post-processing for MPB results')
    parser.add_argument('--analyze-3d', action='store_true',
                        help='Analyze 3D cached results')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to output/')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots interactively')
    args = parser.parse_args()

    if not args.analyze_3d:
        parser.print_help()
        return

    # Load 3D cached data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, 'cache_3d')
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    all_data = load_3d_cache(cache_dir)
    if not all_data:
        print(f"No cached 3D results found in {cache_dir}")
        return

    print(f"Loaded {len(all_data)} cached 3D results\n")

    for idx, data in enumerate(all_data):
        p = data['params']
        a = p['a_nm']
        freqs_te = data['freqs_te']
        k_x = data['k_x']

        print(f"{'='*70}")
        print(f"Structure {idx}: a={a:.0f}nm  h_spine={p['h_spine']:.3f}"
              f"  Wt={p['Wt']:.3f} Wb={p['Wb']:.3f}")
        print(f"  File: {data.get('_file', '?')}")
        print(f"  freqs_te shape: {freqs_te.shape}")

        # --- Unfold bands ---
        freqs_uf, acs = unfold_bands(freqs_te, k_x)

        if acs:
            print(f"  Anti-crossings detected:")
            for b_lo, b_hi, ki in acs:
                print(f"    Bands {b_lo}↔{b_hi} at k={k_x[ki]:.4f} (index {ki})")
        else:
            print(f"  No anti-crossings detected")

        # Show how many k-points changed
        n_changed = 0
        for i in range(freqs_te.shape[0]):
            if not np.allclose(freqs_te[i], freqs_uf[i], atol=1e-8):
                n_changed += 1
        print(f"  Band assignments changed at {n_changed}/{freqs_te.shape[0]} k-points")

        # --- Trim above light cone ---
        freqs_trimmed = trim_above_light_cone(freqs_uf, k_x, N_SIO2)
        for b in range(freqs_trimmed.shape[1]):
            n_trimmed = np.sum(np.isnan(freqs_trimmed[:, b]) & ~np.isnan(freqs_uf[:, b]))
            if n_trimmed > 0:
                n_kept = np.sum(np.isfinite(freqs_trimmed[:, b]))
                print(f"  Band {b}: {n_trimmed} points above light cone trimmed, "
                      f"{n_kept} guided points kept")

        # --- Analyze (on trimmed data) ---
        analysis = analyze_unfolded(freqs_trimmed, k_x, a)

        print(f"\n  Slow-light bands (flat ng region, Δng≤1.0):")
        for r in analysis:
            sm_str = "SINGLE-MODE" if r['single_mode'] else "MULTIMODE"
            mono_str = "monotonic" if r['is_monotonic'] else "NON-MONOTONIC"
            nbr_str = (f"nbr overlap {r['neighbors_overlap']}"
                       if r['neighbors_overlap'] else "no nbr overlap")
            print(f"    Band {r['band']}: BW={r['bw']:.1f}nm  "
                  f"ng={r['ng_mean']:.2f} [{r['ng_min']:.2f},{r['ng_max']:.2f}]  "
                  f"λ=[{r['wl_range'][0]:.0f},{r['wl_range'][1]:.0f}]nm  "
                  f"{mono_str}  {nbr_str}  → {sm_str}")

        # --- Plot ---
        show = not args.no_show
        save_path = None
        if args.save_plots:
            save_path = os.path.join(
                output_dir,
                f"unfolded_3d_{idx}_a{a:.0f}nm.png")

        plot_unfolded(data, freqs_uf, freqs_trimmed, analysis, acs,
                      save_path=save_path, show=show)

        print()


if __name__ == '__main__':
    main()
