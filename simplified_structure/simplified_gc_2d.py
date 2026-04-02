"""
simplified_gc_2d.py — 2D MPB simulation and optimization for the asymmetric
simplified fishbone slow-light grating coupler.

Structure (in MPB unit cell, effective-index 2D model):
  - Central spine: continuous Si ridge, half-width h_spine in y
  - Top rib: one periodic Si block per unit cell, width Wt (x), height ht (y),
             centered at x = +delta_s/2
  - Bottom rib: one periodic Si block per unit cell, width Wb (x), height hb (y),
                centered at x = -delta_s/2
  - Wt, ht, Wb, hb can all differ (fully asymmetric)
  - Background: SiO2 (n=1.44)

No y-mirror symmetry → ms.run() (not ms.run_yeven())

Fabrication constraint: all waveguide features and gaps >= 160nm.

Optimization goal: ng = 6, bandwidth > 5nm (wavelength range where 5 <= ng <= 7),
centered near 1550nm.

Usage:
  python simplified_gc_2d.py --sweep           # Phase 1 symmetric grid sweep
  python simplified_gc_2d.py --sweep-asym      # Phase 2 asymmetric sweep
  python simplified_gc_2d.py --optimize        # Phase 1 + Phase 2 full workflow
  python simplified_gc_2d.py --plot-only       # Plot from cache
  python simplified_gc_2d.py --params a_nm=420 Wt=0.45 ht=0.50 Wb=0.45 hb=0.50 delta_s=0.1
  python simplified_gc_2d.py --export-csv      # Export ranked results to CSV
"""

import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

# =====================================================================
# ALL CONTROL PARAMETERS — modify ONLY this section
# =====================================================================

# --- Default geometry (spatial params in units of a, except a_nm) ---
A_NM      = 420.0   # lattice constant (nm)
H_SPINE   = 0.25    # half-width of central spine in y (units of a)
WT        = 0.48    # top rib width in x (units of a)
HT        = 0.48    # top rib height in y (units of a)
WB        = 0.48    # bottom rib width in x (units of a)
HB        = 0.48    # bottom rib height in y (units of a)
DELTA_S   = 0.0     # lateral offset of top vs bottom rib in x (units of a)
N_EFF     = 2.85    # effective index of 220nm Si TE slab mode at 1550nm
N_SIO2    = 1.44    # SiO2 cladding/substrate index
PAD_Y     = 3.0     # supercell y-padding on each side (units of a)

# --- Simulation control ---
RESOLUTION = 32     # MPB spatial resolution (grid points per a)
NUM_BANDS  = 8      # number of bands (simpler structure needs fewer)
K_MIN      = 0.4    # k-sweep start (near BZ edge where slow light occurs)
K_MAX      = 0.5    # k-sweep end (BZ boundary)
K_INTERP   = 30     # number of k-points from mp.interpolate

# --- Slow light targeting ---
TARGET_NG  = 6.0    # target group index
NG_LOW     = 5.0    # lower ng bound for bandwidth calculation
NG_HIGH    = 7.0    # upper ng bound for bandwidth calculation
MIN_BW_NM  = 5.0    # minimum acceptable slow-light bandwidth (nm)
WL_CENTER  = 1550.0 # target center wavelength (nm)
WL_MIN     = 1400.0 # telecom analysis window lower bound (nm)
WL_MAX     = 1700.0 # telecom analysis window upper bound (nm)

# --- Fabrication ---
MIN_FEAT_NM = 160.0  # minimum feature size (nm)


# =====================================================================
# Fabrication constraint checking
# =====================================================================

def check_fab_constraints(params):
    """Check all fabrication constraints for the simplified structure.

    Returns a dict mapping constraint_name -> (value_nm, passed, message).
    """
    a = params['a_nm']
    checks = {}

    def chk(name, value_nm, label):
        passed = value_nm > MIN_FEAT_NM
        msg = f"{label} = {value_nm:.1f}nm {'OK' if passed else f'FAIL (<{MIN_FEAT_NM}nm)'}"
        checks[name] = (value_nm, passed, msg)

    chk('spine_width',    2 * params['h_spine'] * a,         'Spine width')
    chk('top_rib_width',  params['Wt'] * a,                  'Top rib width')
    chk('top_rib_height', params['ht'] * a,                  'Top rib height')
    chk('bot_rib_width',  params['Wb'] * a,                  'Bottom rib width')
    chk('bot_rib_height', params['hb'] * a,                  'Bottom rib height')
    chk('top_gap',        (1.0 - params['Wt']) * a,          'Top rib gap')
    chk('bot_gap',        (1.0 - params['Wb']) * a,          'Bottom rib gap')

    # Top rib must stay inside unit cell: |delta_s/2| + Wt/2 <= 0.5
    top_extent = abs(params['delta_s'] / 2.0) + params['Wt'] / 2.0
    top_in = top_extent <= 0.5
    checks['top_rib_in_cell'] = (
        top_extent * a,
        top_in,
        f"Top rib extent = {top_extent:.3f}a {'OK' if top_in else 'FAIL (outside unit cell)'}"
    )
    bot_extent = abs(params['delta_s'] / 2.0) + params['Wb'] / 2.0
    bot_in = bot_extent <= 0.5
    checks['bot_rib_in_cell'] = (
        bot_extent * a,
        bot_in,
        f"Bottom rib extent = {bot_extent:.3f}a {'OK' if bot_in else 'FAIL (outside unit cell)'}"
    )

    return checks


def validate_params(params):
    """Raise ValueError if any fabrication constraint is violated."""
    checks = check_fab_constraints(params)
    failures = [msg for _, (_, passed, msg) in checks.items() if not passed]
    if failures:
        raise ValueError("Fabrication constraints violated:\n  " + "\n  ".join(failures))


# =====================================================================
# Geometry
# =====================================================================

def get_default_params():
    """Return default geometry parameter dict."""
    return dict(
        a_nm=A_NM, h_spine=H_SPINE,
        Wt=WT, ht=HT, Wb=WB, hb=HB,
        delta_s=DELTA_S,
        n_eff=N_EFF, n_SiO2=N_SIO2, pad_y=PAD_Y,
    )


def build_geometry_simplified(params):
    """Build 2D MPB lattice and geometry for the asymmetric simplified grating.

    Coordinate system:
      x — periodic direction, period = 1a, unit cell x in [-0.5, 0.5]
      y — transverse direction
      (z not present, 2D effective-index model)

    Structure:
      Central spine: Block(size=(inf, 2*h_spine), center=(0,0))
      Top rib:       Block(size=(Wt, ht),  center=(+delta_s/2,  h_spine+ht/2))
      Bottom rib:    Block(size=(Wb, hb),  center=(-delta_s/2, -(h_spine+hb/2)))
      Background:    SiO2 set as default_material in ModeSolver.

    Returns (lattice, geometry, sy).
    """
    import meep as mp

    p = params
    sy = 2.0 * (p['h_spine'] + max(p['ht'], p['hb'])) + 2.0 * p['pad_y']
    lattice = mp.Lattice(size=mp.Vector3(1, sy))
    mat_si = mp.Medium(index=p['n_eff'])

    geometry = [
        # Central continuous spine
        mp.Block(size=mp.Vector3(mp.inf, 2.0 * p['h_spine']),
                 center=mp.Vector3(0, 0),
                 material=mat_si),
        # Top rib (shifted +delta_s/2 in x)
        mp.Block(size=mp.Vector3(p['Wt'], p['ht']),
                 center=mp.Vector3(p['delta_s'] / 2.0, p['h_spine'] + p['ht'] / 2.0),
                 material=mat_si),
        # Bottom rib (shifted -delta_s/2 in x)
        mp.Block(size=mp.Vector3(p['Wb'], p['hb']),
                 center=mp.Vector3(-p['delta_s'] / 2.0, -(p['h_spine'] + p['hb'] / 2.0)),
                 material=mat_si),
    ]

    return lattice, geometry, sy


# =====================================================================
# Cache infrastructure
# =====================================================================

# Geometry keys used for hashing (simulation params excluded)
_GEO_KEYS = ('a_nm', 'h_spine', 'Wt', 'ht', 'Wb', 'hb', 'delta_s',
             'n_eff', 'n_SiO2', 'pad_y')


def params_hash(params):
    """8-character MD5 hash of geometry params (geometry keys only)."""
    geo = {k: params[k] for k in _GEO_KEYS if k in params}
    s = json.dumps(geo, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def get_cache_dir():
    """Return path to simplified_structure/cache/, creating it if needed."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    os.makedirs(d, exist_ok=True)
    return d


def get_output_dir():
    """Return path to simplified_structure/output/, creating it if needed."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(d, exist_ok=True)
    return d


def cache_path(params, resolution, num_bands):
    """Generate cache file path: cache/simplified2d_res{R}_nb{N}_{hash8}.npz"""
    h = params_hash(params)
    fname = f"simplified2d_res{resolution}_nb{num_bands}_{h}.npz"
    return os.path.join(get_cache_dir(), fname)


def save_results(data, path):
    """Save MPB results dict to npz file."""
    # Serialize params to JSON string for storage
    params_json = json.dumps(data['params'], sort_keys=True)
    np.savez(
        path,
        freqs=data['freqs'],
        k_x=data['k_x'],
        epsilon=data['epsilon'],
        sy=data['sy'],
        params_json=params_json,
        resolution=data['resolution'],
        num_bands=data['num_bands'],
        k_min=data['k_min'],
        k_max=data['k_max'],
    )


def load_results(path):
    """Load MPB results dict from npz file."""
    d = dict(np.load(path, allow_pickle=True))
    d['params'] = json.loads(str(d['params_json']))
    d['sy']         = float(d['sy'])
    d['resolution'] = int(d['resolution'])
    d['num_bands']  = int(d['num_bands'])
    d['k_min']      = float(d['k_min'])
    d['k_max']      = float(d['k_max'])
    return d


# =====================================================================
# MPB simulation runner
# =====================================================================

def run_mpb_simplified(params,
                       k_min=K_MIN, k_max=K_MAX, k_interp=K_INTERP,
                       num_bands=NUM_BANDS, resolution=RESOLUTION):
    """Run 2D MPB simulation for the asymmetric simplified grating coupler.

    Uses ms.run() (no symmetry) because the structure is asymmetric in y.
    Validates fabrication constraints before running.

    Returns dict with keys: freqs, k_x, epsilon, sy, params, resolution,
                             num_bands, k_min, k_max.
    """
    import meep as mp
    import meep.mpb as mpb

    validate_params(params)

    lattice, geometry, sy = build_geometry_simplified(params)
    k_points = mp.interpolate(k_interp, [mp.Vector3(k_min), mp.Vector3(k_max)])

    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=k_points,
        resolution=resolution,
        num_bands=num_bands,
        default_material=mp.Medium(index=params['n_SiO2']),
    )

    print(f"  MPB: a={params['a_nm']:.0f}nm  h_spine={params['h_spine']:.3f}  "
          f"Wt={params['Wt']:.3f} ht={params['ht']:.3f}  "
          f"Wb={params['Wb']:.3f} hb={params['hb']:.3f}  "
          f"ds={params['delta_s']:.3f}  "
          f"res={resolution} nb={num_bands}")

    # Asymmetric structure: no y-mirror symmetry → must use ms.run()
    ms.run()

    freqs = np.array(ms.all_freqs)   # shape (nk, num_bands)
    k_x   = np.array([k.x for k in k_points])
    eps   = np.array(ms.get_epsilon())

    return dict(
        freqs=freqs,
        k_x=k_x,
        epsilon=eps,
        sy=sy,
        params=dict(params),
        resolution=resolution,
        num_bands=num_bands,
        k_min=k_min,
        k_max=k_max,
    )


def run_or_load(params,
                resolution=RESOLUTION, num_bands=NUM_BANDS,
                k_min=K_MIN, k_max=K_MAX, k_interp=K_INTERP,
                force_rerun=False):
    """Load from cache or run MPB if cache is missing/stale."""
    cpath = cache_path(params, resolution, num_bands)
    if not force_rerun and os.path.exists(cpath):
        try:
            data = load_results(cpath)
            if data['freqs'].shape[1] >= num_bands:
                return data
        except Exception as e:
            print(f"  Warning: cache load failed ({e}), re-running.")
    sim_kw = dict(k_min=k_min, k_max=k_max, k_interp=k_interp,
                  num_bands=num_bands, resolution=resolution)
    data = run_mpb_simplified(params, **sim_kw)
    save_results(data, cpath)
    return data


# =====================================================================
# Group index analysis
# =====================================================================

def compute_ng(f_band, k_x, a_nm):
    """Compute group index and wavelength (nm) for one band.

    ng = 1 / (df/dk)  in MPB normalized units (c=1, a=1).
    wavelengths_nm = a_nm / f_band.
    Points where df/dk == 0 become np.nan.
    """
    df_dk = np.gradient(f_band, k_x)
    df_dk = np.where(df_dk == 0.0, np.nan, df_dk)
    ng = 1.0 / df_dk
    with np.errstate(divide='ignore', invalid='ignore'):
        wl = np.where(f_band > 0, a_nm / f_band, np.nan)
    return ng, wl


def find_flat_ng_region(ng, wavelengths, delta_ng=1.0, min_points=3,
                        ng_min_threshold=2.0):
    """Find the widest wavelength region where ng is approximately flat.

    For band-edge (U-shaped) ng curves, the useful slow-light bandwidth
    is the flat bottom/plateau, NOT where ng crosses a threshold on the
    steep sides. We find the widest contiguous wavelength window where
    max(ng) - min(ng) <= delta_ng.

    Parameters
    ----------
    ng : array-like
        Group index values.
    wavelengths : array-like
        Corresponding wavelengths (nm).
    delta_ng : float
        Maximum allowed ng variation within the flat region.
    min_points : int
        Minimum number of data points for a valid region.
    ng_min_threshold : float
        Minimum ng value to be considered slow light (default 2.0).

    Returns
    -------
    dict: bw, ng_mean, ng_min, ng_max, wl_center, wl_range
    """
    ng = np.asarray(ng, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)

    valid = np.isfinite(ng) & np.isfinite(wavelengths) & (ng > ng_min_threshold)
    if np.sum(valid) < min_points:
        return dict(bw=0.0, ng_mean=np.nan, ng_min=np.nan, ng_max=np.nan,
                    wl_center=np.nan, wl_range=(np.nan, np.nan))

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
                    wl_center=np.nan, wl_range=(np.nan, np.nan))

    region_ng = g[best_i:best_j + 1]
    region_wl = wl[best_i:best_j + 1]

    return dict(
        bw=float(best_bw),
        ng_mean=float(np.mean(region_ng)),
        ng_min=float(np.min(region_ng)),
        ng_max=float(np.max(region_ng)),
        wl_center=float((region_wl[0] + region_wl[-1]) / 2),
        wl_range=(float(region_wl[0]), float(region_wl[-1])),
    )


def detect_slow_light_band(freqs, k_x, a_nm,
                            target_ng=TARGET_NG,
                            wl_min=WL_MIN, wl_max=WL_MAX):
    """Auto-detect the band index with the best flat ng region.

    Uses find_flat_ng_region to identify the flat bottom/plateau of ng(λ).
    Score = flat_bandwidth - 2.0 * |ng_mean - target_ng|.
    Returns band index (0-based). Falls back to 0 if no suitable band found.
    """
    num_bands = freqs.shape[1]
    best_band = 0
    best_score = -np.inf

    for b in range(num_bands):
        f_b = freqs[:, b]
        if np.any(f_b <= 1e-6) or not np.all(np.isfinite(f_b)):
            continue

        ng, wl = compute_ng(f_b, k_x, a_nm)
        mask = (wl >= wl_min) & (wl <= wl_max) & np.isfinite(ng) & (ng > 0)
        if np.sum(mask) < 3:
            continue

        flat = find_flat_ng_region(ng[mask], wl[mask])
        if flat['bw'] <= 0:
            continue

        score = flat['bw'] - 2.0 * abs(flat['ng_mean'] - target_ng)

        if score > best_score:
            best_score = score
            best_band = b

    return best_band


def compute_fom(data, target_ng=TARGET_NG, ng_lo=NG_LOW, ng_hi=NG_HIGH,
                band_override=None):
    """Compute figure of merit for one MPB result.

    FOM = flat-region bandwidth (nm) on the best slow-light band.
    The flat region is found by find_flat_ng_region (widest contiguous
    wavelength range where ng variation <= 1.0).

    Returns dict: fom, bandwidth_nm, band, ng_mean, ng_median, wl_center,
                  wl_range, meets_target, params.
    """
    freqs = data['freqs']
    k_x   = data['k_x']
    a_nm  = data['params']['a_nm']

    if band_override is not None:
        band = band_override
    else:
        band = detect_slow_light_band(freqs, k_x, a_nm, target_ng)

    f_b = freqs[:, band]
    ng, wl = compute_ng(f_b, k_x, a_nm)

    mask = (wl >= WL_MIN) & (wl <= WL_MAX) & np.isfinite(ng) & (ng > 0)
    ng_win = np.clip(ng[mask], 0, 200)
    wl_win = wl[mask]

    flat = find_flat_ng_region(ng_win, wl_win)
    bw = flat['bw']
    ng_mean = flat['ng_mean']
    ng_median = flat['ng_mean']  # same for flat region
    wl_center = flat['wl_center']
    wl_range = flat['wl_range']

    meets_target = (bw > MIN_BW_NM) and (ng_lo <= ng_mean <= ng_hi)

    return dict(
        fom=bw,
        bandwidth_nm=bw,
        band=band,
        ng_mean=ng_mean,
        ng_median=ng_median,
        ng_min=flat['ng_min'],
        ng_max=flat['ng_max'],
        wl_center=wl_center,
        wl_range=wl_range,
        meets_target=meets_target,
        params=dict(data['params']),
    )


# =====================================================================
# Parameter sweeps
# =====================================================================

def _fab_ok(params):
    """Return True if params satisfy all fabrication constraints."""
    try:
        validate_params(params)
        return True
    except ValueError:
        return False


def generate_phase1_sweep(n_a=5, n_spine=4, n_W=4, n_h=4):
    """Generate Phase 1 symmetric sweep grid (Wt=Wb, ht=hb, delta_s=0).

    Parameter ranges respect fabrication constraints with a 20nm safety margin.
    """
    a_values = np.linspace(380, 600, n_a)
    sweep = []
    base = get_default_params()
    margin = 20.0  # nm safety margin

    for a in a_values:
        # Constraint-aware bounds in units of a
        hs_lo = (80.0 + margin) / a          # spine half-width lower bound
        hs_hi = 0.55
        W_lo  = (160.0 + margin) / a         # rib width lower bound
        W_hi  = min((a - 160.0 - margin) / a, 0.60)
        h_lo  = (160.0 + margin) / a         # rib height lower bound
        h_hi  = 0.70

        if hs_lo >= hs_hi or W_lo >= W_hi or h_lo >= h_hi:
            continue

        for hs in np.linspace(hs_lo, hs_hi, n_spine):
            for W in np.linspace(W_lo, W_hi, n_W):
                for h in np.linspace(h_lo, h_hi, n_h):
                    p = dict(base)
                    p.update(a_nm=float(a), h_spine=float(hs),
                             Wt=float(W), ht=float(h),
                             Wb=float(W), hb=float(h),
                             delta_s=0.0)
                    if _fab_ok(p):
                        sweep.append(p)

    return sweep


def generate_phase2_sweep(base_params, n_ratio=5, n_delta=5):
    """Generate Phase 2 asymmetric sweep around a best symmetric result.

    Explores:
      rW = Wt/Wb in [0.7, 1.3]  (preserving average W)
      rH = ht/hb in [0.7, 1.3]  (preserving average h)
      delta_s in [0, 0.3]

    Returns list of param dicts; constraint violations are silently dropped.
    """
    W_avg = (base_params['Wt'] + base_params['Wb']) / 2.0
    h_avg = (base_params['ht'] + base_params['hb']) / 2.0

    rW_vals     = np.linspace(0.7, 1.3, n_ratio)
    rH_vals     = np.linspace(0.7, 1.3, n_ratio)
    delta_vals  = np.linspace(0.0, 0.3, n_delta)

    sweep = []
    for rW in rW_vals:
        # Wt = 2*W_avg*rW/(1+rW),  Wb = 2*W_avg/(1+rW)
        Wt = 2.0 * W_avg * rW / (1.0 + rW)
        Wb = 2.0 * W_avg / (1.0 + rW)
        for rH in rH_vals:
            ht = 2.0 * h_avg * rH / (1.0 + rH)
            hb = 2.0 * h_avg / (1.0 + rH)
            for ds in delta_vals:
                p = dict(base_params)
                p.update(Wt=float(Wt), Wb=float(Wb),
                         ht=float(ht), hb=float(hb),
                         delta_s=float(ds))
                if _fab_ok(p):
                    sweep.append(p)

    return sweep


def run_sweep(param_list, resolution=RESOLUTION, num_bands=NUM_BANDS,
              k_min=K_MIN, k_max=K_MAX, k_interp=K_INTERP, force_rerun=False):
    """Run MPB for each param set, using cache when available.

    Returns list of dicts: {data, fom, index}.
    """
    results = []
    N = len(param_list)
    for i, params in enumerate(param_list):
        a = params['a_nm']
        hs = params['h_spine']
        Wt = params['Wt']; ht = params['ht']
        Wb = params['Wb']; hb = params['hb']
        ds = params['delta_s']
        print(f"[{i+1}/{N}] a={a:.0f}nm hs={hs:.3f} "
              f"Wt={Wt:.3f} ht={ht:.3f} Wb={Wb:.3f} hb={hb:.3f} ds={ds:.3f}",
              end=' ... ', flush=True)

        try:
            data = run_or_load(params, resolution=resolution,
                               num_bands=num_bands, k_min=k_min, k_max=k_max,
                               k_interp=k_interp, force_rerun=force_rerun)
            fom = compute_fom(data)
            print(f"FOM={fom['fom']:.2f}nm ng={fom['ng_mean']:.1f} "
                  f"wl_ctr={fom['wl_center']:.0f}nm "
                  f"{'*** MEETS TARGET ***' if fom['meets_target'] else ''}")
        except Exception as e:
            print(f"ERROR: {e}")
            fom = dict(fom=0.0, bandwidth_nm=0.0, band=0,
                       ng_mean=float('nan'), ng_median=float('nan'),
                       wl_center=float('nan'), wl_range=(float('nan'), float('nan')),
                       meets_target=False, params=dict(params))
            data = None

        results.append(dict(data=data, fom=fom, index=i))

    return results


def rank_results(results):
    """Sort results by (FOM desc, |ng_mean-6| asc, |wl_center-1550| asc)."""
    def sort_key(r):
        f = r['fom']
        ng_dist = abs(f['ng_mean'] - TARGET_NG) if np.isfinite(f['ng_mean']) else 1e9
        wl_dist = abs(f['wl_center'] - WL_CENTER) if np.isfinite(f['wl_center']) else 1e9
        return (-f['fom'], ng_dist, wl_dist)

    ranked = sorted(results, key=sort_key)
    for rank, r in enumerate(ranked, 1):
        r['rank'] = rank
    return ranked


# =====================================================================
# Reporting
# =====================================================================

def print_top_results(ranked_results, n_top=10):
    """Print a formatted table of the top-n results."""
    print(f"\n{'='*110}")
    print(f"TOP {n_top} RESULTS (ranked by slow-light bandwidth at ng={TARGET_NG}±{(NG_HIGH-NG_LOW)/2:.0f})")
    print(f"{'='*110}")
    hdr = (f"{'Rank':>4} {'a(nm)':>6} {'hs(nm)':>7} {'Wt(nm)':>7} {'ht(nm)':>7} "
           f"{'Wb(nm)':>7} {'hb(nm)':>7} {'ds(nm)':>7} "
           f"{'Band':>5} {'BW(nm)':>7} {'ng_mean':>8} {'wl_ctr':>8} {'Fab':>5}")
    print(hdr)
    print('-' * 110)

    for r in ranked_results[:n_top]:
        f   = r['fom']
        p   = f['params']
        a   = p['a_nm']
        checks = check_fab_constraints(p)
        fab_ok = all(passed for _, (_, passed, _) in checks.items())

        print(f"{r['rank']:>4} {a:>6.0f} "
              f"{p['h_spine']*a:>7.1f} "
              f"{p['Wt']*a:>7.1f} {p['ht']*a:>7.1f} "
              f"{p['Wb']*a:>7.1f} {p['hb']*a:>7.1f} "
              f"{p['delta_s']*a:>7.1f} "
              f"{f['band']:>5} "
              f"{f['fom']:>7.2f} "
              f"{f['ng_mean']:>8.2f} "
              f"{f['wl_center']:>8.1f} "
              f"{'Yes' if fab_ok else 'No':>5}")

    print(f"{'='*110}")

    # Also print full dimensions of the best result
    if ranked_results:
        best = ranked_results[0]
        p = best['fom']['params']
        a = p['a_nm']
        print(f"\nBest result (Rank 1) physical dimensions:")
        print(f"  Lattice constant a = {a:.1f} nm")
        print(f"  Spine half-width   = {p['h_spine']*a:.1f} nm  (full width {2*p['h_spine']*a:.1f} nm)")
        print(f"  Top rib:  Wt = {p['Wt']*a:.1f} nm,  ht = {p['ht']*a:.1f} nm")
        print(f"  Bot rib:  Wb = {p['Wb']*a:.1f} nm,  hb = {p['hb']*a:.1f} nm")
        print(f"  Offset delta_s = {p['delta_s']*a:.1f} nm")
        print(f"  FOM: bandwidth = {best['fom']['fom']:.2f} nm  ng_mean = {best['fom']['ng_mean']:.2f}")
        print(f"  Center wavelength: {best['fom']['wl_center']:.1f} nm")
        print(f"  Meets target: {best['fom']['meets_target']}")


def export_sweep_csv(ranked_results, path=None):
    """Export ranked results to CSV. Default: output/sweep_results.csv."""
    if path is None:
        path = os.path.join(get_output_dir(), 'sweep_results.csv')

    fieldnames = [
        'rank', 'a_nm', 'h_spine', 'Wt', 'ht', 'Wb', 'hb', 'delta_s',
        'h_spine_nm', 'Wt_nm', 'ht_nm', 'Wb_nm', 'hb_nm', 'delta_s_nm',
        'band', 'bw_nm', 'ng_mean', 'ng_median', 'wl_center_nm',
        'meets_target', 'params_hash',
    ]

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in ranked_results:
            fom = r['fom']
            p   = fom['params']
            a   = p['a_nm']
            row = {
                'rank':         r.get('rank', ''),
                'a_nm':         a,
                'h_spine':      p['h_spine'],
                'Wt':           p['Wt'],
                'ht':           p['ht'],
                'Wb':           p['Wb'],
                'hb':           p['hb'],
                'delta_s':      p['delta_s'],
                'h_spine_nm':   p['h_spine'] * a,
                'Wt_nm':        p['Wt'] * a,
                'ht_nm':        p['ht'] * a,
                'Wb_nm':        p['Wb'] * a,
                'hb_nm':        p['hb'] * a,
                'delta_s_nm':   p['delta_s'] * a,
                'band':         fom['band'],
                'bw_nm':        fom['fom'],
                'ng_mean':      fom['ng_mean'],
                'ng_median':    fom['ng_median'],
                'wl_center_nm': fom['wl_center'],
                'meets_target': fom['meets_target'],
                'params_hash':  params_hash(p),
            }
            writer.writerow(row)

    print(f"Exported {len(ranked_results)} results to {path}")
    return path


# =====================================================================
# Plotting
# =====================================================================

def plot_bands_and_ng(data, best_band=None, save_path=None, show=True):
    """Three-panel figure: geometry | band diagram | ng vs wavelength."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    freqs  = data['freqs']
    k_x    = data['k_x']
    eps    = data['epsilon']
    sy     = data['sy']
    a_nm   = data['params']['a_nm']
    n_SiO2 = data['params']['n_SiO2']

    if best_band is None:
        best_band = detect_slow_light_band(freqs, k_x, a_nm)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 0: Geometry ---
    ax0 = axes[0]
    ax0.imshow(eps.T, origin='lower', cmap='RdBu_r',
               extent=[-0.5, 0.5, -sy/2, sy/2], aspect='auto')
    ax0.set_xlabel('x (units of a)')
    ax0.set_ylabel('y (units of a)')
    ax0.set_title('Unit cell (epsilon)')

    # --- Panel 1: Band diagram (zoomed to f=0.25–0.35) ---
    ax1 = axes[1]
    f_view_lo, f_view_hi = 0.25, 0.35
    # Light cone (only in view range)
    k_lc = np.linspace(k_x[0], k_x[-1], 100)
    f_lc = k_lc / n_SiO2
    ax1.fill_between(k_lc, np.maximum(f_lc, f_view_lo), f_view_hi,
                     alpha=0.15, color='gray', label='Light cone')
    ax1.plot(k_lc, f_lc, color='gray', lw=0.8)
    # All bands (only plot portions in view range)
    for b in range(freqs.shape[1]):
        f_band = freqs[:, b]
        in_view = (f_band >= f_view_lo - 0.01) & (f_band <= f_view_hi + 0.01)
        if not np.any(in_view):
            continue
        lw = 2.5 if b == best_band else 0.8
        color = 'red' if b == best_band else '#555555'
        zorder = 3 if b == best_band else 2
        label = f'Band {b}' if b == best_band else None
        ax1.plot(k_x, f_band, color=color, lw=lw, zorder=zorder, label=label)
    # Target frequency line
    f_target = a_nm / WL_CENTER
    if f_view_lo < f_target < f_view_hi:
        ax1.axhline(f_target, color='blue', ls='--', lw=0.8, label=f'λ={WL_CENTER:.0f}nm')
    ax1.set_xlabel('Wave vector (2π/a)')
    ax1.set_ylabel('Frequency (a/λ)')
    ax1.set_title(f'Band diagram  [band {best_band} in red]')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.set_xlim(k_x[0], k_x[-1])
    ax1.set_ylim(f_view_lo, f_view_hi)

    # --- Panel 2: ng vs wavelength ---
    ax2 = axes[2]
    f_b = freqs[:, best_band]
    ng, wl = compute_ng(f_b, k_x, a_nm)
    mask = (wl >= WL_MIN) & (wl <= WL_MAX) & np.isfinite(ng) & (ng > 0) & (ng < 60)
    if np.any(mask):
        ax2.plot(wl[mask], ng[mask], 'r-o', ms=3, lw=1.5)

    ax2.axhspan(NG_LOW, NG_HIGH, alpha=0.15, color='green', label=f'ng ∈ [{NG_LOW},{NG_HIGH}]')
    ax2.axhline(TARGET_NG, color='green', ls='--', lw=1.0, label=f'ng={TARGET_NG}')
    ax2.axvline(WL_CENTER, color='blue', ls='--', lw=0.8, label=f'{WL_CENTER:.0f}nm')

    # Annotate flat-region bandwidth
    fom = compute_fom(data, band_override=best_band)
    bw = fom['fom']
    if bw > 0 and np.isfinite(fom['wl_center']):
        wl_lo, wl_hi = fom['wl_range']
        ax2.axvspan(wl_lo, wl_hi, alpha=0.15, color='yellow',
                    label=f'Flat region')
        ng_min = fom.get('ng_min', fom['ng_mean'])
        ng_max = fom.get('ng_max', fom['ng_mean'])
        ax2.text(0.05, 0.95,
                 f'Flat BW = {bw:.1f} nm\n'
                 f'ng = {fom["ng_mean"]:.2f} [{ng_min:.1f}, {ng_max:.1f}]',
                 transform=ax2.transAxes, va='top', fontsize=9,
                 bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Group index ng')
    ax2.set_title(f'ng vs wavelength  [band {best_band}]')
    ax2.set_xlim(WL_MIN, WL_MAX)
    ax2.set_ylim(0, 20)
    ax2.legend(fontsize=8)

    p = data['params']
    a = p['a_nm']
    fig.suptitle(
        f"a={a:.0f}nm  h_spine={p['h_spine']*a:.0f}nm  "
        f"Wt={p['Wt']*a:.0f}nm ht={p['ht']*a:.0f}nm  "
        f"Wb={p['Wb']*a:.0f}nm hb={p['hb']*a:.0f}nm  "
        f"Δs={p['delta_s']*a:.0f}nm",
        fontsize=10
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_sweep_summary(ranked_results, save_path=None, show=True, label='phase1'):
    """Summary scatter plots: FOM vs each parameter axis."""
    import matplotlib.pyplot as plt

    if not ranked_results:
        return

    a_vals  = [r['fom']['params']['a_nm']            for r in ranked_results]
    hs_vals = [r['fom']['params']['h_spine'] * r['fom']['params']['a_nm'] for r in ranked_results]
    Wt_vals = [r['fom']['params']['Wt']      * r['fom']['params']['a_nm'] for r in ranked_results]
    ht_vals = [r['fom']['params']['ht']      * r['fom']['params']['a_nm'] for r in ranked_results]
    ds_vals = [r['fom']['params']['delta_s'] * r['fom']['params']['a_nm'] for r in ranked_results]
    foms    = [r['fom']['fom'] for r in ranked_results]
    meets   = [r['fom']['meets_target'] for r in ranked_results]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(a_vals), max(a_vals))

    def scatter_ax(ax, x, xlabel):
        sc = ax.scatter(x, foms, c=a_vals, cmap=cmap, norm=norm,
                        s=20, alpha=0.7, edgecolors='none')
        # Highlight meets-target points
        xt = [xi for xi, m in zip(x, meets) if m]
        yt = [fi for fi, m in zip(foms, meets) if m]
        ax.scatter(xt, yt, s=80, color='red', zorder=5, marker='*', label='Meets target')
        ax.axhline(MIN_BW_NM, color='k', ls='--', lw=0.8, label=f'Min BW={MIN_BW_NM}nm')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('FOM (nm bandwidth)')
        ax.legend(fontsize=7)
        return sc

    scatter_ax(axes[0], a_vals,  'a (nm)')
    scatter_ax(axes[1], hs_vals, 'Spine width (nm)')
    scatter_ax(axes[2], Wt_vals, 'Rib width (nm)')
    sc = scatter_ax(axes[3], ht_vals, 'Rib height (nm)')

    plt.colorbar(sc, ax=axes[3], label='a (nm)')
    fig.suptitle(f'Sweep summary — {label}', fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_top_ng_curves(top_results, n_top=10, save_path=None, show=True):
    """Overlay ng(λ) curves for top-n parameter sets."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if not top_results:
        return

    subset = [r for r in top_results[:n_top] if r['data'] is not None]
    if not subset:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cm.tab10(np.linspace(0, 1, len(subset)))

    ax.axhspan(NG_LOW, NG_HIGH, alpha=0.1, color='green', label=f'ng ∈ [{NG_LOW},{NG_HIGH}]')
    ax.axhline(TARGET_NG, color='green', ls='--', lw=1.0)
    ax.axvline(WL_CENTER, color='gray', ls='--', lw=0.8)

    for r, color in zip(subset, colors):
        data = r['data']
        f    = r['fom']
        p    = f['params']
        a    = p['a_nm']
        band = f['band']

        freqs = data['freqs']
        k_x   = data['k_x']
        ng, wl = compute_ng(freqs[:, band], k_x, a)
        mask = (wl >= WL_MIN) & (wl <= WL_MAX) & np.isfinite(ng) & (ng > 0) & (ng < 60)
        if not np.any(mask):
            continue

        label = (f"#{r['rank']} a={a:.0f} Wt={p['Wt']*a:.0f} ht={p['ht']*a:.0f} "
                 f"Wb={p['Wb']*a:.0f} hb={p['hb']*a:.0f} Δs={p['delta_s']*a:.0f} "
                 f"BW={f['fom']:.1f}nm")
        ax.plot(wl[mask], ng[mask], color=color, lw=1.5, label=label)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Group index ng')
    ax.set_title(f'Top-{n_top} ng curves (ng target = {TARGET_NG})')
    ax.set_xlim(WL_MIN, WL_MAX)
    ax.set_ylim(0, 50)
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# =====================================================================
# Sweep state persistence (to survive between --sweep and --sweep-asym)
# =====================================================================

def _phase1_results_path():
    return os.path.join(get_output_dir(), 'phase1_ranked.json')


def _to_serializable(obj):
    """Recursively convert any object to JSON-safe Python primitives."""
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_serializable(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (v != v) else v  # NaN → null
    if isinstance(obj, float):
        return None if (obj != obj) else obj  # NaN → null
    if isinstance(obj, (int, str, bool, type(None))):
        return obj
    # Fallback: convert to string to avoid serialization errors
    try:
        json_str = json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def save_phase1_results(ranked_results):
    """Save Phase 1 ranked FOMs to JSON (params only, not full MPB data)."""
    data = []
    for r in ranked_results:
        entry = dict(rank=r.get('rank'), fom_dict=_to_serializable(r['fom']))
        data.append(entry)
    with open(_phase1_results_path(), 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Phase 1 results saved to {_phase1_results_path()}")


def load_phase1_results():
    """Load Phase 1 ranked FOMs from JSON."""
    with open(_phase1_results_path()) as f:
        return json.load(f)


# =====================================================================
# CLI entry point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='2D MPB sweep and optimization for asymmetric simplified '
                    'fishbone slow-light grating coupler (ng≈6, BW>5nm)'
    )
    # Modes
    parser.add_argument('--sweep',       action='store_true',
                        help='Run Phase 1 symmetric parameter sweep')
    parser.add_argument('--sweep-asym',  action='store_true',
                        help='Run Phase 2 asymmetric sweep around best Phase 1 results')
    parser.add_argument('--optimize',    action='store_true',
                        help='Run Phase 1 + Phase 2 full optimization workflow')
    parser.add_argument('--plot-only',   action='store_true',
                        help='Plot top results from cache (no new MPB runs)')
    parser.add_argument('--params',      type=str, nargs='+', metavar='KEY=VALUE',
                        help='Single run with custom params, e.g. --params a_nm=420 Wt=0.45')

    # Simulation control
    parser.add_argument('--resolution',  type=int, default=RESOLUTION)
    parser.add_argument('--num-bands',   type=int, default=NUM_BANDS)
    parser.add_argument('--k-interp',    type=int, default=K_INTERP)
    parser.add_argument('--force-rerun', action='store_true',
                        help='Ignore cache and recompute all MPB runs')

    # Grid sizes
    parser.add_argument('--n-a',     type=int, default=5,  help='Phase 1: # a_nm values')
    parser.add_argument('--n-spine', type=int, default=4,  help='Phase 1: # h_spine values')
    parser.add_argument('--n-W',     type=int, default=4,  help='Phase 1: # W values')
    parser.add_argument('--n-h',     type=int, default=4,  help='Phase 1: # h values')
    parser.add_argument('--n-ratio', type=int, default=5,  help='Phase 2: # ratio values')
    parser.add_argument('--n-delta', type=int, default=5,  help='Phase 2: # delta_s values')
    parser.add_argument('--n-top-p2', type=int, default=5, help='Phase 2: top-N from Phase 1')

    # Output
    parser.add_argument('--n-top',     type=int, default=10, help='Number of top results to show')
    parser.add_argument('--export-csv', action='store_true', help='Export results to CSV')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to output/')
    parser.add_argument('--no-show',    action='store_true', help='Do not display plots')

    args = parser.parse_args()

    show_plots = not args.no_show
    save_plots = args.save_plots
    out = get_output_dir()

    sim_kw = dict(
        resolution=args.resolution,
        num_bands=args.num_bands,
        k_interp=args.k_interp,
        force_rerun=args.force_rerun,
    )

    # ------------------------------------------------------------------
    # --params: single run
    # ------------------------------------------------------------------
    if args.params:
        params = get_default_params()
        float_keys = {'a_nm', 'h_spine', 'Wt', 'ht', 'Wb', 'hb',
                      'delta_s', 'n_eff', 'n_SiO2', 'pad_y'}
        for kv in args.params:
            k, v = kv.split('=', 1)
            params[k] = float(v) if k in float_keys else int(v)

        print("Running single MPB simulation with params:")
        for k, v in params.items():
            a = params['a_nm']
            nm_str = f"  ({v*a:.1f}nm)" if k not in ('a_nm', 'n_eff', 'n_SiO2', 'pad_y') else ""
            print(f"  {k} = {v}{nm_str}")

        try:
            validate_params(params)
        except ValueError as e:
            print(f"Fabrication constraint error:\n{e}")
            sys.exit(1)

        data = run_or_load(params, **sim_kw)
        fom  = compute_fom(data)
        print(f"\nFOM: bandwidth = {fom['fom']:.2f} nm,  ng_mean = {fom['ng_mean']:.2f},  "
              f"wl_center = {fom['wl_center']:.1f} nm")
        print(f"Meets target: {fom['meets_target']}")

        sp = os.path.join(out, f"single_{params_hash(params)}.png") if save_plots else None
        plot_bands_and_ng(data, best_band=fom['band'],
                         save_path=sp, show=show_plots)
        return

    # ------------------------------------------------------------------
    # --plot-only: scan cache and plot
    # ------------------------------------------------------------------
    if args.plot_only:
        cache_dir = get_cache_dir()
        npz_files = [f for f in os.listdir(cache_dir) if f.endswith('.npz')]
        if not npz_files:
            print("No cached results found.")
            return

        print(f"Loading {len(npz_files)} cached results...")
        results = []
        for fname in npz_files:
            try:
                data = load_results(os.path.join(cache_dir, fname))
                fom  = compute_fom(data)
                results.append(dict(data=data, fom=fom))
            except Exception as e:
                print(f"  Skip {fname}: {e}")

        ranked = rank_results(results)
        print_top_results(ranked, args.n_top)

        sp = os.path.join(out, 'top10_ng_curves.png') if save_plots else None
        plot_top_ng_curves(ranked, n_top=args.n_top,
                           save_path=sp, show=show_plots)

        if ranked[0]['data'] is not None:
            sp2 = os.path.join(out, 'best_bands_and_ng.png') if save_plots else None
            plot_bands_and_ng(ranked[0]['data'],
                              best_band=ranked[0]['fom']['band'],
                              save_path=sp2, show=show_plots)

        if args.export_csv:
            export_sweep_csv(ranked)
        return

    # ------------------------------------------------------------------
    # --sweep / --optimize: Phase 1
    # ------------------------------------------------------------------
    if args.sweep or args.optimize:
        print(f"\n{'='*60}")
        print(f"PHASE 1: Symmetric sweep")
        print(f"  n_a={args.n_a}, n_spine={args.n_spine}, n_W={args.n_W}, n_h={args.n_h}")
        param_list = generate_phase1_sweep(args.n_a, args.n_spine, args.n_W, args.n_h)
        print(f"  Total parameter sets: {len(param_list)}")
        print(f"{'='*60}")

        results = run_sweep(param_list, **sim_kw)
        ranked  = rank_results(results)

        print_top_results(ranked, args.n_top)
        save_phase1_results(ranked)

        sp1 = os.path.join(out, 'sweep_phase1_summary.png') if save_plots else None
        plot_sweep_summary(ranked, save_path=sp1, show=show_plots, label='Phase 1 symmetric')

        sp2 = os.path.join(out, 'phase1_top10_ng.png') if save_plots else None
        plot_top_ng_curves(ranked, n_top=args.n_top,
                           save_path=sp2, show=show_plots)

        if args.export_csv:
            export_sweep_csv(ranked, os.path.join(out, 'phase1_results.csv'))

        if not args.optimize:
            return

        # Hand off to Phase 2 via the optimize path below
        phase1_ranked = ranked

    # ------------------------------------------------------------------
    # --sweep-asym / --optimize: Phase 2
    # ------------------------------------------------------------------
    if args.sweep_asym or args.optimize:
        if not (args.sweep or args.optimize):
            # Load Phase 1 results from disk
            if not os.path.exists(_phase1_results_path()):
                print("No Phase 1 results found. Run --sweep first.")
                sys.exit(1)
            raw = load_phase1_results()
            phase1_ranked = [dict(data=None, fom=r['fom_dict'],
                                  rank=r['rank']) for r in raw]
            print(f"Loaded {len(phase1_ranked)} Phase 1 results.")

        n_top_p2 = args.n_top_p2
        top_bases = [r for r in phase1_ranked[:n_top_p2] if r['fom']['fom'] > 0]

        if not top_bases:
            print("No Phase 1 results with FOM > 0. Cannot run Phase 2.")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"PHASE 2: Asymmetric sweep around top-{len(top_bases)} Phase 1 results")
        print(f"  n_ratio={args.n_ratio}, n_delta={args.n_delta}")
        print(f"{'='*60}")

        all_p2_results = []
        for i, base in enumerate(top_bases):
            bp = base['fom']['params']
            print(f"\n--- Phase 2 base {i+1}/{len(top_bases)}: "
                  f"a={bp['a_nm']:.0f}nm Wt={bp['Wt']*bp['a_nm']:.0f}nm "
                  f"ht={bp['ht']*bp['a_nm']:.0f}nm ---")
            p2_list = generate_phase2_sweep(bp, args.n_ratio, args.n_delta)
            print(f"  {len(p2_list)} parameter sets")
            p2_res = run_sweep(p2_list, **sim_kw)
            all_p2_results.extend(p2_res)

        ranked_p2 = rank_results(all_p2_results)
        print("\nPhase 2 results:")
        print_top_results(ranked_p2, args.n_top)

        sp1 = os.path.join(out, 'sweep_phase2_summary.png') if save_plots else None
        plot_sweep_summary(ranked_p2, save_path=sp1, show=show_plots, label='Phase 2 asymmetric')

        sp2 = os.path.join(out, 'phase2_top10_ng.png') if save_plots else None
        plot_top_ng_curves(ranked_p2, n_top=args.n_top,
                           save_path=sp2, show=show_plots)

        if args.export_csv:
            export_sweep_csv(ranked_p2, os.path.join(out, 'phase2_results.csv'))

        # Combined ranking
        if args.optimize:
            print(f"\n{'='*60}")
            print("COMBINED Phase 1 + Phase 2 ranking:")
            all_combined = list(phase1_ranked) + all_p2_results
            # Re-load data for Phase 1 entries that have data=None
            for r in all_combined:
                if r['data'] is None:
                    p = r['fom']['params']
                    cpath = cache_path(p, args.resolution, args.num_bands)
                    if os.path.exists(cpath):
                        try:
                            r['data'] = load_results(cpath)
                        except Exception:
                            pass
            combined_ranked = rank_results(all_combined)
            print_top_results(combined_ranked, args.n_top)

            sp3 = os.path.join(out, 'combined_top10_ng.png') if save_plots else None
            plot_top_ng_curves(combined_ranked, n_top=args.n_top,
                               save_path=sp3, show=show_plots)

            if combined_ranked[0]['data'] is not None:
                sp4 = os.path.join(out, 'best_result_bands_ng.png') if save_plots else None
                plot_bands_and_ng(combined_ranked[0]['data'],
                                  best_band=combined_ranked[0]['fom']['band'],
                                  save_path=sp4, show=show_plots)

            if args.export_csv:
                export_sweep_csv(combined_ranked,
                                 os.path.join(out, 'combined_results.csv'))

        return

    # No mode selected
    parser.print_help()


if __name__ == '__main__':
    main()
