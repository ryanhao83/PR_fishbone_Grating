"""
Analyze 3D MPB cached results for single-mode slow-light operation.

Two criteria for single-mode:
1. No neighboring band's guided frequency range overlaps the slow-light window
2. The target band must be monotonic in the guided region where slow light occurs
   (non-monotonic => same freq maps to 2 different k => multimode)
"""

import numpy as np
import json
import os
import glob

CACHE_DIR = "/Users/bryan/codes/PR_fishbone_grating/simplified_structure/cache_3d"
N_CLAD = 1.44  # SiO2 cladding index
NG_LOW, NG_HIGH = 5.0, 7.0

def analyze_file(filepath):
    data = np.load(filepath, allow_pickle=True)
    params = json.loads(str(data["params_json"]))
    a_nm = params["a_nm"]
    k_x = data["k_x"]          # shape (Nk,)
    freqs = data["freqs_te"]    # shape (Nk, Nbands)
    Nk, Nbands = freqs.shape

    print("=" * 80)
    print(f"File: {os.path.basename(filepath)}")
    print(f"Params: {json.dumps(params, indent=2)}")
    print(f"k_x range: [{k_x[0]:.4f}, {k_x[-1]:.4f}], Nk={Nk}, Nbands={Nbands}")
    print()

    # Light line: f_light = k_x / n_clad  (in normalized freq units)
    f_light = k_x / N_CLAD

    # ---------- Per-band analysis ----------
    band_info = []
    slow_light_band = None
    slow_light_fmin = None
    slow_light_fmax = None
    slow_light_band_idx = None

    for b in range(Nbands):
        f_band = freqs[:, b]

        # Guided region: f_band < f_light (below light line)
        guided_mask = f_band < f_light
        # Also require f > 0 (skip zero-freq points)
        guided_mask &= f_band > 1e-6

        if guided_mask.sum() < 3:
            band_info.append({"band": b, "guided": False})
            continue

        # Get contiguous guided indices (take largest contiguous block near zone edge)
        guided_idx = np.where(guided_mask)[0]
        k_guided = k_x[guided_idx]
        f_guided = f_band[guided_idx]

        # Check monotonicity
        df = np.diff(f_guided)
        dk = np.diff(k_guided)
        is_monotonic = np.all(df >= -1e-10) or np.all(df <= 1e-10)
        is_increasing = np.all(df >= -1e-10)

        # Compute ng in guided region
        # Use central differences for interior, one-sided at edges
        if len(f_guided) >= 3:
            df_dk = np.gradient(f_guided, k_guided)
            ng = np.where(np.abs(df_dk) > 1e-12, 1.0 / df_dk, np.inf)
        else:
            ng = np.array([np.inf] * len(f_guided))

        # Find slow-light window: ng in [NG_LOW, NG_HIGH]
        sl_mask = (ng >= NG_LOW) & (ng <= NG_HIGH) & np.isfinite(ng)

        info = {
            "band": b,
            "guided": True,
            "guided_idx": guided_idx,
            "k_guided": k_guided,
            "f_guided": f_guided,
            "f_guided_min": f_guided.min(),
            "f_guided_max": f_guided.max(),
            "is_monotonic": is_monotonic,
            "is_increasing": is_increasing,
            "ng": ng,
            "sl_mask": sl_mask,
            "has_slow_light": sl_mask.sum() >= 2,
        }

        if sl_mask.sum() >= 2:
            sl_freqs = f_guided[sl_mask]
            sl_ks = k_guided[sl_mask]
            sl_ngs = ng[sl_mask]
            info["sl_fmin"] = sl_freqs.min()
            info["sl_fmax"] = sl_freqs.max()
            info["sl_kmin"] = sl_ks.min()
            info["sl_kmax"] = sl_ks.max()
            info["sl_ng_mean"] = sl_ngs.mean()
            info["sl_ng_min"] = sl_ngs.min()
            info["sl_ng_max"] = sl_ngs.max()

            # Convert to wavelength
            wl_min = a_nm / sl_freqs.max()
            wl_max = a_nm / sl_freqs.min()
            info["sl_wl_min"] = wl_min
            info["sl_wl_max"] = wl_max
            info["sl_bw_nm"] = wl_max - wl_min
            info["sl_wl_center"] = (wl_min + wl_max) / 2.0

            # Track the band with the widest slow-light BW as candidate
            if slow_light_band is None or info["sl_bw_nm"] > slow_light_band["sl_bw_nm"]:
                slow_light_band = info
                slow_light_band_idx = b

        band_info.append(info)

    # ---------- Print per-band summary ----------
    print(f"{'Band':>4}  {'Guided?':>7}  {'f_min':>8}  {'f_max':>8}  {'Monotonic':>9}  {'SL pts':>6}  {'SL BW(nm)':>9}  {'SL center(nm)':>13}")
    print("-" * 85)
    for info in band_info:
        b = info["band"]
        if not info["guided"]:
            print(f"{b:>4}  {'No':>7}")
            continue
        mono_str = "Yes" if info["is_monotonic"] else "NO!"
        sl_pts = info["sl_mask"].sum() if "sl_mask" in info else 0
        sl_bw = f"{info['sl_bw_nm']:.1f}" if info.get("has_slow_light") else "-"
        sl_ctr = f"{info['sl_wl_center']:.1f}" if info.get("has_slow_light") else "-"
        print(f"{b:>4}  {'Yes':>7}  {info['f_guided_min']:.5f}  {info['f_guided_max']:.5f}  {mono_str:>9}  {sl_pts:>6}  {sl_bw:>9}  {sl_ctr:>13}")
    print()

    if slow_light_band is None:
        print("*** NO SLOW-LIGHT BAND FOUND (no band has ng in [5,7] with >=2 points) ***")
        print()
        return

    sb = slow_light_band
    bidx = slow_light_band_idx
    print(f"--- Target slow-light band: {bidx} ---")
    print(f"  Slow-light freq range: [{sb['sl_fmin']:.6f}, {sb['sl_fmax']:.6f}]")
    print(f"  Slow-light k range:    [{sb['sl_kmin']:.4f}, {sb['sl_kmax']:.4f}]")
    print(f"  Slow-light wavelength: [{sb['sl_wl_min']:.1f}, {sb['sl_wl_max']:.1f}] nm")
    print(f"  Bandwidth:             {sb['sl_bw_nm']:.1f} nm")
    print(f"  Center wavelength:     {sb['sl_wl_center']:.1f} nm")
    print(f"  ng range:              [{sb['sl_ng_min']:.2f}, {sb['sl_ng_max']:.2f}]")
    print(f"  ng mean:               {sb['sl_ng_mean']:.2f}")
    print()

    # ---------- Criterion 1: Monotonicity of target band ----------
    print("--- Criterion 1: Monotonicity of target band in guided region ---")
    if sb["is_monotonic"]:
        print("  PASS: Band is monotonic in guided region.")
        backward_overlap = False
    else:
        print("  WARNING: Band is NON-MONOTONIC in guided region!")
        # Find where the band changes direction
        f_g = sb["f_guided"]
        k_g = sb["k_guided"]
        df = np.diff(f_g)

        # Identify increasing and decreasing segments
        # Find turning points
        sign_changes = np.where(np.diff(np.sign(df)))[0]
        print(f"  Turning points at k indices: {sign_changes}")
        for sc in sign_changes:
            print(f"    k={k_g[sc+1]:.4f}, f={f_g[sc+1]:.6f}")

        # Split into forward (df/dk > 0) and backward (df/dk < 0) branches
        # Check if backward branch frequencies overlap with slow-light window
        fwd_mask = np.zeros(len(f_g), dtype=bool)
        bwd_mask = np.zeros(len(f_g), dtype=bool)
        # Assign each point to forward or backward based on local derivative
        ng_g = sb["ng"]
        fwd_mask[ng_g > 0] = True   # positive group velocity = forward
        bwd_mask[ng_g < 0] = True   # negative group velocity = backward

        if bwd_mask.any():
            bwd_freqs = f_g[bwd_mask]
            bwd_fmin, bwd_fmax = bwd_freqs.min(), bwd_freqs.max()
            print(f"  Backward branch freq range: [{bwd_fmin:.6f}, {bwd_fmax:.6f}]")
            print(f"  Slow-light freq range:      [{sb['sl_fmin']:.6f}, {sb['sl_fmax']:.6f}]")

            # Check overlap
            overlap = (bwd_fmin <= sb["sl_fmax"]) and (bwd_fmax >= sb["sl_fmin"])
            if overlap:
                ol_fmin = max(bwd_fmin, sb["sl_fmin"])
                ol_fmax = min(bwd_fmax, sb["sl_fmax"])
                ol_wl_min = a_nm / ol_fmax
                ol_wl_max = a_nm / ol_fmin
                print(f"  *** OVERLAP detected: freq [{ol_fmin:.6f}, {ol_fmax:.6f}]")
                print(f"      Wavelength range of overlap: [{ol_wl_min:.1f}, {ol_wl_max:.1f}] nm")
                backward_overlap = True
            else:
                print("  No overlap between backward branch and slow-light window.")
                backward_overlap = False
        else:
            print("  (No backward branch detected despite non-monotonicity — possibly just flat)")
            backward_overlap = False
    print()

    # ---------- Criterion 2: Neighboring band overlap ----------
    print("--- Criterion 2: Neighboring bands overlap with slow-light window ---")
    neighbor_overlap = False
    for info in band_info:
        b = info["band"]
        if b == bidx or not info["guided"]:
            continue
        # Check if this band's guided frequency range overlaps the slow-light window
        fmin_b = info["f_guided_min"]
        fmax_b = info["f_guided_max"]
        overlap = (fmin_b <= sb["sl_fmax"]) and (fmax_b >= sb["sl_fmin"])
        if overlap:
            ol_fmin = max(fmin_b, sb["sl_fmin"])
            ol_fmax = min(fmax_b, sb["sl_fmax"])
            ol_wl_min = a_nm / ol_fmax
            ol_wl_max = a_nm / ol_fmin
            print(f"  *** Band {b} OVERLAPS: guided freq [{fmin_b:.6f}, {fmax_b:.6f}]")
            print(f"      Overlap region: freq [{ol_fmin:.6f}, {ol_fmax:.6f}], wl [{ol_wl_min:.1f}, {ol_wl_max:.1f}] nm")
            neighbor_overlap = True
        else:
            print(f"  Band {b}: guided freq [{fmin_b:.6f}, {fmax_b:.6f}] — no overlap")
    print()

    # ---------- VERDICT ----------
    print("=" * 50)
    if sb["is_monotonic"] and not neighbor_overlap:
        verdict = "SINGLE-MODE in slow-light region"
    elif not sb["is_monotonic"] and backward_overlap:
        if neighbor_overlap:
            verdict = "MULTIMODE — backward branch overlap AND neighbor band overlap"
        else:
            verdict = "MULTIMODE — backward branch of target band overlaps slow-light window"
    elif neighbor_overlap:
        verdict = "MULTIMODE — neighboring band overlaps slow-light window"
    else:
        verdict = "SINGLE-MODE in slow-light region (non-monotonic but no frequency overlap)"

    print(f"  VERDICT: {verdict}")
    print(f"  Band {bidx} | ng_mean={sb['sl_ng_mean']:.2f} | BW={sb['sl_bw_nm']:.1f} nm | center={sb['sl_wl_center']:.1f} nm")
    print("=" * 50)
    print()


def main():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    print(f"Found {len(files)} cached 3D results in {CACHE_DIR}\n")
    for f in files:
        analyze_file(f)


if __name__ == "__main__":
    main()
