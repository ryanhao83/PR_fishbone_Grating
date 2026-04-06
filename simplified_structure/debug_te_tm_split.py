"""
debug_te_tm_split.py — Diagnostic script for TE/TM polarization classification.

Re-runs MPB at all k-points to extract te_frac per (k, band), then:
  1. Plots band diagrams with TE/TM split at different te_frac thresholds
     (0.5, 0.6, 0.7, 0.8, 0.9) so you can visually pick the right cutoff.
  2. At a chosen k-point (default k=0.45), plots the |Ex|, |Ey|, |Ez|
     cross-section for each band to see the modal character directly.

Usage:
  python debug_te_tm_split.py                    # default: res=16, nb=6
  python debug_te_tm_split.py --resolution 32 --num-bands 16
  python debug_te_tm_split.py --k-probe 0.40     # change probe k-point
"""

import argparse
import os
import sys
import numpy as np

# Reuse geometry builder and constants from the main script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simplified_gc_3d import (
    build_geometry_3d, get_structures,
    N_SIO2, K_MIN, K_MAX, K_INTERP, NUM_BANDS, RESOLUTION,
)


def run_and_extract(params, k_min, k_max, k_interp, num_bands, resolution):
    """Run MPB k-by-k and return freqs, te_frac, and e-fields at each k-point."""
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

    num_k = len(k_points)
    nb = num_bands
    k_x = np.array([k.x for k in k_points])

    all_freqs = np.zeros((num_k, nb))
    te_frac = np.zeros((num_k, nb))

    # Store e-field arrays only at the probe k-point (saves memory)
    # We'll collect them in a second pass below
    print(f"  Solving {num_k} k-points x {nb} bands (res={resolution})...")
    for ik, k in enumerate(k_points):
        ms.k_points = [k]
        ms.run()
        all_freqs[ik, :] = ms.all_freqs[0]

        for ib in range(nb):
            ms.get_dfield(ib + 1)
            energy = ms.compute_field_energy()
            total = energy[0]
            if total > 0:
                ez_frac = energy[3] / total
                te_frac[ik, ib] = 1.0 - ez_frac
            else:
                te_frac[ik, ib] = np.nan

        if (ik + 1) % 5 == 0 or ik == num_k - 1:
            print(f"    k-point {ik+1}/{num_k}")

    eps = np.array(ms.get_epsilon())
    return ms, k_points, k_x, all_freqs, te_frac, eps, sy, sz


def extract_fields_at_k(ms, k_probe_vec, num_bands):
    """Re-solve at a single k-point and extract E-field arrays for all bands."""
    import meep as mp

    ms.k_points = [k_probe_vec]
    ms.run()

    efields = []
    for ib in range(num_bands):
        # get_efield returns complex field; calling twice is the MPB pattern
        ms.get_efield(ib + 1, False)
        ef = np.array(ms.get_efield(ib + 1, False))
        efields.append(ef)
    return efields


def plot_threshold_comparison(k_x, all_freqs, te_frac, a_nm, thresholds,
                              save_path=None):
    """Plot band diagrams side-by-side for different te_frac thresholds."""
    import matplotlib.pyplot as plt

    n_thresh = len(thresholds)
    fig, axes = plt.subplots(1, n_thresh, figsize=(5 * n_thresh, 5), sharey=True)
    if n_thresh == 1:
        axes = [axes]

    f_lo, f_hi = 0.20, 0.38
    k_lc = np.linspace(k_x[0], k_x[-1], 100)
    f_lc = k_lc / N_SIO2
    f_1550 = a_nm / 1550.0
    nb = all_freqs.shape[1]

    for ax, thresh in zip(axes, thresholds):
        # Light cone
        ax.fill_between(k_lc, np.maximum(f_lc, f_lo), f_hi,
                        alpha=0.10, color='gray')
        ax.plot(k_lc, f_lc, color='gray', lw=0.6)

        te_plotted = False
        tm_plotted = False
        for b in range(nb):
            f_b = all_freqs[:, b]
            tf_b = te_frac[:, b]

            # Check if any part of this band is in view
            if not np.any((f_b >= f_lo - 0.01) & (f_b <= f_hi + 0.01)):
                continue

            # TE portions
            f_te = np.where(tf_b > thresh, f_b, np.nan)
            if np.any(np.isfinite(f_te)):
                label = f'TE (>{thresh})' if not te_plotted else None
                ax.plot(k_x, f_te, 'b-', lw=1.2, label=label)
                te_plotted = True

            # TM portions
            f_tm = np.where(tf_b <= thresh, f_b, np.nan)
            if np.any(np.isfinite(f_tm)):
                label = f'TM (<={thresh})' if not tm_plotted else None
                ax.plot(k_x, f_tm, 'r--', lw=1.2, label=label)
                tm_plotted = True

        if f_lo < f_1550 < f_hi:
            ax.axhline(f_1550, color='green', ls=':', lw=0.8, label='1550nm')

        ax.set_xlim(k_x[0], k_x[-1])
        ax.set_ylim(f_lo, f_hi)
        ax.set_xlabel('k (2pi/a)')
        ax.set_title(f'te_frac > {thresh}')
        ax.legend(fontsize=6, loc='upper left')

    axes[0].set_ylabel('Frequency (a/lambda)')
    fig.suptitle(f'TE/TM split at different thresholds (a={a_nm:.0f}nm)', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close(fig)


def plot_te_frac_heatmap(k_x, all_freqs, te_frac, a_nm, save_path=None):
    """Heatmap of te_frac vs (k, band) — shows where hybridization occurs."""
    import matplotlib.pyplot as plt

    nb = all_freqs.shape[1]
    fig, ax = plt.subplots(figsize=(8, 5))

    for b in range(nb):
        f_b = all_freqs[:, b]
        tf_b = te_frac[:, b]
        sc = ax.scatter(k_x, f_b, c=tf_b, cmap='coolwarm', vmin=0, vmax=1,
                        s=8, zorder=2)

    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('TE fraction')

    f_1550 = a_nm / 1550.0
    ax.axhline(f_1550, color='green', ls=':', lw=0.8, label='1550nm')

    k_lc = np.linspace(k_x[0], k_x[-1], 100)
    f_lc = k_lc / N_SIO2
    ax.plot(k_lc, f_lc, color='gray', lw=0.6, label='light line')

    ax.set_xlim(k_x[0], k_x[-1])
    ax.set_ylim(0.20, 0.38)
    ax.set_xlabel('k (2pi/a)')
    ax.set_ylabel('Frequency (a/lambda)')
    ax.set_title(f'TE fraction per (k, band) — a={a_nm:.0f}nm')
    ax.legend(fontsize=7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close(fig)


def plot_modal_fields(efields, freqs_at_k, te_frac_at_k, k_probe, a_nm,
                      sy, sz, save_path=None):
    """Plot |Ex|^2, |Ey|^2, |Ez|^2 cross-sections (y-z plane, x=mid) for each band."""
    import matplotlib.pyplot as plt

    nb = len(efields)
    fig, axes = plt.subplots(nb, 3, figsize=(10, 2.5 * nb))
    if nb == 1:
        axes = axes[np.newaxis, :]

    comp_labels = ['|Ex|^2', '|Ey|^2', '|Ez|^2']

    for ib in range(nb):
        ef = efields[ib]  # shape: (nx, ny, nz, 3) complex
        nx = ef.shape[0]
        x_mid = nx // 2

        # Take y-z slice at x = x_mid
        for ic in range(3):
            ax = axes[ib, ic]
            field_slice = np.abs(ef[x_mid, :, :, ic])**2
            im = ax.imshow(field_slice.T, origin='lower', cmap='hot',
                           aspect='auto',
                           extent=[-sy/2, sy/2, -sz/2, sz/2])
            if ic == 0:
                f_val = freqs_at_k[ib]
                wl = a_nm / f_val if f_val > 0 else 0
                tf = te_frac_at_k[ib]
                ax.set_ylabel(f'band {ib}\nf={f_val:.4f}\n'
                              f'wl={wl:.0f}nm\n'
                              f'TE={tf:.2f}',
                              fontsize=7, rotation=0, labelpad=60,
                              va='center')
            if ib == 0:
                ax.set_title(comp_labels[ic])
            if ib == nb - 1:
                ax.set_xlabel('y (a)')
            else:
                ax.set_xticklabels([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'Modal E-field at k={k_probe:.3f} (a={a_nm:.0f}nm)\n'
                 f'y-z cross section at x=0 (one period center)',
                 fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Debug TE/TM classification by testing different te_frac thresholds')
    parser.add_argument('--resolution', type=int, default=RESOLUTION)
    parser.add_argument('--num-bands', type=int, default=NUM_BANDS)
    parser.add_argument('--k-interp', type=int, default=K_INTERP)
    parser.add_argument('--k-probe', type=float, default=0.45,
                        help='k-point at which to plot modal fields')
    parser.add_argument('--struct-id', type=int, default=0,
                        help='Structure index (0-2)')
    parser.add_argument('--save-plots', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    structs = get_structures()
    s = structs[args.struct_id]
    label = s.pop('label')
    params = dict(s)
    a_nm = params['a_nm']
    print(f"Structure: {label}")
    print(f"  a={a_nm}nm, res={args.resolution}, bands={args.num_bands}")

    # --- Step 1: Run MPB and extract te_frac ---
    ms, k_points, k_x, all_freqs, te_frac, eps, sy, sz = run_and_extract(
        params, K_MIN, K_MAX, args.k_interp,
        args.num_bands, args.resolution)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out, exist_ok=True)

    # Print te_frac summary
    print(f"\n  te_frac summary per band (median across k-points):")
    for b in range(args.num_bands):
        med = np.nanmedian(te_frac[:, b])
        lo = np.nanmin(te_frac[:, b])
        hi = np.nanmax(te_frac[:, b])
        f_med = np.nanmedian(all_freqs[:, b])
        wl = a_nm / f_med if f_med > 0 else 0
        print(f"    band {b:2d}: te_frac = {med:.3f} [{lo:.3f} - {hi:.3f}]  "
              f"f={f_med:.4f}  wl~{wl:.0f}nm")

    # --- Step 2: Plot threshold comparison ---
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    sp = os.path.join(out, 'debug3_te_tm_thresholds.png') if args.save_plots else None
    plot_threshold_comparison(k_x, all_freqs, te_frac, a_nm, thresholds,
                             save_path=sp)

    # --- Step 3: Plot te_frac heatmap ---
    sp = os.path.join(out, 'debug3_te_frac_heatmap.png') if args.save_plots else None
    plot_te_frac_heatmap(k_x, all_freqs, te_frac, a_nm, save_path=sp)

    skip=1
    # --- Step 4: Extract and plot modal fields at k_probe ---
    if skip != 1:
        import meep as mp
        k_probe = args.k_probe
        k_probe_vec = mp.Vector3(k_probe)
        print(f"\n  Extracting E-fields at k={k_probe}...")
        efields = extract_fields_at_k(ms, k_probe_vec, args.num_bands)

        # Get freqs and te_frac at the probe k-point
        # Find closest k index
        ik_probe = int(np.argmin(np.abs(k_x - k_probe)))
        freqs_at_k = all_freqs[ik_probe, :]
        te_frac_at_k = te_frac[ik_probe, :]

        print(f"  Closest k-point: k={k_x[ik_probe]:.4f} (index {ik_probe})")
        for ib in range(args.num_bands):
            wl = a_nm / freqs_at_k[ib] if freqs_at_k[ib] > 0 else 0
            print(f"    band {ib}: f={freqs_at_k[ib]:.4f}  wl={wl:.0f}nm  "
                f"te_frac={te_frac_at_k[ib]:.3f}")

        sp = os.path.join(out, 'debug3_modal_fields.png') if args.save_plots else None
        plot_modal_fields(efields, freqs_at_k, te_frac_at_k, k_probe, a_nm,
                        sy, sz, save_path=sp)

    print("\nDone.")


if __name__ == '__main__':
    main()
