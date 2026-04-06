import argparse
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import meep as mp
import meep.mpb as mpb

from sweep_ysym_partial_etch import (
    T_SLAB_NM, N_POLY_SI, N_SIO2, PAD_Y, PAD_Z,
    K_MIN, K_MAX, K_INTERP
)
BASE_PARAMS = dict(
    a_nm=496.0, h_spine=0.550,
    W_rib=0.484, h_rib=0.514,   # single rib size (both sides identical)
)


def build_geometry_ysym_140nm(params):
    """Y-symmetric fishbone: identical ribs on +y and -y, delta_s=0.
    For hsi = 140nm partial etch depth."""
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
    t_partial = 140.0 / a_nm
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

def run_mpb_yeven(params, k_min=K_MIN, k_max=K_MAX, k_interp=K_INTERP, num_bands=30, resolution=16):
    """Run MPB with Y-EVEN symmetry for the 140nm partial etch structure."""
    import meep as mp
    import meep.mpb as mpb
    import numpy as np

    lattice, geometry, sy, sz, t_slab = build_geometry_ysym_140nm(params)
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
    print(f"3D MPB (Y-even Only): a={a:.0f}nm  t_partial=140nm  res={resolution}  bands={num_bands}")
    print(f"{'='*70}")

    print("Running with ms.run_yeven() to target specific symmetric modes...")
    ms.run_yeven()
    
    all_freqs = np.array(ms.all_freqs)
    k_x = np.array([k.x for k in k_points])
    eps = np.array(ms.get_epsilon())

    return dict(
        all_freqs=all_freqs,
        k_x=k_x,
        epsilon=eps,
        sy=sy, sz=sz, t_slab=t_slab,
        params=dict(params),
        t_partial_nm=140.0,
        resolution=resolution,
        num_bands=num_bands
    )

def plot_bands(data, out_path):
    """Plot the band diagram of the computed modes."""
    import matplotlib.pyplot as plt
    import numpy as np

    k_x = data['k_x']
    all_freqs = data['all_freqs']
    a_nm = data['params']['a_nm']

    f_lo, f_hi = 0.20, 0.35
    k_lc = np.linspace(k_x[0], k_x[-1], 100)
    f_lc = k_lc / N_SIO2
    f_1550 = a_nm / 1550.0

    plt.figure(figsize=(8, 6))
    
    # Light cone
    plt.fill_between(k_lc, np.maximum(f_lc, f_lo), f_hi, alpha=0.12, color='gray', label='Light cone')
    plt.plot(k_lc, f_lc, color='gray', lw=1.0)
    
    # Plot all bands
    num_bands = all_freqs.shape[1]
    for b in range(num_bands):
        f_b = all_freqs[:, b]
        if np.any((f_b >= f_lo - 0.01) & (f_b <= f_hi + 0.01)):
            plt.plot(k_x, f_b, '-', lw=1.5, label=f'Y-even Band {b}' if b < 5 else None)

    if f_lo < f_1550 < f_hi:
        plt.axhline(f_1550, color='red', ls='--', lw=1.0, label='1550nm')

    plt.xlabel('Wave vector (2\u03c0/a)')
    plt.ylabel('Frequency (a/\u03bb)')
    plt.title('Band Diagram: hsi = 140.0 nm (Y-even Symmetry)')
    plt.xlim(k_x[0], k_x[-1])
    plt.ylim(f_lo, f_hi)
    
    # Reduce clutter in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=16)
    parser.add_argument('--num-bands', type=int, default=30)
    parser.add_argument('--save-plots', action='store_true')
    parser.add_argument('--save-data', action='store_true')
    parser.add_argument('--k-min', type=float, default=K_MIN)
    parser.add_argument('--k-max', type=float, default=K_MAX)
    parser.add_argument('--k-interp', type=int, default=K_INTERP)
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    
    data = run_mpb_yeven(
        BASE_PARAMS, 
        k_min=args.k_min,
        k_max=args.k_max,
        k_interp=args.k_interp,
        resolution=args.resolution, 
        num_bands=args.num_bands
    )
    
    if args.save_plots:
        plot_path = os.path.join(out_dir, 'yeven_140nm_bands.png')
        plot_bands(data, plot_path)
    else:
        plot_bands(data, 'yeven_140nm_bands.png')
        
    if args.save_data:
        data_path = os.path.join(out_dir, 'yeven_140nm_data.npz')
        np.savez(data_path, **data)
        print(f"Saved data to {data_path}")

if __name__ == '__main__':
    main()