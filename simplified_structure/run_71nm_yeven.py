import argparse
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import meep as mp
import meep.mpb as mpb


N_POLY_SI   = 3.48
N_SIO2      = 1.44
T_SLAB_NM   = 211.0

RESOLUTION  = 32
NUM_BANDS   = 9
K_MIN       = 0.46
K_MAX       = 0.50
K_INTERP    = 45
PAD_Y       = 2.0
PAD_Z       = 1.5



BASE_PARAMS = dict(
    a_nm=496.0, h_spine=0.550,
    W_rib=0.484, h_rib=0.514,   # single rib size (both sides identical)
)


def build_geometry_ysym_71nm(params):
    """Y-symmetric fishbone: identical ribs on +y and -y, delta_s=0.
    For hsi = 71nm partial etch depth."""
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
    t_partial = 71.0 / a_nm
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
    """Run MPB with Y-EVEN symmetry for the 71nm partial etch structure."""
    import meep as mp
    import meep.mpb as mpb
    import numpy as np

    lattice, geometry, sy, sz, t_slab = build_geometry_ysym_71nm(params)
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
    print(f"3D MPB (Y-even Only): a={a:.0f}nm  t_partial=71nm  res={resolution}  bands={num_bands}")
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
        t_partial_nm=71.0,
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
    plt.title('Band Diagram: hsi = 71.0 nm (Y-even Symmetry)')
    plt.xlim(k_x[0], k_x[-1])
    plt.ylim(f_lo, f_hi)
    
    # Reduce clutter in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
    # plt.show() # prevent blocking in terminal

def get_fields_at_k(params, k_val, resolution, target_bands):
    import meep as mp
    import meep.mpb as mpb
    import numpy as np

    lattice, geometry, sy, sz, t_slab = build_geometry_ysym_71nm(params)
    k_points = [mp.Vector3(k_val)]
    
    max_band = max(target_bands) if target_bands else 1

    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=k_points,
        resolution=resolution,
        num_bands=max_band,
    )
    print(f"\nSolving modes at k={k_val} for field extraction (71nm)...")
    ms.run_yeven()
    
    efields = []
    all_freqs = np.array(ms.all_freqs)[0, :]
    freqs = []
    
    for b in target_bands:
        if b <= max_band:
            ms.get_efield(b, False)
            ef = np.array(ms.get_efield(b, False))
            efields.append(ef)
            freqs.append(all_freqs[b-1])
    
    return efields, freqs, sy, sz

def plot_modal_fields(efields, freqs, k_val, a_nm, sy, sz, target_bands, out_path):
    import matplotlib.pyplot as plt
    import numpy as np

    nb = len(efields)
    fig, axes = plt.subplots(nb, 3, figsize=(10, 2.5 * nb))
    if nb == 1:
        axes = axes[np.newaxis, :]

    comp_labels = ['|Ex|² (TE_y)', '|Ey|² (TE_x)', '|Ez|² (TM)']

    for ib in range(nb):
        ef = efields[ib]
        nx = ef.shape[0]
        x_mid = nx // 2
        band_index = target_bands[ib]

        for ic in range(3):
            ax = axes[ib, ic]
            field_slice = np.abs(ef[x_mid, :, :, ic])**2
            im = ax.imshow(field_slice.T, origin='lower', cmap='hot',
                           aspect='auto',
                           extent=[-sy/2, sy/2, -sz/2, sz/2])
            if ic == 0:
                f_val = freqs[ib]
                wl = a_nm / f_val if f_val > 0 else 0
                ax.set_ylabel(f'Band {band_index}\nf={f_val:.4f}\nwl={wl:.0f}nm',
                              fontsize=10, rotation=0, labelpad=60,
                              va='center')
            if ib == 0:
                ax.set_title(comp_labels[ic])
            if ib == nb - 1:
                ax.set_xlabel('y (a)')
            else:
                ax.set_xticklabels([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Mode Fields at k={k_val} (a={a_nm}nm, hsi=71nm)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=200)
    print(f"Saved mode fields plot to {out_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=32)
    parser.add_argument('--num-bands', type=int, default=8)
    parser.add_argument('--save-plots', action='store_true')
    parser.add_argument('--save-data', action='store_true')
    parser.add_argument('--plot-modes', action='store_true', help='Plot mode fields at specific k points')
    parser.add_argument('--target-bands', type=int, nargs='+', default=[7, 8], help='Target bands to plot/save for mode fields')
    parser.add_argument('--k-min', type=float, default=K_MIN)
    parser.add_argument('--k-max', type=float, default=K_MAX)
    parser.add_argument('--k-interp', type=int, default=K_INTERP)
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    
    if args.plot_modes:
        print(f"\n[Mode Plotting Mode] Targeting bands: {args.target_bands}")
        modes_data = {'target_bands': args.target_bands, 'a_nm': BASE_PARAMS['a_nm']}
        for k_val in [0.486, 0.495]:
            efields, freqs, sy, sz = get_fields_at_k(BASE_PARAMS, k_val, args.resolution, args.target_bands)
            
            # Save fields to variables
            modes_data['sy'] = sy
            modes_data['sz'] = sz
            modes_data[f'k_{k_val:.3f}_efields'] = efields
            modes_data[f'k_{k_val:.3f}_freqs'] = freqs
            
            plot_path = os.path.join(out_dir, f'yeven_71nm_modes_k{k_val:.3f}.png')
            plot_modal_fields(efields, freqs, k_val, BASE_PARAMS['a_nm'], sy, sz, args.target_bands, plot_path)
            
        if args.save_data:
            np.savez(os.path.join(out_dir, 'yeven_71nm_modes_data.npz'), **modes_data)
            print("Saved detailed mode data to yeven_71nm_modes_data.npz.")
            
        print("Mode plotting completed.")
        return  # Skip the full band run if only asking to plot modes
        
    data = run_mpb_yeven(
        BASE_PARAMS, 
        k_min=args.k_min,
        k_max=args.k_max,
        k_interp=args.k_interp,
        resolution=args.resolution, 
        num_bands=args.num_bands
    )
    
    if args.save_plots:
        plot_path = os.path.join(out_dir, 'yeven_71nm_bands.png')
        plot_bands(data, plot_path)
    else:
        plot_bands(data, 'yeven_71nm_bands.png')

    if args.save_data:
        data_path = os.path.join(out_dir, 'yeven_71nm_data.npz')
        np.savez(data_path, **data)
        print(f"Saved data to {data_path}")

if __name__ == '__main__':
    main()