import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def calc_ng(k_x, freqs):
    if len(k_x) < 2:
        return np.zeros_like(k_x)
    dk = np.gradient(k_x)
    df = np.gradient(freqs)
    vg = df / dk
    # ng = 1 / vg
    ng = np.zeros_like(vg)
    nonzero = np.abs(vg) > 1e-10
    ng[nonzero] = 1.0 / vg[nonzero]
    return ng

def run_script(script_name, num_bands):
    cmd = [
        sys.executable, script_name,
        "--save-data",
        "--save-plots",
        "--resolution", "32",
        "--num-bands", str(num_bands),
        "--k-min", "0.46",
        "--k-max", "0.50",
        "--k-interp", "48"  # mp.interpolate(48, [a, b]) gives 50 points
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_script_modes(script_name):
    cmd = [
        sys.executable, script_name,
        "--save-data",
        "--plot-modes",
        "--resolution", "32",
    ]
    print(f"Running mode extraction: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    script_dir = os.path.dirname(__file__)
    out_dir = os.path.join(script_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)

    data_71_path = os.path.join(out_dir, 'yeven_71nm_data.npz')
    data_140_path = os.path.join(out_dir, 'yeven_140nm_data.npz')

    data_71_modes_path = os.path.join(out_dir, 'yeven_71nm_modes_data.npz')
    data_140_modes_path = os.path.join(out_dir, 'yeven_140nm_modes_data.npz')

    # Run the simulations only if data is not already saved
    if not os.path.exists(data_71_path):
        run_script(os.path.join(script_dir, 'run_71nm_yeven.py'), 7)
    else:
        print(f"Found cached data: {data_71_path}, skipping simulation.")

    if not os.path.exists(data_140_path):
        run_script(os.path.join(script_dir, 'run_140nm_yeven.py'), 13)
    else:
        print(f"Found cached data: {data_140_path}, skipping simulation.")

    # Run mode extraction if not exists
    if not os.path.exists(data_71_modes_path):
        run_script_modes(os.path.join(script_dir, 'run_71nm_yeven.py'))
    else:
        print(f"Found cached modes data: {data_71_modes_path}, skipping mode extraction.")

    if not os.path.exists(data_140_modes_path):
        run_script_modes(os.path.join(script_dir, 'run_140nm_yeven.py'))
    else:
        print(f"Found cached modes data: {data_140_modes_path}, skipping mode extraction.")

    # Post process: Plot ng for specified bands
    plt.figure(figsize=(10, 5))

    # --- 71nm processing ---
    data_71 = np.load(data_71_path, allow_pickle=True)
    k_71 = data_71['k_x']
    freqs_71 = data_71['all_freqs']
    # 71nm requested: bands 6 and 7 (which correspond to indices 5 and 6)
    for b in [6, 7]:
        if b < freqs_71.shape[1]:
            f_b = freqs_71[:, b]
            ng_b = calc_ng(k_71, f_b)
            wavelengths = 496.0 / f_b  # a_nm = 496
            plt.plot(wavelengths, ng_b, label=f'71nm: Band {b}', lw=2)

    # --- 140nm processing ---
    data_140 = np.load(data_140_path, allow_pickle=True)
    k_140 = data_140['k_x']
    freqs_140 = data_140['all_freqs']
    # 140nm requested: bands 12 and 13 (which correspond to indices 11 and 12)
    for b in [12, 13]:
        if b < freqs_140.shape[1]:
            f_b = freqs_140[:, b]
            ng_b = calc_ng(k_140, f_b)
            wavelengths = 496.0 / f_b
            plt.plot(wavelengths, ng_b, '--', label=f'140nm: Band {b}', lw=2)

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Group Index $n_g$')
    plt.title('Group Index vs. Wavelength')
    plt.grid(True)
    plt.legend()
    # plt.ylim(0, 40)  # Removed ylim limitation as requested
    # plt.xlim(1500, 1600)  # You can keep or adjust xlim based on your interest, but ylim is removed
    
    out_plot = os.path.join(out_dir, 'ng_comparison.png')
    plt.tight_layout()
    plt.savefig(out_plot, dpi=200)
    plt.show()
    print(f"Saved ng comparison plot to {out_plot}")
    plt.close()

    def plot_cached_mode_fields(npz_path, name, k_values, eps):
        data = np.load(npz_path, allow_pickle=True)
        sy = float(data['sy'])
        sz = float(data['sz'])
        a_nm = float(data['a_nm'])
        target_bands = data['target_bands']
        
        t_slab = 220.0 / a_nm
        if name == "71nm":
            t_partial = 71.0 / a_nm
        else:
            t_partial = 140.0 / a_nm
            
        z_val_partial = -t_slab/2.0 + t_partial/2.0
        
        # Get contour of the grating ribs from near the top of the slab
        # Since the continuous partial etch layer may cover the entire plane at z=0 and z_partial,
        # the contour structure from the top of the slab perfectly portrays the rib/spine boundaries.
        nz_total = eps.shape[2]
        z_rib_idx = int(round(((0.9 * t_slab / 2.0) + sz/2.0) / sz * nz_total))
        eps_rib_structure = eps[:, :, z_rib_idx]
        
        for k_val in k_values:
            efields = data[f'k_{k_val:.3f}_efields']
            freqs = data[f'k_{k_val:.3f}_freqs']
            nb = len(efields)
            fig, axes = plt.subplots(nb, 6, figsize=(18, 2.5 * nb))
            if nb == 1:
                axes = axes[np.newaxis, :]

            for ib in range(nb):
                ef = efields[ib]
                nx, ny, nz = ef.shape[:3]
                band_index = target_bands[ib]
                
                z_0_idx = nz // 2
                z_p_idx = int(round((z_val_partial + sz/2.0) / sz * nz))

                extent_xy = [-0.5, 0.5, -sy/2, sy/2]

                for cut_idx, (z_idx, label_suffix) in enumerate([
                    (z_0_idx, "z=0"), 
                    (z_p_idx, "z_partial")
                ]):
                    for ic in range(3):
                        ax_idx = cut_idx * 3 + ic
                        ax = axes[ib, ax_idx]
                        field_slice = np.abs(ef[:, :, z_idx, ic])**2
                        im = ax.imshow(field_slice.T, origin='lower', cmap='hot',
                                       aspect='auto', extent=extent_xy)
                        
                        # Overlay the grating ribs boundary
                        ax.contour(eps_rib_structure.T, levels=[5.0], colors='w', linewidths=0.8, alpha=0.6,
                                   extent=extent_xy)
                        
                        if ax_idx == 0:
                            f_val = freqs[ib]
                            wl = a_nm / f_val if f_val > 0 else 0
                            ax.set_ylabel(f'Band {band_index}\nf={f_val:.4f}\nwl={wl:.0f}nm',
                                          fontsize=10, rotation=0, labelpad=60, va='center')
                        if ib == 0:
                            comp = ['Ex', 'Ey', 'Ez'][ic]
                            ax.set_title(f'|{comp}|² ({label_suffix})')
                        if ib == nb - 1:
                            ax.set_xlabel('x (a)')
                        else:
                            ax.set_xticklabels([])
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.suptitle(f"Mode Fields at k={k_val} ({name}) XY Planes", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            out_mode_plot = os.path.join(out_dir, f'mode_comparison_{name}_k{k_val:.3f}_xy.png')
            plt.savefig(out_mode_plot, dpi=200)
            print(f"Saved mode fields comparison to {out_mode_plot}")
            plt.close(fig)

    print("\n--- Generating Mode Field Plots ---")
    plot_cached_mode_fields(data_71_modes_path, "71nm", [0.486, 0.495], data_71['epsilon'])
    plot_cached_mode_fields(data_140_modes_path, "140nm", [0.486, 0.495], data_140['epsilon'])

if __name__ == '__main__':
    main()