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

def main():
    script_dir = os.path.dirname(__file__)
    out_dir = os.path.join(script_dir, 'output')

    # Run the simulations
    run_script(os.path.join(script_dir, 'run_71nm_yeven.py'), 7)
    run_script(os.path.join(script_dir, 'run_140nm_yeven.py'), 13)

    # Post process: Plot ng for specified bands
    plt.figure(figsize=(10, 5))

    # --- 71nm processing ---
    data_71 = np.load(os.path.join(out_dir, 'yeven_71nm_data.npz'), allow_pickle=True)
    k_71 = data_71['k_x']
    freqs_71 = data_71['all_freqs']
    # 71nm requested: bands 6 and 7 (which correspond to indices 5 and 6)
    for b in [5, 6]:
        if b < freqs_71.shape[1]:
            f_b = freqs_71[:, b]
            ng_b = calc_ng(k_71, f_b)
            wavelengths = 496.0 / f_b  # a_nm = 496
            plt.plot(wavelengths, ng_b, label=f'71nm: Band {b+1}', lw=2)

    # --- 140nm processing ---
    data_140 = np.load(os.path.join(out_dir, 'yeven_140nm_data.npz'), allow_pickle=True)
    k_140 = data_140['k_x']
    freqs_140 = data_140['all_freqs']
    # 140nm requested: bands 12 and 13 (which correspond to indices 11 and 12)
    for b in [11, 12]:
        if b < freqs_140.shape[1]:
            f_b = freqs_140[:, b]
            ng_b = calc_ng(k_140, f_b)
            wavelengths = 496.0 / f_b
            plt.plot(wavelengths, ng_b, '--', label=f'140nm: Band {b+1}', lw=2)

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Group Index $n_g$')
    plt.title('Group Index vs. Wavelength')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 40)
    plt.xlim(1500, 1600)  # Common telecom C-band limit
    
    out_plot = os.path.join(out_dir, 'ng_comparison.png')
    plt.tight_layout()
    plt.savefig(out_plot, dpi=200)
    print(f"Saved ng comparison plot to {out_plot}")

if __name__ == '__main__':
    main()