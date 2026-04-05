"""
run_sweep_simp_3d_gc.py — Custom parameter sweep for the 3D simplified fishbone GC model.
"""

import os
import numpy as np
import csv
from datetime import datetime

# Import core simulation and utility functions from the existing 3D script
from simplified_gc_3d import run_mpb_3d, cache_path, save_results, load_results, analyze_3d
from simplified_gc_3d import RESOLUTION, NUM_BANDS, K_MIN, K_MAX, K_INTERP

def main():
    # ---------------------------------------------------------
    # 1. Base parameters (e.g., from the best 2D results)
    # ---------------------------------------------------------
    base_params = dict(
        a_nm=380.0, 
        h_spine=0.550,
        Wt=0.484, ht=0.513, 
        Wb=0.568, hb=0.734,
        delta_s=0.0
    )

    # ---------------------------------------------------------
    # 2. Sweep Configuration
    # ---------------------------------------------------------
    sweep_param = 'hb'  # Easy to change to 'delta_s', 'ht', 'Wt', etc.
    
    # Sweep from 0.70 to 0.76 with an increment of 0.02
    # (Note: Assumed 0.02 instead of 0.2, as 0.2 is larger than the 0.7~0.76 range)
    step = 0.02
    sweep_values = np.arange(0.70, 0.76 + step/2, step) 

    print(f"{'='*60}")
    print(f"3D Sweep Configuration")
    print(f"  Sweeping parameter : {sweep_param}")
    print(f"  Sweep values       : {sweep_values}")
    print(f"  Base parameters    : {base_params}")
    print(f"{'='*60}\n")

    all_results = []

    # ---------------------------------------------------------
    # 3. Running the Sweep Loop
    # ---------------------------------------------------------
    for val in sweep_values:
        # Copy base params and update the swept variable
        params = dict(base_params)
        params[sweep_param] = float(val)

        print(f"\n--- Current Run: {sweep_param} = {val:.4f} ---")
        
        # Use the imported hashing logic to find/create the cache path
        cpath = cache_path(params, RESOLUTION, NUM_BANDS)
        
        if os.path.exists(cpath):
            print(f"  [Cache detected] Loading results from {cpath}...")
            data = load_results(cpath)
        else:
            print(f"  [No cache] Running 3D MPB simulation (res={RESOLUTION}, bands={NUM_BANDS})...")
            # Invoke the external runner
            data = run_mpb_3d(params, k_min=K_MIN, k_max=K_MAX, k_interp=K_INTERP,
                              num_bands=NUM_BANDS, resolution=RESOLUTION)
            # Save results to cache
            save_results(data, cpath)

        # Analyze the data (group index, bandwidth, single-mode verification)
        ana = analyze_3d(data)
        
        print(f"  -> Result ({sweep_param}={val:.3f}): ")
        print(f"     Bandwidth:       {ana['bw']:.1f} nm")
        print(f"     Mean n_g:        {ana['ng_mean']:.2f}")
        print(f"     Center λ:        {ana['wl_center']:.1f} nm")
        print(f"     TE Single-mode:  {ana['single_mode_te']}")
        
        all_results.append({
            'sweep_param': sweep_param,
            'val': val,
            'bw': ana['bw'],
            'ng': ana['ng_mean'],
            'wl': ana['wl_center'],
            'sm': ana['single_mode_te'],
            'cache_file': os.path.relpath(cpath)
        })

    # ---------------------------------------------------------
    # 4. Save results to local CSV file
    # ---------------------------------------------------------
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(output_dir, f"sweep_3d_{sweep_param}_{timestamp}.csv")
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sweep_Param', 'Value', 'Bandwidth(nm)', 'Mean_ng', 'Center_wl(nm)', 'Single_Mode', 'Cache_File'])
        for r in all_results:
            writer.writerow([r['sweep_param'], r['val'], r['bw'], r['ng'], r['wl'], r['sm'], r['cache_file']])

    print(f"\n[Saved] Sweep results successfully exported to: {csv_filename}")

    # ---------------------------------------------------------
    # 5. Summary Table
    # ---------------------------------------------------------
    print(f"\n\n{'='*70}")
    print(f"SWEEP SUMMARY: {sweep_param}")
    print(f"{'='*70}")
    print(f"{sweep_param:>10} | {'BW (nm)':>10} | {'Mean ng':>10} | {'Center wl':>12} | {'TE Single-mode'}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['val']:10.4f} | {r['bw']:10.1f} | {r['ng']:10.2f} | {r['wl']:12.1f} | {str(r['sm']):>14}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
