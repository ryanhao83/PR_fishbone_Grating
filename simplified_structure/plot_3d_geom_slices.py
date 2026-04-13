"""
plot_3d_geom_slices.py — Visually confirm the 3D geometry and the new partial etch layer
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import meep as mp
import meep.mpb as mpb

from simplified_gc_3d import build_geometry_3d, get_structures

def main():
    os.makedirs('output', exist_ok=True)

    # Use the first parameter set
    params = get_structures()[0]
    # Remove label string
    params = {k: v for k, v in params.items() if k != 'label'}
    
    print(f"Building geometry with a={params['a_nm']} nm...")
    lattice, geometry, sy, sz, t_slab = build_geometry_3d(params)

    # Initialize ModeSolver to discretize and get epsilon
    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        resolution=32
    )

    print("Initializing MPB parameters and resolving epsilon array...")
    ms.init_params(mp.NO_PARITY, True)
    eps = ms.get_epsilon()

    nx, ny, nz = eps.shape
    print(f"Geometry grid size: {nx} x {ny} x {nz}")
    a = params['a_nm']

    # We want axes in units of 'a'
    x_extent = [-0.5, 0.5]
    y_extent = [-sy/2, sy/2]
    z_extent = [-sz/2, sz/2]

    # Subplots for XY, XZ, YZ slices
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Calculate precise z indices representing slab regions
    # The computational cell spans sz along Z, centered at z=0.
    # z coordinates go from -sz/2 to sz/2
    # So physical z -> index: int((z / sz + 0.5) * nz)
    z_bottom_etch = -t_slab/2 + (71.0/a)*0.5  # middle of 71nm etch layer
    z_top_rib = t_slab/2 - 0.01               # top rib layer

    idx_z_bottom = int((z_bottom_etch / sz + 0.5) * nz)
    idx_z_mid = int((0.0 / sz + 0.5) * nz)
    idx_z_top = int((z_top_rib / sz + 0.7) * nz)

    # 1. XY slice inside the 71nm partial etch layer
    axes[0, 0].imshow(eps[:, :, idx_z_bottom].T, origin='lower', cmap='gray', aspect='equal', extent=x_extent + y_extent)
    axes[0, 0].set_title(f'XY Slice (Inside 71nm etch, z={z_bottom_etch:.2f}a)')
    axes[0, 0].set_xlabel('x (a)')
    axes[0, 0].set_ylabel('y (a)')

    # 2. XY slice at center of slab
    axes[0, 1].imshow(eps[:, :, idx_z_mid].T, origin='lower', cmap='gray', aspect='equal', extent=x_extent + y_extent)
    axes[0, 1].set_title(f'XY Slice (Center of slab, z=0)')
    axes[0, 1].set_xlabel('x (a)')
    axes[0, 1].set_ylabel('y (a)')

    # 3. XY slice near the top (ribs)
    axes[0, 2].imshow(eps[:, :, idx_z_top].T, origin='lower', cmap='gray', aspect='equal', extent=x_extent + y_extent)
    axes[0, 2].set_title(f'XY Slice (Near top, z={z_top_rib:.2f}a)')
    axes[0, 2].set_xlabel('x (a)')
    axes[0, 2].set_ylabel('y (a)')

    # 4. XZ slice along the center spine (y = 0)
    idx_y_mid = int(ny / 2)
    axes[1, 0].imshow(eps[:, idx_y_mid, :].T, origin='lower', cmap='gray', aspect='equal', extent=x_extent + z_extent)
    axes[1, 0].set_title('XZ Slice along center spine (y=0)')
    axes[1, 0].set_xlabel('x (a)')
    axes[1, 0].set_ylabel('z (a)')
    axes[1, 0].axhline(-t_slab/2 + 71.0/a, color='red', linestyle='--', lw=1, label='Top of 71nm Etch Layer')
    axes[1, 0].axhline(-t_slab/2, color='blue', linestyle='--', lw=1, label='Bottom of 211nm Slab')
    axes[1, 0].legend(loc='lower right', fontsize=8)

    # 5. YZ slice across the transverse direction (x = 0)
    idx_x_mid = int(nx / 2)
    axes[1, 1].imshow(eps[idx_x_mid, :, :].T, origin='lower', cmap='gray', aspect='equal', extent=y_extent + z_extent)
    axes[1, 1].set_title('YZ Slice across center (x=0)')
    axes[1, 1].set_xlabel('y (a)')
    axes[1, 1].set_ylabel('z (a)')
    axes[1, 1].axhline(-t_slab/2 + 71.0/a, color='red', linestyle='--', lw=1)
    axes[1, 1].axhline(-t_slab/2, color='blue', linestyle='--', lw=1)
    
    # 6. Hide the empty 6th subplot
    axes[1, 2].axis('off')

    plt.suptitle("3D Fishbone Grating with 71nm Partial Etch Layer", fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = 'output/geom_3d_slices.png'
    plt.savefig(save_path, dpi=200)
    print(f"Geometry successfully saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    main()
