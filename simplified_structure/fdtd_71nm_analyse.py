"""
Post-processing of 3D FDTD simulation data for 71nm fishbone grating.
Loads fdtd_71nm_data.npz produced by fdtd_71nm_sim.py and generates all plots.

Usage:
    python fdtd_71nm_analyse.py
    python fdtd_71nm_analyse.py --data output/fdtd_71nm_data.npz --no-show
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import hilbert


def load_data(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def plot_geometry(data, out_dir, show):
    """Top-view and side-view of dielectric structure."""
    eps_xy = data['eps_xy']
    eps_xz = data['eps_xz']
    a_nm = float(data['a_nm'])
    sx, sy, sz = float(data['sx']), float(data['sy']), float(data['sz'])
    snap_region_x = float(data['snap_region_x'])
    snap_region_y = float(data['snap_region_y'])
    src_x  = float(data['src_x'])
    mon1_x = float(data['mon1_x'])
    mon2_x = float(data['mon2_x'])
    grating_start = float(data['grating_start'])
    grating_end   = float(data['grating_end'])
    PML_THICK = 1.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 5))

    ext_xy = np.array([-snap_region_x/2, snap_region_x/2,
                        -snap_region_y/2, snap_region_y/2]) * a_nm / 1000
    ax1.imshow(eps_xy.T, origin='lower', cmap='binary', extent=ext_xy, aspect='auto')
    for x_val, c, lbl in [(src_x, 'green', 'Source'),
                           (mon1_x, 'blue', 'Mon1'), (mon2_x, 'red', 'Mon2'),
                           (grating_start, 'orange', 'Grating start'),
                           (grating_end, 'purple', 'Grating end / PML')]:
        ax1.axvline(x_val * a_nm / 1000, color=c, ls='--', lw=1.5, label=lbl)
    ax1.set_xlabel('x (µm)'); ax1.set_ylabel('y (µm)')
    ax1.set_title('Top view (z=0)'); ax1.legend(fontsize=8, loc='upper right')

    ext_xz = np.array([-snap_region_x/2, snap_region_x/2,
                        -(sz - 2*PML_THICK)/2, (sz - 2*PML_THICK)/2]) * a_nm / 1000
    ax2.imshow(eps_xz.T, origin='lower', cmap='binary', extent=ext_xz, aspect='auto')
    ax2.set_xlabel('x (µm)'); ax2.set_ylabel('z (µm)')
    ax2.set_title('Side view (y=0)')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fdtd_71nm_geometry.png'), dpi=150)
    if show: plt.show()
    plt.close()


def plot_time_signals(data, out_dir, show):
    """Raw Ey(t) at both monitors."""
    t = data['t_record']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(t, data['ey_t_in'], 'b-', lw=0.8)
    ax1.set_ylabel('Ey (input monitor)')
    ax1.set_title('Time-domain signals at monitor points')
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, data['ey_t_out'], 'r-', lw=0.8)
    ax2.set_ylabel('Ey (output monitor)')
    ax2.set_xlabel('Time (Meep units)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fdtd_71nm_time_signals.png'), dpi=150)
    if show: plt.show()
    plt.close()


def compute_ng(data, out_dir, show):
    """Compute ng via broadband peak delay and bandpass filtering."""
    t       = data['t_record']
    ey_in   = data['ey_t_in']
    ey_out  = data['ey_t_out']
    a_nm    = float(data['a_nm'])
    L_mon   = float(data['L_mon'])
    LAMBDA_MIN = float(data['LAMBDA_MIN_NM'])
    LAMBDA_MAX = float(data['LAMBDA_MAX_NM'])

    # ── Method 1: broadband peak delay ────────────────────────────────
    env_in  = np.abs(hilbert(ey_in))
    env_out = np.abs(hilbert(ey_out))

    t_peak_in  = t[np.argmax(env_in)]
    t_peak_out = t[np.argmax(env_out)]
    delta_t = t_peak_out - t_peak_in
    ng_broadband = delta_t / L_mon

    print(f"Peak time at Mon1 (input):  t1 = {t_peak_in:.2f}")
    print(f"Peak time at Mon2 (output): t2 = {t_peak_out:.2f}")
    print(f"Δt = {delta_t:.2f}  (Meep units = a/c)")
    print(f"L_mon = {L_mon:.1f} a")
    print(f"Δt in ps = {delta_t * a_nm / 3e5:.3f} ps")
    print(f"\n>>> Broadband ng = {ng_broadband:.2f} <<<")

    # ── Method 2: ng(λ) via bandpass filtering ────────────────────────
    dt_samp = t[1] - t[0]
    N = len(t)
    fft_freqs  = np.fft.rfftfreq(N, d=dt_samp)
    Ey_in_fft  = np.fft.rfft(ey_in)
    Ey_out_fft = np.fft.rfft(ey_out)

    wl_targets = np.linspace(LAMBDA_MIN, LAMBDA_MAX, 50)
    bw_nm = 5.0  # Gaussian filter FWHM [nm]

    ng_vs_wl = []
    for wl_c in wl_targets:
        f_c = a_nm / wl_c
        sigma_f = a_nm / wl_c**2 * bw_nm / (2 * np.sqrt(2 * np.log(2)))
        gauss = np.exp(-0.5 * ((fft_freqs - f_c) / sigma_f)**2)

        sig_in_filt  = np.fft.irfft(Ey_in_fft  * gauss, n=N)
        sig_out_filt = np.fft.irfft(Ey_out_fft * gauss, n=N)

        env_in_f  = np.abs(hilbert(sig_in_filt))
        env_out_f = np.abs(hilbert(sig_out_filt))

        t1 = t[np.argmax(env_in_f)]
        t2 = t[np.argmax(env_out_f)]
        ng_vs_wl.append((t2 - t1) / L_mon)

    ng_vs_wl = np.array(ng_vs_wl)

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.plot(t, ey_in,  'b-', lw=0.5, alpha=0.4, label='Mon1 raw')
    ax.plot(t, ey_out, 'r-', lw=0.5, alpha=0.4, label='Mon2 raw')
    ax.plot(t, env_in,  'b-', lw=1.5, label='Mon1 envelope')
    ax.plot(t, env_out, 'r-', lw=1.5, label='Mon2 envelope')
    ax.axvline(t_peak_in,  color='blue', ls='--', lw=1)
    ax.axvline(t_peak_out, color='red',  ls='--', lw=1)
    ax.annotate(f'Δt = {delta_t:.1f}',
                xy=((t_peak_in + t_peak_out) / 2, max(env_in) * 0.8),
                ha='center', fontsize=12, color='green',
                arrowprops=dict(arrowstyle='<->', color='green'),
                xytext=((t_peak_in + t_peak_out) / 2, max(env_in) * 0.95))
    ax.set_xlabel('Time (Meep units)')
    ax.set_ylabel('Ey')
    ax.set_title(f'Pulse arrival: Δt = {delta_t:.2f} → ng(broadband) = {ng_broadband:.2f}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(wl_targets, ng_vs_wl, 'r-o', ms=3, lw=1.5, label='FDTD ng(λ) [bandpass]')
    ax.axhline(ng_broadband, color='green', ls='--', lw=1, label=f'Broadband ng = {ng_broadband:.2f}')
    ax.axhline(4.5, color='gray', ls=':', lw=1, label='MPB ng ≈ 4.5')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Group Index $n_g$')
    ax.set_title('FDTD Group Index vs Wavelength (bandpass peak-delay method)')
    ax.set_xlim(LAMBDA_MIN - 5, LAMBDA_MAX + 5)
    ax.set_ylim(0, 12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fdtd_71nm_ng.png'), dpi=150)
    if show: plt.show()
    plt.close()

    mask = (ng_vs_wl > 1) & (ng_vs_wl < 20)
    if np.any(mask):
        print(f"\n{'='*60}")
        print(f"ng(λ) in {LAMBDA_MIN:.0f}–{LAMBDA_MAX:.0f} nm (filtered):")
        print(f"  Mean ng = {np.mean(ng_vs_wl[mask]):.2f}")
        print(f"  Std  ng = {np.std(ng_vs_wl[mask]):.2f}")
        print(f"  Range: {np.min(ng_vs_wl[mask]):.2f} – {np.max(ng_vs_wl[mask]):.2f}")
        print(f"{'='*60}")

    return ng_broadband, wl_targets, ng_vs_wl


def compute_ng_phase(data, out_dir, show, ref_data=None):
    """Method 3: ng(λ) via spectral phase derivative dφ/dω.

    H(ω) = FFT[Ey_out] / FFT[Ey_in]  → φ(ω) = unwrap(angle(H))
    ng(λ) = -(c/L) · dφ/dω   (in Meep units: ng = -(1/L_mon) · dφ/dω)

    If ref_data is provided (strip-only reference), normalise:
        H_norm = H_grating / H_ref
    to remove the strip waveguide contribution.
    """
    t       = data['t_record']
    ey_in   = data['ey_t_in']
    ey_out  = data['ey_t_out']
    a_nm    = float(data['a_nm'])
    L_mon   = float(data['L_mon'])
    LAMBDA_MIN = float(data['LAMBDA_MIN_NM'])
    LAMBDA_MAX = float(data['LAMBDA_MAX_NM'])

    dt = t[1] - t[0]
    N  = len(t)

    # FFT (positive frequencies only)
    fft_freqs  = np.fft.rfftfreq(N, d=dt)   # in units of c/a
    Ey_in_fft  = np.fft.rfft(ey_in)
    Ey_out_fft = np.fft.rfft(ey_out)

    # Transfer function H(ω) = Out / In
    eps = np.max(np.abs(Ey_in_fft)) * 1e-10
    H = Ey_out_fft / np.where(np.abs(Ey_in_fft) > eps, Ey_in_fft, eps)

    # Normalise by reference if available
    label_suffix = ''
    if ref_data is not None:
        t_ref     = ref_data['t_record']
        ey_in_r   = ref_data['ey_t_in']
        ey_out_r  = ref_data['ey_t_out']
        dt_ref    = t_ref[1] - t_ref[0]
        N_ref     = len(t_ref)
        fft_freqs_ref = np.fft.rfftfreq(N_ref, d=dt_ref)
        In_ref  = np.fft.rfft(ey_in_r)
        Out_ref = np.fft.rfft(ey_out_r)
        eps_ref = np.max(np.abs(In_ref)) * 1e-10
        H_ref = Out_ref / np.where(np.abs(In_ref) > eps_ref, In_ref, eps_ref)

        # Interpolate H_ref onto grating frequency grid if lengths differ
        if len(fft_freqs_ref) != len(fft_freqs) or not np.allclose(fft_freqs_ref, fft_freqs):
            from scipy.interpolate import interp1d
            H_ref_amp   = interp1d(fft_freqs_ref, np.abs(H_ref), fill_value='extrapolate')(fft_freqs)
            H_ref_phase = interp1d(fft_freqs_ref, np.unwrap(np.angle(H_ref)), fill_value='extrapolate')(fft_freqs)
            H_ref_interp = H_ref_amp * np.exp(1j * H_ref_phase)
        else:
            H_ref_interp = H_ref

        eps_href = np.max(np.abs(H_ref_interp)) * 1e-10
        H = H / np.where(np.abs(H_ref_interp) > eps_href, H_ref_interp, eps_href)
        label_suffix = ' (ref-normalised)'
        print("Reference normalisation applied.")

    # Phase and unwrap
    phase = np.unwrap(np.angle(H))

    # Angular frequency ω = 2πf  (f in units of c/a → ω in units of 2πc/a)
    omega = 2.0 * np.pi * fft_freqs  # units: 2πc/a

    # ng = -(1/L_mon) · dφ/dω
    d_omega = np.gradient(omega)
    d_phase = np.gradient(phase)
    ng_raw  = -(1.0 / L_mon) * d_phase / np.where(np.abs(d_omega) > 1e-30, d_omega, 1e-30)

    # Convert frequency to wavelength: λ = a_nm / f
    wl_all = np.where(fft_freqs > 0, a_nm / fft_freqs, np.inf)

    # Restrict to wavelength window of interest
    wl_mask = (wl_all >= LAMBDA_MIN - 10) & (wl_all <= LAMBDA_MAX + 10)
    wl_phase = wl_all[wl_mask]
    ng_phase = ng_raw[wl_mask]

    # Sort by wavelength for clean plotting
    sort_idx = np.argsort(wl_phase)
    wl_phase = wl_phase[sort_idx]
    ng_phase = ng_phase[sort_idx]

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Phase plot
    ax = axes[0]
    wl_for_phase = np.where(fft_freqs > 0, a_nm / fft_freqs, np.inf)
    phase_mask = (wl_for_phase >= LAMBDA_MIN - 10) & (wl_for_phase <= LAMBDA_MAX + 10)
    wl_p = wl_for_phase[phase_mask]
    ph_p = phase[phase_mask]
    sort_p = np.argsort(wl_p)
    ax.plot(wl_p[sort_p], ph_p[sort_p], 'b-', lw=1.0)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Phase φ(ω) (rad)')
    ax.set_title(f'Unwrapped spectral phase of H(ω){label_suffix}')
    ax.grid(True, alpha=0.3)

    # ng plot
    ax = axes[1]
    ax.plot(wl_phase, ng_phase, 'r-', lw=1.5, label=f'Phase method{label_suffix}')
    ax.axhline(4.5, color='gray', ls=':', lw=1, label='MPB ng ≈ 4.5')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Group Index $n_g$')
    ax.set_title(f'ng(λ) from spectral phase derivative{label_suffix}')
    ax.set_xlim(LAMBDA_MIN - 5, LAMBDA_MAX + 5)
    ax.set_ylim(0, 12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = '_refnorm' if ref_data is not None else ''
    plt.savefig(os.path.join(out_dir, f'fdtd_71nm_ng_phase{suffix}.png'), dpi=150)
    if show: plt.show()
    plt.close()

    # Summary
    bw_mask = (wl_phase >= LAMBDA_MIN) & (wl_phase <= LAMBDA_MAX) & (ng_phase > 1) & (ng_phase < 20)
    if np.any(bw_mask):
        ng_sel = ng_phase[bw_mask]
        print(f"\nng(λ) from phase method{label_suffix} [{LAMBDA_MIN:.0f}–{LAMBDA_MAX:.0f} nm]:")
        print(f"  Mean ng = {np.mean(ng_sel):.2f}")
        print(f"  Std  ng = {np.std(ng_sel):.2f}")
        print(f"  Range: {np.min(ng_sel):.2f} – {np.max(ng_sel):.2f}")

    return wl_phase, ng_phase


def plot_transmission(data, out_dir, show):
    """Transmission spectrum from flux monitors."""
    a_nm = float(data['a_nm'])
    flux_freqs = data['flux_freqs']
    P_in  = data['P_in']
    P_out = data['P_out']
    LAMBDA_MIN = float(data['LAMBDA_MIN_NM'])
    LAMBDA_MAX = float(data['LAMBDA_MAX_NM'])

    wl_flux = a_nm / flux_freqs
    T_flux  = P_out / np.where(np.abs(P_in) > 1e-30, P_in, 1e-30)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(wl_flux, P_in, 'b-', lw=1.0, label='Input flux')
    ax1.plot(wl_flux, P_out, 'r-', lw=1.0, label='Output flux')
    ax1.set_ylabel('Flux (a.u.)')
    ax1.set_title('Flux spectra at input and output monitors')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(wl_flux, T_flux, 'k-', lw=1.0)
    ax2.set_ylabel('Transmission T = P_out / P_in')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_title('Transmission spectrum through grating')
    ax2.set_xlim(LAMBDA_MIN - 30, LAMBDA_MAX + 30)
    ax2.set_ylim(-0.1, 1.2)
    ax2.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fdtd_71nm_transmission.png'), dpi=150)
    if show: plt.show()
    plt.close()


def compare_mpb(data, ng_broadband, wl_targets, ng_vs_wl, out_dir, show):
    """Overlay FDTD ng with MPB band-7 ng."""
    a_nm = float(data['a_nm'])
    LAMBDA_MIN = float(data['LAMBDA_MIN_NM'])
    LAMBDA_MAX = float(data['LAMBDA_MAX_NM'])

    mpb_path = os.path.join(out_dir, 'yeven_71nm_data.npz')
    try:
        mpb_data = np.load(mpb_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"MPB data not found at {mpb_path} — skipping comparison.")
        return

    k_mpb = mpb_data['k_x']
    freqs_mpb = mpb_data['all_freqs']
    a_mpb = float(mpb_data['params'].item()['a_nm'])

    band_idx = 7
    f_b7 = freqs_mpb[:, band_idx - 1]

    dk = np.gradient(k_mpb)
    df_mpb = np.gradient(f_b7)
    vg = df_mpb / dk
    ng_mpb = np.zeros_like(vg)
    nz = np.abs(vg) > 1e-10
    ng_mpb[nz] = 1.0 / vg[nz]
    wl_mpb = a_mpb / f_b7

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wl_mpb, ng_mpb, 'b-o', ms=3, lw=2.0, label=f'MPB Band {band_idx} (yeven)')
    ax.plot(wl_targets, ng_vs_wl, 'r-s', ms=3, lw=1.5, alpha=0.8, label='FDTD ng(λ) [bandpass]')
    ax.axhline(ng_broadband, color='green', ls='--', lw=1.5,
               label=f'FDTD broadband ng = {ng_broadband:.2f}')
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Group Index $n_g$', fontsize=12)
    ax.set_title('MPB vs FDTD Group Index Comparison (71nm partial etch)', fontsize=13)
    ax.set_xlim(LAMBDA_MIN - 15, LAMBDA_MAX + 15)
    ax.set_ylim(0, 10)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fdtd_vs_mpb_ng_71nm.png'), dpi=150)
    if show: plt.show()
    plt.close()


def make_movie(data, out_dir, show):
    """Animate Ey snapshots in z=0 plane."""
    N_SI = 3.48
    a_nm = float(data['a_nm'])
    ey_stack = data['ey_snapshots']
    snap_times = data['snap_times']
    eps_xy = data['eps_xy']
    snap_region_x = float(data['snap_region_x'])
    snap_region_y = float(data['snap_region_y'])
    mon1_x = float(data['mon1_x'])
    mon2_x = float(data['mon2_x'])

    vmax = np.percentile(np.abs(ey_stack), 99.5)
    ext_snap = np.array([-snap_region_x/2, snap_region_x/2,
                          -snap_region_y/2, snap_region_y/2]) * a_nm / 1000

    fig, ax = plt.subplots(figsize=(16, 3))
    im = ax.imshow(ey_stack[0].T, origin='lower', cmap='RdBu_r',
                   extent=ext_snap, aspect='auto', vmin=-vmax, vmax=vmax)
    ax.contour(eps_xy.T, levels=[N_SI**2 * 0.5], colors='k', linewidths=0.3,
               alpha=0.4, extent=ext_snap)
    ax.axvline(mon1_x * a_nm / 1000, color='blue', ls='--', lw=1, alpha=0.6)
    ax.axvline(mon2_x * a_nm / 1000, color='red',  ls='--', lw=1, alpha=0.6)
    ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)')
    title = ax.set_title(f'Ey(z=0)  t = {snap_times[0]:.0f}')
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    def update(frame):
        im.set_data(ey_stack[frame].T)
        title.set_text(f'Ey(z=0)  t = {snap_times[frame]:.0f}')
        return [im, title]

    ani = animation.FuncAnimation(fig, update, frames=len(ey_stack),
                                   interval=80, blit=True)
    plt.tight_layout()

    try:
        writer = animation.FFMpegWriter(fps=15, bitrate=2000)
        ani_path = os.path.join(out_dir, 'fdtd_71nm_ey_movie.mp4')
        ani.save(ani_path, writer=writer)
        print(f"Saved movie to {ani_path}")
    except Exception:
        ani_path = os.path.join(out_dir, 'fdtd_71nm_ey_movie.gif')
        ani.save(ani_path, writer='pillow', fps=15)
        print(f"Saved movie to {ani_path}")

    if show: plt.show()
    plt.close()


def plot_snapshots(data, out_dir, show):
    """Selected Ey field snapshots."""
    N_SI = 3.48
    a_nm = float(data['a_nm'])
    ey_stack = data['ey_snapshots']
    snap_times = data['snap_times']
    eps_xy = data['eps_xy']
    snap_region_x = float(data['snap_region_x'])
    snap_region_y = float(data['snap_region_y'])

    vmax = np.percentile(np.abs(ey_stack), 99.5)
    ext_snap = np.array([-snap_region_x/2, snap_region_x/2,
                          -snap_region_y/2, snap_region_y/2]) * a_nm / 1000

    n_show = min(8, len(ey_stack))
    indices = np.linspace(0, len(ey_stack) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(n_show, 1, figsize=(16, 2.2 * n_show))
    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.imshow(ey_stack[idx].T, origin='lower', cmap='RdBu_r',
                  extent=ext_snap, aspect='auto', vmin=-vmax, vmax=vmax)
        ax.contour(eps_xy.T, levels=[N_SI**2 * 0.5], colors='k', linewidths=0.3,
                   alpha=0.3, extent=ext_snap)
        ax.set_ylabel(f't={snap_times[idx]:.0f}', fontsize=10)
        if i < n_show - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel('x (µm)')

    plt.suptitle('Ey field snapshots (z=0 plane)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fdtd_71nm_snapshots.png'), dpi=150)
    if show: plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyse 71nm FDTD grating simulation')
    parser.add_argument('--data', type=str, default='output/fdtd_71nm_data.npz')
    parser.add_argument('--ref-data', type=str, default=None,
                        help='Reference data (strip-only) for phase normalisation, e.g. output/fdtd_71nm_data_ref.npz')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots (save only)')
    parser.add_argument('--skip-movie', action='store_true', help='Skip movie generation')
    args = parser.parse_args()

    show = not args.no_show
    out_dir = os.path.dirname(args.data) or '.'

    print(f"Loading data from {args.data} ...")
    data = load_data(args.data)

    ref_data = None
    if args.ref_data:
        print(f"Loading reference data from {args.ref_data} ...")
        ref_data = load_data(args.ref_data)

    print("\n── 1. Geometry ──")
    plot_geometry(data, out_dir, show)

    print("\n── 2. Time-domain signals ──")
    plot_time_signals(data, out_dir, show)

    print("\n── 3. Group index (peak delay) ──")
    ng_broadband, wl_targets, ng_vs_wl = compute_ng(data, out_dir, show)

    print("\n── 4. Group index (spectral phase) ──")
    compute_ng_phase(data, out_dir, show)
    if ref_data is not None:
        print("\n── 4b. Group index (spectral phase, ref-normalised) ──")
        compute_ng_phase(data, out_dir, show, ref_data=ref_data)

    print("\n── 5. Transmission spectrum ──")
    plot_transmission(data, out_dir, show)

    print("\n── 6. MPB comparison ──")
    compare_mpb(data, ng_broadband, wl_targets, ng_vs_wl, out_dir, show)

    if not args.skip_movie:
        print("\n── 7. Ey movie ──")
        make_movie(data, out_dir, show)

    print("\n── 8. Snapshots ──")
    plot_snapshots(data, out_dir, show)

    print("\nAll postprocessing complete. Output saved in:", out_dir)


if __name__ == '__main__':
    main()
