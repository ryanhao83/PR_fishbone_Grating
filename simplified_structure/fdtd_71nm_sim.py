"""
3D FDTD simulation of 71nm partial-etch fishbone grating for ng verification.
Builds the structure, runs the simulation, and saves all raw data to .npz.

Usage:
    python fdtd_71nm_sim.py
    python fdtd_71nm_sim.py --resolution 20 --run-time 600   # quick test
"""
import argparse
import os
import numpy as np
import meep as mp

# ── Physical constants (match run_71nm_yeven.py) ─────────────────────
N_SI         = 3.48
N_SIO2       = 1.44
T_SLAB_NM    = 211.0
T_PARTIAL_NM = 71.0

a_nm    = 496.0        # lattice constant [nm]
h_spine = 0.550        # half-width of central spine [a]
W_rib   = 0.484        # rib length along x [a]
h_rib   = 0.514        # rib width along y [a]


# ── Adiabatic taper function (period chirp) ──────────────────────────
# When enabled, the first/last N_TAPER periods have their lattice constant
# gradually reduced so that the band-edge shifts and ng ramps smoothly
# from fast-light (small period) to slow-light (nominal period).

def taper_period(i_period, n_total, n_taper, a_min_frac=0.85):
    """Return the local lattice constant (in units of nominal a) for the i-th period.

    Input taper:  periods 0..n_taper-1  ramp from a_min_frac*a up to a.
    Output taper: periods (n_total-n_taper)..n_total-1  ramp from a down to a_min_frac*a.
    Uniform region: returns 1.0 (= nominal a).

    a_min_frac: smallest period as fraction of nominal a (e.g. 0.85 = 85% of a).
    """
    if i_period < n_taper:
        alpha = (i_period + 1) / (n_taper + 1)
        return a_min_frac + (1.0 - a_min_frac) * alpha
    elif i_period >= n_total - n_taper:
        alpha = (n_total - i_period) / (n_taper + 1)
        return a_min_frac + (1.0 - a_min_frac) * alpha
    else:
        return 1.0


def build_and_run(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Simulation parameters ────────────────────────────────────────
    NUM_PERIODS    = args.num_periods
    USE_TAPER      = args.use_taper
    N_TAPER        = args.n_taper
    L_STRIP        = args.l_strip
    SRC_OFFSET     = args.src_offset
    RESOLUTION     = args.resolution
    PAD_Y          = 1.5
    PAD_Z          = 1.0
    PML_THICK      = 1.0
    MON_IN_OFFSET  = args.mon_in_offset
    MON_OUT_OFFSET = args.mon_out_offset
    LAMBDA_MIN_NM  = args.lambda_min
    LAMBDA_MAX_NM  = args.lambda_max
    SNAPSHOT_INTERVAL = args.snapshot_interval
    RUN_TIME       = args.run_time

    t_slab    = T_SLAB_NM / a_nm
    t_partial = T_PARTIAL_NM / a_nm

    print(f"a = {a_nm} nm")
    print(f"Grating: {NUM_PERIODS} periods = {NUM_PERIODS * a_nm / 1000:.1f} µm")
    print(f"Strip wg: {L_STRIP}a = {L_STRIP * a_nm / 1e3:.2f} µm")
    print(f"Taper: {'ON (' + str(N_TAPER) + ' periods each end)' if USE_TAPER else 'OFF'}")
    print(f"Pulse: {LAMBDA_MIN_NM:.0f}–{LAMBDA_MAX_NM:.0f} nm")
    print(f"Resolution: {RESOLUTION} px/a")

    # ── Layout along x ───────────────────────────────────────────────
    #
    # |<PML>|<-- strip wg -->|<====== grating periods ======>|<PML>|
    #        ^src             ^grating_start       grating_end^
    #                               ^mon1       mon2^

    struct_half_y = h_spine + h_rib + PAD_Y

    # Compute period lengths
    period_a = []
    for i in range(NUM_PERIODS):
        if USE_TAPER:
            period_a.append(taper_period(i, NUM_PERIODS, N_TAPER))
        else:
            period_a.append(1.0)
    period_a = np.array(period_a)
    total_grating_length = np.sum(period_a)

    # Cell size — PML flush against grating output
    sx = PML_THICK + L_STRIP + total_grating_length + PML_THICK
    sy = 2 * struct_half_y + 2 * PML_THICK
    sz = t_slab + 2 * PAD_Z + 2 * PML_THICK

    # Absolute x-coordinates
    strip_left    = -sx / 2 + PML_THICK
    grating_start = strip_left + L_STRIP
    grating_end   = grating_start + total_grating_length

    # Centre of each period (cumulative)
    period_x_centres = []
    x_cursor = grating_start
    for i in range(NUM_PERIODS):
        xc = x_cursor + period_a[i] / 2.0
        period_x_centres.append(xc)
        x_cursor += period_a[i]

    # ── Materials ─────────────────────────────────────────────────────
    Si   = mp.Medium(index=N_SI)
    SiO2 = mp.Medium(index=N_SIO2)

    # ── Geometry ──────────────────────────────────────────────────────
    geometry = []

    # SiO2 substrate
    geometry.append(mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, PAD_Z + PML_THICK),
        center=mp.Vector3(0, 0, -(t_slab / 2 + (PAD_Z + PML_THICK) / 2)),
        material=SiO2))

    # SiO2 top cladding
    geometry.append(mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, PAD_Z + PML_THICK),
        center=mp.Vector3(0, 0, t_slab / 2 + (PAD_Z + PML_THICK) / 2),
        material=SiO2))

    # Continuous partial-etch Si layer (71 nm, spans strip + grating)
    total_wg_length = L_STRIP + total_grating_length
    wg_centre_x = (strip_left + grating_end) / 2.0
    geometry.append(mp.Block(
        size=mp.Vector3(total_wg_length, mp.inf, t_partial),
        center=mp.Vector3(wg_centre_x, 0, -t_slab / 2 + t_partial / 2),
        material=Si))

    # Continuous spine (strip wg + grating)
    geometry.append(mp.Block(
        size=mp.Vector3(total_wg_length, 2 * h_spine, t_slab),
        center=mp.Vector3(wg_centre_x, 0, 0),
        material=Si))

    # Grating ribs (skipped in reference mode → strip waveguide only)
    if args.reference:
        print("*** REFERENCE MODE: no grating ribs — strip waveguide only ***")
    else:
        for i in range(NUM_PERIODS):
            xc = period_x_centres[i]
            geometry.append(mp.Block(
                size=mp.Vector3(W_rib, h_rib, t_slab),
                center=mp.Vector3(xc, h_spine + h_rib / 2, 0),
                material=Si))
            geometry.append(mp.Block(
                size=mp.Vector3(W_rib, h_rib, t_slab),
                center=mp.Vector3(xc, -(h_spine + h_rib / 2), 0),
                material=Si))

    if USE_TAPER:
        print("Taper period profile (nm):")
        for i in range(NUM_PERIODS):
            tag = " <-- taper" if (i < N_TAPER or i >= NUM_PERIODS - N_TAPER) else ""
            if i < N_TAPER or i >= NUM_PERIODS - N_TAPER or i == N_TAPER or i == NUM_PERIODS - N_TAPER - 1:
                print(f"  Period {i:2d}: a_local = {period_a[i]:.3f}a = {period_a[i]*a_nm:.1f} nm{tag}")

    # ── Source ────────────────────────────────────────────────────────
    fcen = a_nm / ((LAMBDA_MIN_NM + LAMBDA_MAX_NM) / 2)
    df   = a_nm * (1 / LAMBDA_MIN_NM - 1 / LAMBDA_MAX_NM)
    src_x = strip_left + SRC_OFFSET

    sources = [mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=df),
        component=mp.Ey,
        center=mp.Vector3(src_x, 0, 0),
        size=mp.Vector3(0, 2 * h_spine + 0.2, t_slab),
    )]

    # ── Monitor positions ─────────────────────────────────────────────
    mon1_x = grating_start + MON_IN_OFFSET
    mon2_x = grating_end   - MON_OUT_OFFSET
    L_mon  = mon2_x - mon1_x

    mon_size_y = 2 * h_spine + 0.2
    mon_size_z = t_slab
    nfreqs = 200

    print(f"\nCell: {sx:.1f} x {sy:.1f} x {sz:.1f} a  =  "
          f"{sx*a_nm/1e3:.1f} x {sy*a_nm/1e3:.1f} x {sz*a_nm/1e3:.1f} µm")
    print(f"Strip wg: x = [{strip_left:.1f}, {grating_start:.1f}]a  ({L_STRIP:.0f}a)")
    print(f"Grating:  x = [{grating_start:.1f}, {grating_end:.1f}]a  "
          f"({NUM_PERIODS} periods, L={total_grating_length:.1f}a)")
    print(f"PML flush at grating output (x = {grating_end:.1f}a)")
    print(f"Source at x = {src_x:.2f}a  (on strip wg)")
    print(f"Mon1 x = {mon1_x:.2f}a,  Mon2 x = {mon2_x:.2f}a")
    print(f"L_mon = {L_mon:.1f}a = {L_mon*a_nm/1e3:.2f} µm")
    print(f"fcen = {fcen:.4f}, df = {df:.4f}  (centre λ = {a_nm/fcen:.1f} nm)")

    # ── Create simulation ─────────────────────────────────────────────
    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy, sz),
        geometry=geometry,
        sources=sources,
        resolution=RESOLUTION,
        boundary_layers=[mp.PML(PML_THICK)],
        symmetries=[mp.Mirror(mp.Y, phase=-1),
                    mp.Mirror(mp.Z, phase=+1)],
    )

    # Flux monitors
    flux_in = sim.add_flux(
        fcen, df, nfreqs,
        mp.FluxRegion(center=mp.Vector3(mon1_x, 0, 0),
                      size=mp.Vector3(0, mon_size_y, mon_size_z)))
    flux_out = sim.add_flux(
        fcen, df, nfreqs,
        mp.FluxRegion(center=mp.Vector3(mon2_x, 0, 0),
                      size=mp.Vector3(0, mon_size_y, mon_size_z)))

    # Time-domain point recorders
    ey_t_in  = []
    ey_t_out = []
    t_record = []

    def record_time_domain(sim):
        ey1 = sim.get_field_point(mp.Ey, mp.Vector3(mon1_x, 0, 0))
        ey2 = sim.get_field_point(mp.Ey, mp.Vector3(mon2_x, 0, 0))
        ey_t_in.append(float(np.real(ey1)))
        ey_t_out.append(float(np.real(ey2)))
        t_record.append(sim.meep_time())

    # Ey snapshot recorder (z=0 plane)
    ey_snapshots = []
    snap_times   = []
    snap_region_x = sx - 2 * PML_THICK
    snap_region_y = sy - 2 * PML_THICK

    def record_ey_snapshot(sim):
        ey = sim.get_array(center=mp.Vector3(0, 0, 0),
                           size=mp.Vector3(snap_region_x, snap_region_y, 0),
                           component=mp.Ey)
        ey_snapshots.append(ey.copy())
        snap_times.append(sim.meep_time())

    # ── Visualise geometry ────────────────────────────────────────────
    sim.init_sim()

    eps_xy = sim.get_array(center=mp.Vector3(0, 0, 0),
                           size=mp.Vector3(snap_region_x, snap_region_y, 0),
                           component=mp.Dielectric)
    eps_xz = sim.get_array(center=mp.Vector3(0, 0, 0),
                           size=mp.Vector3(snap_region_x, 0, sz - 2 * PML_THICK),
                           component=mp.Dielectric)

    # ── Run ───────────────────────────────────────────────────────────
    print(f"\nRunning FDTD for t = {RUN_TIME} ...")
    sim.run(
        mp.at_every(0.5, record_time_domain),
        mp.at_every(SNAPSHOT_INTERVAL, record_ey_snapshot),
        until=RUN_TIME,
    )

    ey_t_in  = np.array(ey_t_in)
    ey_t_out = np.array(ey_t_out)
    t_record = np.array(t_record)
    snap_times = np.array(snap_times)

    print(f"Done. t = {sim.meep_time():.1f} Meep units")
    print(f"Time-domain samples: {len(t_record)}")
    print(f"Ey snapshots: {len(ey_snapshots)}")

    # ── Extract flux spectra ──────────────────────────────────────────
    flux_freqs = np.array(mp.get_flux_freqs(flux_in))
    P_in  = np.array(mp.get_fluxes(flux_in))
    P_out = np.array(mp.get_fluxes(flux_out))

    # ── Save ──────────────────────────────────────────────────────────
    tag = '_ref' if args.reference else ''
    save_path = os.path.join(output_dir, f'fdtd_71nm_data{tag}.npz')
    np.savez_compressed(save_path,
        t_record=t_record,
        ey_t_in=ey_t_in,
        ey_t_out=ey_t_out,
        flux_freqs=flux_freqs,
        P_in=P_in,
        P_out=P_out,
        ey_snapshots=np.array(ey_snapshots),
        snap_times=snap_times,
        eps_xy=eps_xy,
        eps_xz=eps_xz,
        a_nm=a_nm,
        L_mon=L_mon,
        mon1_x=mon1_x,
        mon2_x=mon2_x,
        src_x=src_x,
        grating_start=grating_start,
        grating_end=grating_end,
        snap_region_x=snap_region_x,
        snap_region_y=snap_region_y,
        sx=sx, sy=sy, sz=sz,
        NUM_PERIODS=NUM_PERIODS,
        RESOLUTION=RESOLUTION,
        USE_TAPER=USE_TAPER,
        N_TAPER=N_TAPER,
        LAMBDA_MIN_NM=LAMBDA_MIN_NM,
        LAMBDA_MAX_NM=LAMBDA_MAX_NM,
    )
    print(f"\nSaved all data to {save_path}")
    print(f"  File size: {os.path.getsize(save_path) / 1e6:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D FDTD verification of 71nm fishbone grating')
    parser.add_argument('--num-periods',   type=int,   default=40)
    parser.add_argument('--use-taper',     action='store_true', default=False)
    parser.add_argument('--n-taper',       type=int,   default=3)
    parser.add_argument('--l-strip',       type=float, default=5.0)
    parser.add_argument('--src-offset',    type=float, default=2.0)
    parser.add_argument('--resolution',    type=int,   default=32)
    parser.add_argument('--mon-in-offset', type=float, default=2.0)
    parser.add_argument('--mon-out-offset',type=float, default=2.0)
    parser.add_argument('--lambda-min',    type=float, default=1508.0)
    parser.add_argument('--lambda-max',    type=float, default=1549.0)
    parser.add_argument('--snapshot-interval', type=float, default=20.0)
    parser.add_argument('--run-time',      type=float, default=1200.0)
    parser.add_argument('--output-dir',    type=str,   default='output')
    parser.add_argument('--reference',     action='store_true', default=False,
                        help='Reference run: strip waveguide only (no grating ribs)')
    args = parser.parse_args()
    build_and_run(args)
