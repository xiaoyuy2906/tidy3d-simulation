"""Two-stage SiV cavity characterization: scout -> lock-in Q extraction.

Stage 1 (scout): broadband Ey dipole, minimal probe monitor, symmetry-reduced
domain. The ``ResonanceFinder`` locates the resonance wavelength and a first Q
estimate (see ``examples/DiamondPhotonicCrystalCavity.ipynb`` section 5).

Stage 2 (lock-in): a narrowband source centred on the detected resonance with a
2% bandwidth and the full 7-monitor suite (section 6). A longer run_time and a
finer grid give an accurate, high-Q measurement (target: millions).
"""

# Load project .env BEFORE tidy3d is imported (API key is read at import time).
from siv_cavity.config import load_project_env

load_project_env()

import tidy3d as td
import tidy3d.web as web

from siv_cavity.config import (
    GDS_DIR,
    PERIOD_UM,
    RESULTS_DIR,
    SIDEWALL_ANGLE_DEG,
    WAVELENGTH_SCOUT_UM,
    ensure_runtime_dirs,
)
from siv_cavity.geometry import (
    build_nanobeam_cavity,
    cavity_bbox_um,
    generate_local_gds_from_specs,
)
from siv_cavity.materials import air_medium, diamond_medium, n_diamond
from siv_cavity.simulation import SiVNanobeamSimulationSetup, print_fdtd_summary
from siv_cavity.analysis import extract_resonance, print_resonance
from siv_cavity.postprocess import run_full_analysis

# ── Shared FDTD settings (baseline = carbon sweep; keep identical for comparison)
SCOUT_RUN_TIME_PS = 8.0
SCOUT_STEPS_PER_WVL = 12
SCOUT_BANDWIDTH_REL = 0.12
LOCKIN_RUN_TIME_PS = 15.0
LOCKIN_STEPS_PER_WVL = 18
LOCKIN_BANDWIDTH_REL = 0.02

# Two fine-mesh overrides — lock-in only (scout uses AutoGrid alone).
#
# Coarse box over the cavity centre. 15 nm is deliberately close to the AutoGrid
# cell in diamond at 18 steps/λ (λ/n/18 = 17.0 nm), so it mainly regularises the
# grid rather than refining it.
FINE_MESH_DL_UM = 0.015  # 15 nm
FINE_MESH_SIZE_X_UM = 3.0
FINE_MESH_SIZE_Y_UM = 0.75
FINE_MESH_SIZE_Z_UM = 0.55

# Carbon box: same x/y footprint, thin z slab over the film.
#
# dz sets the Courant time step for the WHOLE domain, so it is the dominant cost
# knob: 1 nm → dt 2.57 as (35.2 FlexCredits for the 5-case sweep), 2.5 nm →
# dt 5.10 as (12.6 FC). 2.5 nm gives 1/2/3/4 cells through the 2.5/5/7.5/10 nm
# films and relies on Tidy3D's subpixel averaging for the thinnest one.
CARBON_MESH_DL_Z_UM = 0.0025  # 2.5 nm
CARBON_MESH_HEIGHT_FACTOR = 1.25

FINE_MESH_KW = dict(
    finemesh_dl_um=FINE_MESH_DL_UM,
    finemesh_size_x_um=FINE_MESH_SIZE_X_UM,
    finemesh_size_y_um=FINE_MESH_SIZE_Y_UM,
    finemesh_size_z_um=FINE_MESH_SIZE_Z_UM,
    carbonmesh_dl_z_um=CARBON_MESH_DL_Z_UM,
    carbonmesh_height_factor=CARBON_MESH_HEIGHT_FACTOR,
)

# Geometry matching the carbon sweep.
END_WG_LENGTH_UM = 5.0

# Do not reuse diverged / stale baseline hdf5 from Sellmeier runs.
REUSE_SCOUT_HDF5 = False
SCOUT_HDF5 = RESULTS_DIR / "scout_baseline.hdf5"
LOCKIN_HDF5 = RESULTS_DIR / "lockin_baseline.hdf5"


def build_setup(
    cavity,
    cavity_gds,
    holes_gds,
    bbox,
    specs,
    wavelength_um,
    bandwidth_rel,
    finemesh_dl_um=None,
    finemesh_size_x_um=None,
    finemesh_size_y_um=None,
    finemesh_size_z_um=None,
    carbonmesh_dl_z_um=None,
    carbonmesh_height_factor=1.25,
    carbonmesh_ref_thickness_um=None,
    include_carbon=False,
    carbon_thickness_um=0.0,
    carbon_medium=None,
    fixed_grid_spec=None,
):
    return SiVNanobeamSimulationSetup(
        cavity_gds=cavity_gds,
        holes_gds=holes_gds,
        cavity_bbox_um=bbox,
        wg_width_um=cavity.wg_width,
        thickness_um=cavity.thickness,
        wavelength_um=wavelength_um,
        hole_x_positions_um=specs["HOLE_X_POSITIONS_UM"],
        hole_radius_x_um=specs["HOLE_RADIUS_X_UM"],
        hole_center_y_um=specs["HOLE_CENTER_Y_UM"],
        hole_radius_y_um=specs["HOLE_RADIUS_Y_UM"],
        ellipse_tolerance_um=specs["ELLIPSE_TOLERANCE_UM"],
        source_bandwidth_rel=bandwidth_rel,
        end_wg_length_um=cavity.end_wg_length,
        diamond_medium=diamond_medium,
        clad_medium=air_medium,
        include_carbon=include_carbon,
        carbon_thickness_um=carbon_thickness_um,
        carbon_medium=carbon_medium,
        finemesh_dl_um=finemesh_dl_um,
        finemesh_size_x_um=finemesh_size_x_um,
        finemesh_size_y_um=finemesh_size_y_um,
        finemesh_size_z_um=finemesh_size_z_um,
        carbonmesh_dl_z_um=carbonmesh_dl_z_um,
        carbonmesh_height_factor=carbonmesh_height_factor,
        carbonmesh_ref_thickness_um=carbonmesh_ref_thickness_um,
        fixed_grid_spec=fixed_grid_spec,
    )


def run_or_load_scout(scout_setup):
    """Reuse baseline scout hdf5 when present; otherwise run Stage 1 on the cloud."""
    if REUSE_SCOUT_HDF5 and SCOUT_HDF5.exists():
        print(f"Reusing scout data: {SCOUT_HDF5}")
        return td.SimulationData.from_file(str(SCOUT_HDF5))

    sim_scout = scout_setup.create_q_scout_simulation(
        run_time_ps=SCOUT_RUN_TIME_PS, min_steps_per_wvl=SCOUT_STEPS_PER_WVL
    )
    print_fdtd_summary(sim_scout, scout_setup, "Stage 1 Scout", SCOUT_STEPS_PER_WVL)
    return web.run(sim_scout, task_name="SiV_scout_baseline", path=str(SCOUT_HDF5))


def main():
    ensure_runtime_dirs()

    balance = float(web.account().credit or 0.0)
    print(f"FlexCredit balance: {balance:.3f}")
    print(
        f"Baseline settings: steps/λ scout={SCOUT_STEPS_PER_WVL}, "
        f"lockin={LOCKIN_STEPS_PER_WVL}; fine mesh (lock-in only) "
        f"dl={FINE_MESH_DL_UM*1e3:.1f} nm, "
        f"size=({FINE_MESH_SIZE_X_UM}, {FINE_MESH_SIZE_Y_UM}, {FINE_MESH_SIZE_Z_UM}) µm; "
        f"end_wg={END_WG_LENGTH_UM:g} µm"
    )
    print(
        "  NOTE: this bare-cavity run carries no carbon override box, so its grid "
        "differs\n        from the carbon sweep cases. For a Δλ baseline matched to "
        "the sweep mesh,\n        run run_carbon_sweep.py with thickness 0.0 instead."
    )

    # 1. Design cavity + GDS (same geometry as carbon sweep)
    cavity = build_nanobeam_cavity(
        period=PERIOD_UM,
        hole_radius=0.075,
        wg_width=0.365,
        ang=SIDEWALL_ANGLE_DEG,
        n_hole=20,
        n_taper=8,
        end_wg_length=END_WG_LENGTH_UM,
    )
    specs = cavity.get_hole_specs()
    bbox = cavity_bbox_um(cavity)
    cavity_gds, holes_gds = generate_local_gds_from_specs(
        gds_dir=GDS_DIR,
        cavity_bbox_um=bbox,
        hole_x_positions_um=specs["HOLE_X_POSITIONS_UM"],
        hole_radius_x_um=specs["HOLE_RADIUS_X_UM"],
        hole_center_y_um=specs["HOLE_CENTER_Y_UM"],
        hole_radius_y_um=specs["HOLE_RADIUS_Y_UM"],
        ellipse_tolerance_um=specs["ELLIPSE_TOLERANCE_UM"],
        force=True,
    )
    print(
        f"n(diamond) @ {WAVELENGTH_SCOUT_UM * 1e3:.0f} nm = {float(n_diamond(WAVELENGTH_SCOUT_UM)):.4f}"
    )

    # 2. Stage 1: scout (AutoGrid only — no MeshOverride)
    scout_setup = build_setup(
        cavity,
        cavity_gds,
        holes_gds,
        bbox,
        specs,
        WAVELENGTH_SCOUT_UM,
        SCOUT_BANDWIDTH_REL,
    )
    data_scout = run_or_load_scout(scout_setup)

    res_scout = extract_resonance(
        data_scout,
        wavelength_centre_um=WAVELENGTH_SCOUT_UM,
        bandwidth_rel=SCOUT_BANDWIDTH_REL,
    )
    print_resonance(res_scout, "Stage 1 (scout) resonance")
    wavelength_lockin = res_scout["wavelength_um"]

    # 3. Stage 2: narrowband lock-in at the detected resonance, full monitor suite
    lockin_setup = build_setup(
        cavity,
        cavity_gds,
        holes_gds,
        bbox,
        specs,
        wavelength_lockin,
        LOCKIN_BANDWIDTH_REL,
        **FINE_MESH_KW,
    )
    sim_lockin = lockin_setup.create_simulation(
        run_time_ps=LOCKIN_RUN_TIME_PS,
        min_steps_per_wvl=LOCKIN_STEPS_PER_WVL,
        with_farfield=True,
    )
    print_fdtd_summary(
        sim_lockin, lockin_setup, "Stage 2 Lock-in", LOCKIN_STEPS_PER_WVL
    )
    data_lockin = web.run(
        sim_lockin, task_name="SiV_lockin_baseline", path=str(LOCKIN_HDF5)
    )
    print(f"  - Monitors: {list(data_lockin.monitor_data.keys())}")

    # 4. Stage 2 Q extraction (narrow window around the lock-in wavelength)
    res_lockin = extract_resonance(
        data_lockin,
        wavelength_centre_um=wavelength_lockin,
        bandwidth_rel=LOCKIN_BANDWIDTH_REL,
    )
    print_resonance(res_lockin, "Stage 2 (lock-in) resonance")

    # 5. Post-processing: ringdown figures, near-field map, V_eff, Purcell
    results = run_full_analysis(
        data_scout=data_scout,
        data_lockin=data_lockin,
        res_scout=res_scout,
        res_lockin=res_lockin,
        cavity_bbox=bbox,
        holes_gds=holes_gds,
        thickness_um=cavity.thickness,
        fig_dir=RESULTS_DIR,
    )

    print("\n" + "=" * 62)
    print("  SiV DIAMOND NANOCAVITY - BASELINE Q-FACTOR SUMMARY")
    print("=" * 62)
    print(
        f"  Scout   : lambda = {res_scout['wavelength_nm']:.3f} nm, Q = {res_scout['Q']:.3e}"
    )
    print(
        f"  Lock-in : lambda = {res_lockin['wavelength_nm']:.3f} nm, Q = {res_lockin['Q']:.3e}"
    )
    print(f"  A_eff   = {results['confinement']['A_eff_um2']:.4f} um^2")
    print(
        f"  V_eff   = {results['V_eff_um3']:.4f} um^3 = {results['V_norm']:.3f} x (lambda/n)^3"
    )
    print(f"  F_P     = {results['F_P']:.1f}")
    print("=" * 62)

    balance_after = float(web.account().credit or 0.0)
    print(
        f"FlexCredit balance after: {balance_after:.3f} (spent ≈ {balance - balance_after:.3f})"
    )


if __name__ == "__main__":
    main()
