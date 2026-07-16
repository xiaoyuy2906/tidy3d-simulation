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

# ── Optimized FDTD settings ───────────────────────────────────────────────────
SCOUT_RUN_TIME_PS = 10.0      # broadband resonance search
SCOUT_STEPS_PER_WVL = 18
SCOUT_BANDWIDTH_REL = 0.12
LOCKIN_RUN_TIME_PS = 30.0     # long ringdown for accurate high-Q fit
LOCKIN_STEPS_PER_WVL = 20     # finer grid -> higher numerical-Q ceiling
LOCKIN_BANDWIDTH_REL = 0.02   # narrowband at resonance
LOCKIN_CORE_MESH_DL_UM = 0.010  # 10 nm mesh override over the cavity core


def build_setup(
    cavity, cavity_gds, holes_gds, bbox, specs, wavelength_um, bandwidth_rel,
    core_mesh_dl_um=None,
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
        core_mesh_dl_um=core_mesh_dl_um,
        diamond_medium=diamond_medium,
        clad_medium=air_medium,
    )


def main():
    ensure_runtime_dirs()

    # 1. Design cavity + GDS
    cavity = build_nanobeam_cavity(
        period=PERIOD_UM,
        hole_radius=0.075,
        wg_width=0.365,
        ang=SIDEWALL_ANGLE_DEG,
        n_hole=20,
        n_taper=8,
        end_wg_length=5.0,
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
    print(f"n(diamond) @ {WAVELENGTH_SCOUT_UM * 1e3:.0f} nm = {float(n_diamond(WAVELENGTH_SCOUT_UM)):.4f}")

    # 2. Stage 1: optimized scout (symmetry + longer run_time)
    scout_setup = build_setup(
        cavity, cavity_gds, holes_gds, bbox, specs, WAVELENGTH_SCOUT_UM, SCOUT_BANDWIDTH_REL
    )
    sim_scout = scout_setup.create_q_scout_simulation(
        run_time_ps=SCOUT_RUN_TIME_PS, min_steps_per_wvl=SCOUT_STEPS_PER_WVL
    )
    print_fdtd_summary(sim_scout, scout_setup, "Stage 1 Scout", SCOUT_STEPS_PER_WVL)
    scout_path = RESULTS_DIR / "scout_opt.hdf5"
    data_scout = web.run(sim_scout, task_name="SiV_scout_opt", path=str(scout_path))

    res_scout = extract_resonance(
        data_scout,
        wavelength_centre_um=WAVELENGTH_SCOUT_UM,
        bandwidth_rel=SCOUT_BANDWIDTH_REL,
    )
    print_resonance(res_scout, "Stage 1 (scout) resonance")
    wavelength_lockin = res_scout["wavelength_um"]

    # 3. Stage 2: narrowband lock-in at the detected resonance, full monitor suite
    lockin_setup = build_setup(
        cavity, cavity_gds, holes_gds, bbox, specs, wavelength_lockin, LOCKIN_BANDWIDTH_REL,
        core_mesh_dl_um=LOCKIN_CORE_MESH_DL_UM,
    )
    sim_lockin = lockin_setup.create_simulation(
        run_time_ps=LOCKIN_RUN_TIME_PS,
        min_steps_per_wvl=LOCKIN_STEPS_PER_WVL,
        with_farfield=True,
    )
    print_fdtd_summary(sim_lockin, lockin_setup, "Stage 2 Lock-in", LOCKIN_STEPS_PER_WVL)
    lockin_path = RESULTS_DIR / "lockin.hdf5"
    data_lockin = web.run(sim_lockin, task_name="SiV_lockin", path=str(lockin_path))
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
    print("  SiV DIAMOND NANOCAVITY - Q-FACTOR SUMMARY")
    print("=" * 62)
    print(f"  Scout   : lambda = {res_scout['wavelength_nm']:.3f} nm, Q = {res_scout['Q']:.3e}")
    print(f"  Lock-in : lambda = {res_lockin['wavelength_nm']:.3f} nm, Q = {res_lockin['Q']:.3e}")
    print(f"  A_eff   = {results['confinement']['A_eff_um2']:.4f} um^2")
    print(f"  V_eff   = {results['V_eff_um3']:.4f} um^3 = {results['V_norm']:.3f} x (lambda/n)^3")
    print(f"  F_P     = {results['F_P']:.1f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
