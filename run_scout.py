"""Build geometry and scout FDTD simulation."""

# Load project .env BEFORE tidy3d is imported (API key is read at import time).
from siv_cavity.config import load_project_env

load_project_env()

import matplotlib.pyplot as plt
import tidy3d.web as web

from siv_cavity.config import (
    GDS_DIR,
    MIN_STEPS_PER_WVL,
    PERIOD_UM,
    RESULTS_DIR,
    SCOUT_BANDWIDTH_REL,
    SCOUT_RUN_TIME_PS,
    SIDEWALL_ANGLE_DEG,
    WAVELENGTH_SCOUT_UM,
    ensure_runtime_dirs,
)
from siv_cavity.geometry import (
    build_nanobeam_cavity,
    cavity_bbox_um,
    generate_local_gds_from_specs,
    plot_top_view_geometry,
)
from siv_cavity.materials import air_medium, diamond_medium, n_diamond
from siv_cavity.simulation import SiVNanobeamSimulationSetup, print_fdtd_summary


def main():
    # 1. Design cavity
    print("\n[Step 3] Generating ideal SiV nanobeam cavity geometry...")
    cavity = build_nanobeam_cavity(
        period=PERIOD_UM,
        hole_radius=0.075,
        wg_width=0.365,
        ang=SIDEWALL_ANGLE_DEG,
        n_hole=20,
        n_taper=8,
        end_wg_length=5.0,
    )
    print(f"  - Beam length : {cavity.cav_len:.3f} µm")
    print(f"  - Beam width  : {cavity.wg_width:.3f} µm")
    print(f"  - Beam height : {cavity.thickness:.3f} µm")

    # 2. Export GDS
    ensure_runtime_dirs()
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
    print(f"  - Cavity GDS  : {cavity_gds}")
    print(f"  - Holes GDS   : {holes_gds}")

    # 3. Preview GDS top view
    plot_top_view_geometry(holes_gds, bbox, (0.0, 0.0))

    # 4. Build scout simulation
    setup = SiVNanobeamSimulationSetup(
        cavity_gds=cavity_gds,
        holes_gds=holes_gds,
        cavity_bbox_um=bbox,
        wg_width_um=cavity.wg_width,
        thickness_um=cavity.thickness,
        wavelength_um=WAVELENGTH_SCOUT_UM,
        hole_x_positions_um=specs["HOLE_X_POSITIONS_UM"],
        hole_radius_x_um=specs["HOLE_RADIUS_X_UM"],
        hole_center_y_um=specs["HOLE_CENTER_Y_UM"],
        hole_radius_y_um=specs["HOLE_RADIUS_Y_UM"],
        ellipse_tolerance_um=specs["ELLIPSE_TOLERANCE_UM"],
        source_bandwidth_rel=SCOUT_BANDWIDTH_REL,
        end_wg_length_um=cavity.end_wg_length,
        diamond_medium=diamond_medium,
        clad_medium=air_medium,
    )
    print(f"  - n(diamond)  : {float(n_diamond(WAVELENGTH_SCOUT_UM)):.4f}")

    sim = setup.create_q_scout_simulation(
        run_time_ps=SCOUT_RUN_TIME_PS,
        min_steps_per_wvl=MIN_STEPS_PER_WVL,
    )
    print_fdtd_summary(sim, setup, "SiV Scout Simulation", MIN_STEPS_PER_WVL)

    # 5. Tidy3D geometry preview
    geom = setup.geometry_params()
    cx = 0.5 * (geom["xmin"] + geom["xmax"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))
    sim.plot(z=0.0, ax=ax1)
    ax1.set_title("Top view (x–y)")
    sim.plot(x=cx, ax=ax2)
    ax2.set_title(f"Cross-section (y–z, x = {cx:.2f} µm)")
    plt.tight_layout()
    plt.show()

    # 6. Cloud auth check
    web.test()
    print(f"  - Ready to run: web.run(..., path={RESULTS_DIR / 'scout.hdf5'})")


if __name__ == "__main__":
    main()
