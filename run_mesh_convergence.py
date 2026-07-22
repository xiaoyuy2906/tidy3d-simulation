"""Lock-in mesh-convergence sweep: vary the fine-mesh override dl only.

Everything else is locked to the values just validated in run_lockin.py's
baseline (full un-truncated run, correct background grid):
    LOCKIN_RUN_TIME_PS = 15.0, LOCKIN_STEPS_PER_WVL = 25, shutoff = 1e-8
(shutoff is the SiVNanobeamSimulationSetup.create_simulation default).

Reuses the cached scout_baseline.hdf5 for the resonance wavelength -- the
scout uses AutoGrid only (no fine-mesh override), so its result does not
depend on the lock-in mesh being swept here, and does not need to be re-run.

Each mesh point is saved to its own file (lockin_conv_<dl>nm.hdf5) so the
existing 7.5 nm baseline result (lockin_baseline.hdf5, Q=2.673e6) is never
overwritten and can be reused as one point of the sweep.
"""

import sys

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
from siv_cavity.analysis import extract_resonance, print_resonance
from siv_cavity.simulation import print_fdtd_summary
from run_lockin import (
    build_setup,
    END_WG_LENGTH_UM,
    FINE_MESH_SIZE_X_UM,
    FINE_MESH_SIZE_Y_UM,
    FINE_MESH_SIZE_Z_UM,
    LOCKIN_BANDWIDTH_REL,
    LOCKIN_RUN_TIME_PS,
    LOCKIN_STEPS_PER_WVL,
    SCOUT_HDF5,
)


def run_one(dl_um: float):
    dl_nm = dl_um * 1e3
    ensure_runtime_dirs()

    if not SCOUT_HDF5.exists():
        raise FileNotFoundError(
            f"{SCOUT_HDF5} not found -- run run_lockin.py once first to "
            "produce the cached baseline scout."
        )
    data_scout = td.SimulationData.from_file(str(SCOUT_HDF5))
    res_scout = extract_resonance(
        data_scout,
        wavelength_centre_um=WAVELENGTH_SCOUT_UM,
        bandwidth_rel=0.12,
    )
    print_resonance(res_scout, "Cached scout resonance (reused, not re-run)")
    wavelength_lockin = res_scout["wavelength_um"]

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

    lockin_setup = build_setup(
        cavity,
        cavity_gds,
        holes_gds,
        bbox,
        specs,
        wavelength_lockin,
        LOCKIN_BANDWIDTH_REL,
        finemesh_dl_um=dl_um,
        finemesh_size_x_um=FINE_MESH_SIZE_X_UM,
        finemesh_size_y_um=FINE_MESH_SIZE_Y_UM,
        finemesh_size_z_um=FINE_MESH_SIZE_Z_UM,
    )
    sim_lockin = lockin_setup.create_simulation(
        run_time_ps=LOCKIN_RUN_TIME_PS,
        min_steps_per_wvl=LOCKIN_STEPS_PER_WVL,
        with_farfield=True,
        # shutoff defaults to 1e-8 (fixed default) -- not overridden here.
    )
    print_fdtd_summary(
        sim_lockin, lockin_setup, f"Mesh convergence: dl={dl_nm:.1f} nm", LOCKIN_STEPS_PER_WVL
    )

    out_path = RESULTS_DIR / f"lockin_conv_{dl_nm:.1f}nm.hdf5"
    data_lockin = web.run(
        sim_lockin, task_name=f"SiV_conv_{dl_nm:.1f}nm", path=str(out_path)
    )

    res_lockin = extract_resonance(
        data_lockin,
        wavelength_centre_um=wavelength_lockin,
        bandwidth_rel=LOCKIN_BANDWIDTH_REL,
    )
    print_resonance(res_lockin, f"Mesh convergence result: dl={dl_nm:.1f} nm")
    print(f"\nSaved -> {out_path}")
    return res_lockin


if __name__ == "__main__":
    dl_arg_nm = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    balance = float(web.account().credit or 0.0)
    print(f"FlexCredit balance: {balance:.3f}")
    res = run_one(dl_arg_nm * 1e-3)
    balance_after = float(web.account().credit or 0.0)
    print(f"FlexCredit balance after: {balance_after:.3f} (spent ~= {balance - balance_after:.3f})")
