"""Post-process saved scout/lock-in results: ringdown, near-field, Purcell.

Loads ``scout_opt.hdf5`` and ``lockin.hdf5`` from RESULTS_DIR (written by
``run_lockin.py``) and runs the full analysis suite — no cloud runs needed:

  - ringdown visualisation (time-domain decay + FFT spectrum) for both stages
  - near-field |E|^2 map with GDS outline + confinement / mode-area fits
  - mode volume V_eff and Purcell factor F_P from the 3-D field box

Figures are saved as PNGs next to the hdf5 files and shown interactively.
"""

# Load project .env BEFORE tidy3d is imported (API key is read at import time).
from siv_cavity.config import load_project_env

load_project_env()

import tidy3d as td

from siv_cavity.config import (
    GDS_DIR,
    PERIOD_UM,
    RESULTS_DIR,
    SIDEWALL_ANGLE_DEG,
    WAVELENGTH_SCOUT_UM,
)
from siv_cavity.geometry import (
    build_nanobeam_cavity,
    cavity_bbox_um,
    generate_local_gds_from_specs,
)
from siv_cavity.analysis import extract_resonance, print_resonance
from siv_cavity.postprocess import run_full_analysis

# Must match the source bandwidths used in run_lockin.py.
SCOUT_BANDWIDTH_REL = 0.12
LOCKIN_BANDWIDTH_REL = 0.02


def main():
    # 1. Rebuild the cavity design (same parameters as run_lockin.py) to get
    #    the outline bbox, slab thickness and the holes GDS for the overlay.
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
        force=False,
    )

    # 2. Load the saved simulation data.
    scout_path = RESULTS_DIR / "scout_opt.hdf5"
    lockin_path = RESULTS_DIR / "lockin.hdf5"
    for p in (scout_path, lockin_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run run_lockin.py first.")
    print(f"Loading {scout_path.name} + {lockin_path.name} from {RESULTS_DIR}")
    data_scout = td.SimulationData.from_file(str(scout_path))
    data_lockin = td.SimulationData.from_file(str(lockin_path))

    # 3. Resonance extraction (same windows as run_lockin.py).
    res_scout = extract_resonance(
        data_scout,
        wavelength_centre_um=WAVELENGTH_SCOUT_UM,
        bandwidth_rel=SCOUT_BANDWIDTH_REL,
    )
    print_resonance(res_scout, "Stage 1 (scout) resonance")

    res_lockin = extract_resonance(
        data_lockin,
        wavelength_centre_um=res_scout["wavelength_um"],
        bandwidth_rel=LOCKIN_BANDWIDTH_REL,
    )
    print_resonance(res_lockin, "Stage 2 (lock-in) resonance")

    # 4. Full analysis: ringdown figures, near-field map, V_eff, Purcell.
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
    print("  SiV DIAMOND NANOCAVITY - ANALYSIS SUMMARY")
    print("=" * 62)
    print(f"  Scout   : lambda = {res_scout['wavelength_nm']:.3f} nm, Q = {res_scout['Q']:.3e}")
    print(f"  Lock-in : lambda = {res_lockin['wavelength_nm']:.3f} nm, Q = {res_lockin['Q']:.3e}")
    print(f"  A_eff   = {results['confinement']['A_eff_um2']:.4f} um^2")
    print(f"  V_eff   = {results['V_eff_um3']:.4f} um^3 = {results['V_norm']:.3f} x (lambda/n)^3")
    print(f"  F_P     = {results['F_P']:.1f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
