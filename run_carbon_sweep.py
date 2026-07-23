"""Carbon-on-cavity thickness × κ sweep (scout → lock-in).

Reuses the Stage-1 / Stage-2 workflow from ``run_lockin.py`` for a
rectangular carbon film extruded from the same ``Cavity_Ideal.gds``
(same footprint and holes) on top of the triangular diamond nanobeam.

Sweep grid (5 cases):
  thickness_um ∈ {0.0, 0.0025, 0.005, 0.0075, 0.010}  ×  with_k = False
Thickness 0.0 is the bare-cavity baseline used as the Δλ reference.

Uses ``GridSpec.auto`` (scout 12 / lock-in 18 steps/λ). Scout has no mesh
override; lock-in uses TWO ``MeshOverrideStructure`` boxes: a coarse one over
the cavity centre (``FINE_MESH_*``) and a thin z slab over the carbon film
(``CARBON_MESH_*``).
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path


@contextlib.contextmanager
def _quiet():
    """Swallow the verbose per-structure build chatter of a setup."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield

# Load project .env BEFORE tidy3d is imported (API key is read at import time).
from siv_cavity.config import load_project_env

load_project_env()

import matplotlib.pyplot as plt
import numpy as np
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
from siv_cavity.materials import carbon_medium_fixed, n_carbon, n_diamond
from siv_cavity.simulation import print_fdtd_summary
from siv_cavity.analysis import extract_resonance, print_resonance

from run_lockin import (
    CARBON_MESH_DL_Z_UM,
    CARBON_MESH_HEIGHT_FACTOR,
    END_WG_LENGTH_UM,
    FINE_MESH_KW,
    FINE_MESH_DL_UM,
    FINE_MESH_SIZE_X_UM,
    FINE_MESH_SIZE_Y_UM,
    FINE_MESH_SIZE_Z_UM,
    LOCKIN_BANDWIDTH_REL,
    LOCKIN_RUN_TIME_PS,
    LOCKIN_STEPS_PER_WVL,
    SCOUT_BANDWIDTH_REL,
    SCOUT_RUN_TIME_PS,
    SCOUT_STEPS_PER_WVL,
    build_setup,
)

# ── Sweep controls ────────────────────────────────────────────────────────────
RUN_CLOUD = False  # submits paid solves when True; the cost estimate below
#                     only runs while this is False, so review it first.
ESTIMATE_COST = True  # upload Jobs solely for web.estimate_cost when dry-run
REUSE_CASE_HDF5 = True  # reuse per-case scout/lockin hdf5 if already present
SAVE_GEOMETRY_PLOTS = (
    False  # off by default (many PNGs per case); set True to inspect geometry
)

# Thin-film batch matching the measured devices (2.5/5/7.5/10 nm). The carbon
# mesh override requests dz = 2.5 nm, but the mesher divides the box into whole
# cells, so the realised dz is 1.56/2.08/2.34/2.08 nm and the film spans
# 1.6/2.4/3.2/4.8 cells — the THINNEST film lands on the FINEST grid.
# Thickness 0.0 is the bare-cavity baseline: no carbon structure, but the same
# override boxes, so Δλ is taken against a comparable grid.
BASELINE_THICKNESS_UM = 0.0
CARBON_THICKNESSES_UM = (BASELINE_THICKNESS_UM, 0.0025, 0.005, 0.0075, 0.010)
WITH_K_CASES = (False,)  # lossless carbon only for now; add True later for κ comparison

# Reference thickness the baseline's carbon override box is sized from, so its
# grid sits in the same family as the sweep cases.
CARBON_MESH_REF_THICKNESS_UM = 0.010

# Override the carbon refractive index. None → use the tabulated value from
# Carbon_interp_noExtinction.txt (n = 2.5140 @ 737 nm). Set a float to test a
# different film index; the sweep root is renamed to match so runs at different
# n never overwrite each other.
CARBON_N_OVERRIDE: float | None = 2.45

# Separate root per mesh/material setting: these cases use a different mesh
# (15 nm coarse + 2.5 nm dz, 18 steps/λ) from the 10–30 nm sweep in
# ``carbon_sweep/``, and must not share a summary.csv with it —
# experiment_export/tidy3d_carbon_sweep.csv is derived from that older file.
_ROOT_NAME = "carbon_sweep_thin"
if CARBON_N_OVERRIDE is not None:
    _ROOT_NAME += f"_n{CARBON_N_OVERRIDE:g}".replace(".", "p")
SWEEP_ROOT = RESULTS_DIR / _ROOT_NAME

# Cloud task names must carry the same discriminator as the output directory.
# Without it, runs at different carbon indices produce identically-named tasks
# that are indistinguishable in the web UI and break per-run cost accounting.
TASK_PREFIX = f"SiV_{_ROOT_NAME.replace('carbon_sweep_', '')}"
SUMMARY_CSV = SWEEP_ROOT / "summary.csv"
TASKS_JSON = SWEEP_ROOT / "tasks.json"
Q_PLOT = SWEEP_ROOT / "q_vs_thickness.png"


def case_tag(thickness_um: float, with_k: bool) -> str:
    suffix = "k" if with_k else "nok"
    if thickness_um <= 0.0:
        return f"baseline_{suffix}"
    nm = thickness_um * 1e3
    # Avoid trailing zeros: 2.5nm, 5.0nm -> 2p5nm / 5nm
    nm_str = f"{nm:g}".replace(".", "p")
    return f"t{nm_str}nm_{suffix}"


def case_dir(thickness_um: float, with_k: bool) -> Path:
    return SWEEP_ROOT / case_tag(thickness_um, with_k)


def prepare_design():
    ensure_runtime_dirs()
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

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
        f"n(diamond) @ {WAVELENGTH_SCOUT_UM * 1e3:.0f} nm = "
        f"{float(n_diamond(WAVELENGTH_SCOUT_UM)):.4f}"
    )
    return cavity, specs, bbox, cavity_gds, holes_gds


def save_geometry_plots(sim: td.Simulation, out_dir: Path, label: str) -> None:
    """Save cross-section plots of the simulation geometry.

    Plot a symmetry-free copy so figures are not half-shaded by the
    ``(1, -1, 0)`` reduced-domain visualization. Cuts use small offsets from
    the midplanes so PolySlab faces render clearly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Full-domain view (no symmetry hatching) — geometry identical to ``sim``.
    plot_sim = sim.updated_copy(symmetry=(0, 0, 0))

    x_cut = 0.05  # µm — triangular cross-section
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_sim.plot(x=x_cut, ax=ax)
    ax.set_title(f"{label}  (x={x_cut:g} µm, triangular CS)")
    fig.tight_layout()
    fig.savefig(out_dir / "geometry_x0.png", dpi=150)
    plt.close(fig)

    y_cut = 0.05  # µm — side view through hole row
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    plot_sim.plot(y=y_cut, ax=ax)
    ax.set_title(f"{label}  (y={y_cut:g} µm, side view)")
    fig.tight_layout()
    fig.savefig(out_dir / "geometry_y0.png", dpi=150)
    plt.close(fig)

    try:
        for s in plot_sim.structures:
            if getattr(s, "name", None) == "carbon_film":
                z0, z1 = s.geometry.bounds[0][2], s.geometry.bounds[1][2]
                z_cut = 0.5 * (float(z0) + float(z1))
                fig, ax = plt.subplots(1, 1, figsize=(10, 3))
                plot_sim.plot(z=z_cut, ax=ax)
                ax.set_title(f"{label}  (z={z_cut:.4f} µm, carbon midplane)")
                fig.tight_layout()
                fig.savefig(out_dir / "geometry_z_carbon.png", dpi=150)
                plt.close(fig)
                break
    except Exception as exc:  # noqa: BLE001
        print(f"  - carbon midplane plot skipped: {exc}")


def estimate_sim_cost(sim: td.Simulation, task_name: str) -> float | None:
    """Upload a Job for cost estimation only (no solve). Returns FlexCredits or None."""
    try:
        job = web.Job(simulation=sim, task_name=task_name, verbose=False)
        cost = float(web.estimate_cost(job.task_id))
        print(f"  - estimate {task_name}: {cost:.3f} FlexCredits")
        return cost
    except Exception as exc:  # noqa: BLE001 — dry-run must continue offline
        print(f"  - estimate skipped for {task_name}: {exc}")
        return None


def run_or_load(
    sim: td.Simulation,
    path: Path,
    task_name: str,
    *,
    run_cloud: bool,
) -> td.SimulationData | None:
    if REUSE_CASE_HDF5 and path.exists():
        print(f"Reusing {path}")
        return td.SimulationData.from_file(str(path))
    if not run_cloud:
        print(f"  - DRY RUN: would submit {task_name} -> {path.name}")
        return None
    return web.run(sim, task_name=task_name, path=str(path))


def write_summary(rows: list[dict]) -> None:
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "thickness_nm",
        "with_k",
        "lambda_scout_nm",
        "Q_scout",
        "lambda_lockin_nm",
        "Q_lockin",
        "scout_cost",
        "lockin_cost",
        "scout_path",
        "lockin_path",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    with TASKS_JSON.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {TASKS_JSON}")


def plot_q_vs_thickness(rows: list[dict]) -> None:
    """Plot lock-in Q vs thickness for with-k and no-k (skips incomplete rows)."""
    series = {True: [], False: []}
    for row in rows:
        q = row.get("Q_lockin")
        if q is None:
            continue
        series[bool(row["with_k"])].append((float(row["thickness_nm"]), float(q)))

    if not any(series.values()):
        print("No lock-in Q values yet — skipping q_vs_thickness plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for with_k, pts in series.items():
        if not pts:
            continue
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.semilogy(xs, ys, "o-", label=("with κ" if with_k else "no κ"))
    ax.set_xlabel("Carbon thickness (nm)")
    ax.set_ylabel("Lock-in Q")
    ax.set_title("Carbon film on SiV nanobeam — Q vs thickness")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Q_PLOT, dpi=150)
    plt.close(fig)
    print(f"Wrote {Q_PLOT}")


def run_one_case(
    *,
    cavity,
    cavity_gds,
    holes_gds,
    bbox,
    specs,
    thickness_um: float,
    with_k: bool,
    carbon_medium: td.Medium,
    run_cloud: bool,
    estimate_cost: bool,
) -> dict:
    tag = case_tag(thickness_um, with_k)
    out = case_dir(thickness_um, with_k)
    out.mkdir(parents=True, exist_ok=True)
    is_baseline = thickness_um <= 0.0
    print("\n" + "=" * 62)
    label = "BARE CAVITY BASELINE" if is_baseline else f"t={thickness_um * 1e3:g} nm"
    print(f"  CASE {tag}  |  {label}  |  with_k={with_k}")
    print("=" * 62)

    row = {
        "case": tag,
        "thickness_nm": thickness_um * 1e3,
        "with_k": with_k,
        "lambda_scout_nm": None,
        "Q_scout": None,
        "lambda_lockin_nm": None,
        "Q_lockin": None,
        "scout_cost": None,
        "lockin_cost": None,
        "scout_path": str(out / "scout.hdf5"),
        "lockin_path": str(out / "lockin.hdf5"),
    }

    # The baseline carries no carbon film, but still builds the carbon override
    # box (sized from CARBON_MESH_REF_THICKNESS_UM) so its grid matches the
    # sweep cases and the mesh error largely cancels in Δλ.
    carbon_kw = dict(
        include_carbon=not is_baseline,
        carbon_thickness_um=0.0 if is_baseline else thickness_um,
        carbon_medium=None if is_baseline else carbon_medium,
        carbonmesh_ref_thickness_um=CARBON_MESH_REF_THICKNESS_UM,
    )

    def make_setup(wavelength_um, bandwidth_rel, *, fine_mesh: bool):
        return build_setup(
            cavity,
            cavity_gds,
            holes_gds,
            bbox,
            specs,
            wavelength_um,
            bandwidth_rel,
            **carbon_kw,
            **(FINE_MESH_KW if fine_mesh else {}),
        )

    scout_setup = make_setup(
        WAVELENGTH_SCOUT_UM, SCOUT_BANDWIDTH_REL, fine_mesh=False
    )
    sim_scout = scout_setup.create_q_scout_simulation(
        run_time_ps=SCOUT_RUN_TIME_PS, min_steps_per_wvl=SCOUT_STEPS_PER_WVL
    )
    print_fdtd_summary(sim_scout, scout_setup, f"Scout {tag}", SCOUT_STEPS_PER_WVL)
    if SAVE_GEOMETRY_PLOTS:
        save_geometry_plots(sim_scout, out, tag)

    if estimate_cost and not run_cloud:
        row["scout_cost"] = estimate_sim_cost(sim_scout, f"{TASK_PREFIX}_scout_{tag}")

    data_scout = run_or_load(
        sim_scout, out / "scout.hdf5", f"{TASK_PREFIX}_scout_{tag}", run_cloud=run_cloud
    )
    if data_scout is None:
        # Still build lock-in geometry for cost / plots even if scout not run.
        lockin_setup = make_setup(
            WAVELENGTH_SCOUT_UM, LOCKIN_BANDWIDTH_REL, fine_mesh=True
        )
        sim_lockin = lockin_setup.create_simulation(
            run_time_ps=LOCKIN_RUN_TIME_PS,
            min_steps_per_wvl=LOCKIN_STEPS_PER_WVL,
            with_farfield=False,
        )
        print_fdtd_summary(
            sim_lockin,
            lockin_setup,
            f"Lock-in {tag} (pre-scout λ)",
            LOCKIN_STEPS_PER_WVL,
        )
        if estimate_cost and not run_cloud:
            row["lockin_cost"] = estimate_sim_cost(
                sim_lockin, f"{TASK_PREFIX}_lockin_{tag}"
            )
        return row

    res_scout = extract_resonance(
        data_scout,
        wavelength_centre_um=WAVELENGTH_SCOUT_UM,
        bandwidth_rel=SCOUT_BANDWIDTH_REL,
    )
    print_resonance(res_scout, f"Scout {tag}")
    row["lambda_scout_nm"] = res_scout["wavelength_nm"]
    row["Q_scout"] = res_scout["Q"]
    wavelength_lockin = res_scout["wavelength_um"]

    lockin_setup = make_setup(wavelength_lockin, LOCKIN_BANDWIDTH_REL, fine_mesh=True)
    sim_lockin = lockin_setup.create_simulation(
        run_time_ps=LOCKIN_RUN_TIME_PS,
        min_steps_per_wvl=LOCKIN_STEPS_PER_WVL,
        with_farfield=False,
    )
    print_fdtd_summary(sim_lockin, lockin_setup, f"Lock-in {tag}", LOCKIN_STEPS_PER_WVL)

    if estimate_cost and not run_cloud:
        row["lockin_cost"] = estimate_sim_cost(sim_lockin, f"{TASK_PREFIX}_lockin_{tag}")

    data_lockin = run_or_load(
        sim_lockin, out / "lockin.hdf5", f"{TASK_PREFIX}_lockin_{tag}", run_cloud=run_cloud
    )
    if data_lockin is None:
        return row

    res_lockin = extract_resonance(
        data_lockin,
        wavelength_centre_um=wavelength_lockin,
        bandwidth_rel=LOCKIN_BANDWIDTH_REL,
    )
    print_resonance(res_lockin, f"Lock-in {tag}")
    row["lambda_lockin_nm"] = res_lockin["wavelength_nm"]
    row["Q_lockin"] = res_lockin["Q"]
    return row


def report_lockin_grids(cavity, cavity_gds, holes_gds, bbox, specs, carbon_medium):
    """Offline grid report for every case — cells, dt, steps, film resolution.

    Builds the lock-in simulations locally (no upload, no FlexCredits) so the
    mesh can be checked before anything is submitted. The z-grid is expected to
    differ between thicknesses: AutoGrid snaps lines to the carbon film's own
    material interface, so Δλ carries a residual mesh uncertainty that the
    baseline only partly cancels.
    """
    print("\nLock-in grid report (built offline, no credits spent):")
    print(
        f"  {'case':>10s} {'cells':>26s} {'total':>12s} {'dt(as)':>8s} "
        f"{'steps':>10s} {'dz(nm)':>8s} {'film cells':>11s}"
    )
    z_lines = {}
    for thickness_um in CARBON_THICKNESSES_UM:
        is_baseline = thickness_um <= 0.0
        setup = build_setup(
            cavity,
            cavity_gds,
            holes_gds,
            bbox,
            specs,
            WAVELENGTH_SCOUT_UM,
            LOCKIN_BANDWIDTH_REL,
            include_carbon=not is_baseline,
            carbon_thickness_um=0.0 if is_baseline else thickness_um,
            carbon_medium=None if is_baseline else carbon_medium,
            carbonmesh_ref_thickness_um=CARBON_MESH_REF_THICKNESS_UM,
            **FINE_MESH_KW,
        )
        with _quiet():
            sim = setup.create_simulation(
                run_time_ps=LOCKIN_RUN_TIME_PS,
                min_steps_per_wvl=LOCKIN_STEPS_PER_WVL,
                with_farfield=False,
            )
        cells = tuple(int(n) for n in sim.grid.num_cells)
        # Measure the realised dz off the built grid — dl_z is only a request
        # and the mesher's rounding cannot be predicted analytically (for the
        # 10 nm film it uses 6 cells where box_h/dl_z is exactly 5).
        if is_baseline:
            n_film, dz_real = "—", float("nan")
        else:
            zb = np.asarray(sim.grid.boundaries.z)
            z0, z1 = setup.carbon_slab_bounds()
            near = zb[(zb > z0 - 0.02) & (zb < z1 + 0.02)]
            dz_real = float(np.min(np.diff(near))) if near.size > 1 else float("nan")
            n_film = f"{thickness_um / dz_real:.2f}"
        z_lines[thickness_um] = len(sim.grid.boundaries.z)
        print(
            f"  {case_tag(thickness_um, False).replace('_nok',''):>10s} "
            f"{str(cells):>26s} {int(cells[0])*int(cells[1])*int(cells[2]):>12,} "
            f"{sim.dt*1e18:>8.2f} {sim.num_time_steps:>10,} {dz_real*1e3:>8.3f} {n_film:>11s}"
        )
    if len(set(z_lines.values())) > 1:
        print(
            f"  NOTE: z-grid differs between cases (n_z = "
            f"{', '.join(str(v) for v in z_lines.values())}). AutoGrid snaps to the\n"
            "        film interface, so Δλ carries a residual mesh uncertainty. "
            "Bound it by\n        comparing the 10 nm case against the older "
            "carbon_sweep/ result (753.075 nm)."
        )


def main():
    # The film is resolved by the carbon override (dz), not by the coarse box —
    # so this checks against CARBON_MESH_DL_Z_UM. Thickness 0.0 is the bare-cavity
    # baseline and is exempt.
    films = [t for t in CARBON_THICKNESSES_UM if t > 0.0]
    if films and min(films) < CARBON_MESH_DL_Z_UM:
        raise ValueError(
            f"Every film must be ≥ CARBON_MESH_DL_Z_UM ({CARBON_MESH_DL_Z_UM*1e3:g} nm) "
            "so at least one cell spans it."
        )

    cavity, specs, bbox, cavity_gds, holes_gds = prepare_design()
    # Non-dispersive carbon (constant n[,k] at WAVELENGTH_SCOUT_UM), same
    # treatment as diamond_medium -- avoids the "dispersive medium into PML"
    # divergence the fitted PoleResidue carbon medium triggers on this
    # full-length film (see siv_cavity/materials.py carbon_medium_fixed).
    if CARBON_N_OVERRIDE is None:
        medium_with_k = carbon_medium_fixed(WAVELENGTH_SCOUT_UM, with_k=True)
        medium_no_k = carbon_medium_fixed(WAVELENGTH_SCOUT_UM, with_k=False)
        n_source = "tabulated"
    else:
        n_tab = n_carbon(WAVELENGTH_SCOUT_UM, with_k=False)
        medium_no_k = td.Medium(permittivity=float(CARBON_N_OVERRIDE) ** 2)
        # Keep the tabulated k, override only n (unused while WITH_K_CASES=(False,)).
        _, k_tab = n_carbon(WAVELENGTH_SCOUT_UM, with_k=True)
        medium_with_k = td.Medium.from_nk(
            n=float(CARBON_N_OVERRIDE), k=k_tab, freq=td.C_0 / WAVELENGTH_SCOUT_UM
        )
        n_source = f"OVERRIDE (tabulated was {n_tab:.4f})"
    print(
        f"Carbon medium (non-dispersive @ {WAVELENGTH_SCOUT_UM * 1e3:.0f} nm): "
        f"no_k n={medium_no_k.permittivity**0.5:.4f}  [{n_source}]"
    )

    print(
        f"AutoGrid + 2 mesh overrides (lock-in only):\n"
        f"  coarse box : dl={FINE_MESH_DL_UM*1e3:.1f} nm, "
        f"size=({FINE_MESH_SIZE_X_UM:.3f}, {FINE_MESH_SIZE_Y_UM:.3f}, "
        f"{FINE_MESH_SIZE_Z_UM:.3f}) µm\n"
        f"  carbon box : dl_z={CARBON_MESH_DL_Z_UM*1e3:.1f} nm, "
        f"height={CARBON_MESH_HEIGHT_FACTOR:g}× film "
        f"(baseline uses {CARBON_MESH_REF_THICKNESS_UM*1e3:g} nm reference)\n"
        f"  steps/λ    : scout={SCOUT_STEPS_PER_WVL}, lockin={LOCKIN_STEPS_PER_WVL}\n"
        f"  end_wg={cavity.end_wg_length:g} µm; with_k={list(WITH_K_CASES)}; "
        f"thicknesses={[t*1e3 for t in CARBON_THICKNESSES_UM]} nm\n"
        f"  output     : {SWEEP_ROOT}"
    )

    balance_before = None
    if RUN_CLOUD:
        balance_before = float(web.account().credit or 0.0)
        print(f"FlexCredit balance: {balance_before:.3f}")
    else:
        print(
            f"DRY RUN (RUN_CLOUD=False). ESTIMATE_COST={ESTIMATE_COST}. "
            "No FDTD solves will be submitted."
        )

    # Offline mesh check BEFORE anything is submitted (its whole purpose).
    report_lockin_grids(cavity, cavity_gds, holes_gds, bbox, specs, medium_no_k)

    rows: list[dict] = []
    for thickness_um in CARBON_THICKNESSES_UM:
        for with_k in WITH_K_CASES:
            medium = medium_with_k if with_k else medium_no_k
            row = run_one_case(
                cavity=cavity,
                cavity_gds=cavity_gds,
                holes_gds=holes_gds,
                bbox=bbox,
                specs=specs,
                thickness_um=thickness_um,
                with_k=with_k,
                carbon_medium=medium,
                run_cloud=RUN_CLOUD,
                estimate_cost=ESTIMATE_COST,
            )
            rows.append(row)

    write_summary(rows)
    # plot_q_vs_thickness(rows) intentionally skipped -- not useful with only
    # this run's 2 points; results reported as a table instead.

    total_est = sum((r["scout_cost"] or 0.0) + (r["lockin_cost"] or 0.0) for r in rows)
    if any(
        r.get("scout_cost") is not None or r.get("lockin_cost") is not None
        for r in rows
    ):
        print(
            f"\nEstimated total FlexCredits "
            f"(scout+lockin, {len(rows)} cases): {total_est:.3f}"
        )
    if balance_before is not None:
        # The estimate calls are skipped when RUN_CLOUD is on, so the only way to
        # capture what this sweep actually cost is the balance delta.
        balance_after = float(web.account().credit or 0.0)
        print(
            f"\nFlexCredit balance after: {balance_after:.3f} "
            f"(spent ≈ {balance_before - balance_after:.3f})"
        )
    else:
        print(
            "\nTo submit cloud solves: set RUN_CLOUD = True in run_carbon_sweep.py "
            "and re-run after reviewing the estimate."
        )


if __name__ == "__main__":
    main()
