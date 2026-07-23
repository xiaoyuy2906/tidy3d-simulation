# SiV diamond nanobeam cavity — Tidy3D FDTD

3D FDTD simulations of a silicon-vacancy (SiV) photonic-crystal nanobeam cavity
in diamond, built on [Tidy3D](https://www.flexcompute.com/tidy3d/). The cavity
has a **triangular (apex-down) cross-section**, the profile left by angled
etching of bulk diamond.

The current line of work is the resonance red-shift Δλ caused by depositing a
thin carbon film on the beam, compared against measurements from 16 fabricated
devices.

## ⚠️ Tidy3D is a paid cloud solver

Every `tidy3d.web` call uploads to Flexcompute's servers, and `web.run` **spends
FlexCredits**. One lock-in run in this project costs roughly 2–3 credits; a
five-case sweep is ~13.

- `run_carbon_sweep.py` ships with `RUN_CLOUD = False`. Leave it that way until
  you have read the printed cost estimate. Note the estimate only runs **while
  `RUN_CLOUD` is False** — turning it on skips the very review it asks for.
- `run_scout.py`, `run_lockin.py` and `run_mesh_convergence.py` call `web.run`
  unconditionally, with no confirmation prompt. Read them before running them.
- Everything except `web.*` — geometry assembly, building and validating the
  `td.Simulation`, mesh inspection, plotting — runs offline for free. That is
  the right way to check a change.

## Setup

[`uv`](https://docs.astral.sh/uv/) manages the environment; Python is pinned to
3.12 via `.python-version`.

```bash
uv sync
uv run python run_carbon_sweep.py     # dry run: builds and prices, submits nothing
```

Credentials go in a project-local `.env`:

```
TIDY3D_API_KEY=...
```

`siv_cavity.config.load_project_env()` maps it to `TIDY3D_WEB__APIKEY` (what
Tidy3D 2.11+ reads) and **must run before `import tidy3d`** — every driver does
this in its first lines. `.env` is git-ignored.

## Layout

```
siv_cavity/
  geometry.py     nanobeam + hole layout, GDS generation
  materials.py    diamond / air / carbon media
  simulation.py   SiVNanobeamSimulationSetup — structures, monitors, symmetry, mesh
  analysis.py     resonance + Q extraction from the ringdown (ResonanceFinder)
  postprocess.py  mode volume, effective area, Purcell factor, figures
  config.py       paths and physical constants
  material/       carbon n,k tables

run_scout.py            single broadband scout run
run_lockin.py           scout → resonance → narrowband lock-in Q workflow
run_carbon_sweep.py     carbon-thickness sweep, reusing the two-stage workflow
run_mesh_convergence.py Q vs mesh size
analyze_lockin.py       post-process cached lock-in results
```

Simulation outputs (`.hdf5`, figures, per-sweep summaries) land under
`tidy3d_SiV_diamond_photonic_cavity_runtime/` and are **not tracked** — the
`.hdf5` files run to tens of MB each. The GDS files there are regenerated on
every run: they are outputs, not inputs.

## Method

Two stages, because a high-Q cavity cannot be located and measured in one run:

1. **Scout** — broadband Ey dipole, one time-domain probe, coarse grid.
   `ResonanceFinder` (matrix-pencil) locates the resonance.
2. **Lock-in** — a narrowband source centred on that resonance (2 % bandwidth),
   a finer grid and a longer run, giving an accurate Q.

Points that matter for reproducing results:

- **Symmetry is `(1, -1, 0)`**: the mode is even in x, an Ey dipole is odd in y,
  and the apex-down triangle breaks z. Cuts the domain 4× *and* selects the
  target mode.
- **High-Q cavities never fully decay** within a practical `run_time`, so the
  "field decay" shutoff warning is expected. Q comes from fitting the ringdown,
  not from waiting for it to vanish.
- **Materials are non-dispersive.** Diamond is a constant `n = 2.4064`, and the
  carbon film is a single constant index too: `carbon_medium_fixed` interpolates
  the n,k table at one wavelength (0.737 µm → n = 2.5140) and builds a plain
  `td.Medium`. The tables in `siv_cavity/material/` are therefore only the
  *source* of that one number — no dispersion model reaches the solver. A fitted
  `PoleResidue` carbon medium was tried and abandoned: the film runs the full
  length of the beam and into the PML, where a dispersive medium makes the solve
  diverge (see `materials.carbon_medium_fixed`). `fit_carbon()` keeps that path
  available for reference but no driver calls it.
- The tracked tables are a 1000-row subsample of 100k-row originals. Since only
  a single interpolated value is ever used, this changes n by 1.6e-8 relative
  (≈1e-5 nm of wavelength) — far below the discretization error.

## Known limitations

Read these before quoting a number out of this repo.

- **Q does not mesh-converge.** The apex of the triangular cross-section is a
  field singularity, so Q keeps drifting as the grid is refined. Values are
  lower-bound estimates, and Q from two different mesh settings is **not
  comparable** — the same cavity gives 2.7e6 on one grid and 5.4e5 on another.
  Δλ is far more robust, since the mesh error largely cancels in a difference.
- **Δλ still carries a residual mesh uncertainty.** AutoGrid snaps grid lines to
  the carbon film's interfaces, so the grid differs slightly between film
  thicknesses. Measured at the one thickness two independent sweeps share, this
  is ~0.14 nm.
- **The film's faces do not sit on grid lines.** The mesh override shadows the
  film structure, leaving both interfaces mid-cell by a thickness-dependent
  offset, so the shift depends on subpixel averaging at a case-dependent
  sub-cell position.
- **The requested `dl_z` is not the realised cell size.** The mesher fits whole
  cells into the override box, so a 2.5 nm request comes out as 1.56–2.34 nm
  depending on film thickness — and the *thinnest* film lands on the *finest*
  grid, making it the most expensive case.
- **Reused `.hdf5` results are not validated against the current mesh.** If mesh
  constants change, delete the cached files rather than trusting a re-run.
