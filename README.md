# SiV diamond nanobeam cavity — Tidy3D FDTD

3D FDTD simulations of a silicon-vacancy (SiV) photonic-crystal nanobeam cavity
in diamond, built on [Tidy3D](https://www.flexcompute.com/tidy3d/). The cavity
has a triangular (apex-down) cross-section, the profile left by angled etching
of bulk diamond.

## Origin

This project started from the Tidy3D Community Library notebook
**"Fabrication-aware modeling of a diamond nanobeam photonic crystal cavity"**
by Alessandro Buzzi (MIT), MIT-licensed, which accompanies
[arXiv:2601.20025](https://arxiv.org/abs/2601.20025).

Taken from it: the two-stage broadband-search → narrowband-characterization
workflow, the `ResonanceFinder` ringdown fit for Q, and the mode-volume /
Purcell / far-field post-processing.

Changed here: the notebook targets the **tin-vacancy (SnV)** range, this
project the **silicon-vacancy (SiV)** range; the self-contained notebook has
been refactored into the reusable `siv_cavity` package with scripted drivers;
and the geometry uses a triangular cross-section with a carbon overlayer.

## ⚠️ Tidy3D is a paid cloud solver

`web.run` **spends FlexCredits**. Before running anything:

- `run_carbon_sweep.py` ships with `RUN_CLOUD = False`, which builds and prices
  the sweep without submitting it. Read the printed estimate first.
- `run_scout.py`, `run_lockin.py` and `run_mesh_convergence.py` call `web.run`
  unconditionally, with no confirmation prompt.
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
`tidy3d_SiV_diamond_photonic_cavity_runtime/` and are not tracked — the `.hdf5`
files run to tens of MB each. The GDS files there are regenerated on every run:
they are outputs, not inputs.

## Method

Two stages, because a high-Q cavity cannot be located and measured in one run:

1. **Scout** — broadband Ey dipole, one time-domain probe, coarse grid.
   `ResonanceFinder` (matrix-pencil) locates the resonance.
2. **Lock-in** — a narrowband source centred on that resonance, a finer grid and
   a longer run, giving an accurate Q.

Implementation notes worth knowing before changing anything:

- **Symmetry is `(1, -1, 0)`**: the mode is even in x, an Ey dipole is odd in y,
  and the apex-down triangle breaks z. Cuts the domain 4× *and* selects the
  target mode.
- **High-Q cavities never fully decay** within a practical `run_time`, so the
  "field decay" shutoff warning is expected. Q comes from fitting the ringdown,
  not from waiting for it to vanish.
- **Materials are non-dispersive.** Both diamond and the carbon film use a
  constant index: `carbon_medium_fixed` interpolates the n,k table at a single
  wavelength and builds a plain `td.Medium`, so no dispersion model reaches the
  solver. A fitted `PoleResidue` medium was tried and abandoned — the film runs
  the full length of the beam and into the PML, where a dispersive medium makes
  the solve diverge. `fit_carbon()` keeps that path available for reference but
  no driver calls it.
- **Q is sensitive to the mesh** and should not be compared across different
  mesh settings. Resonance shifts are more robust, since the mesh error largely
  cancels in a difference between two runs on the same grid.
- **Cached `.hdf5` results are not validated against the current mesh.** If mesh
  constants change, delete the cached files rather than trusting a re-run.
