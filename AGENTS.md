# AGENTS.md

## Cursor Cloud specific instructions

This repo is a small [`uv`](https://docs.astral.sh/uv/)-managed Python 3.12 project. It
designs a SiV diamond triangular-cross-section nanobeam photonic-crystal cavity, builds a
Tidy3D FDTD simulation of it, and (via the cloud solver) analyzes Q-factor and mode volume.
Dependencies are `tidy3d`, `gdstk`, `pipx` (+ their scientific stack) declared in
`pyproject.toml` and pinned in `uv.lock`.

### Layout
- `siv_cavity/` is the core package: `geometry.py` (cavity design + GDS export), `materials.py`,
 `simulation.py` (`SiVNanobeamSimulationSetup` builds/validates the `td.Simulation`), and
 `config.py` (defaults, runtime dirs, `.env` loading).
- `run_scout.py` is the primary entry point: it designs the cavity, exports GDS into
 `tidy3d_SiV_diamond_photonic_cavity_runtime/gds/`, builds + validates the scout FDTD
 simulation, plots it, then submits it to the Tidy3D cloud. `SiV_cavity_base.py` is a thin
 backward-compatible wrapper around `run_scout.main`.
- `SiV_cavity_3d_preview.py` is a standalone offline `td.Scene.plot_3d()` preview.
- Notebooks live in the repo root (`SiV_cavity_base_notebook.ipynb`) and `examples/`
 (`NanobeamCavity.ipynb`, `DiamondPhotonicCrystalCavity.ipynb`).

### Environment
- `uv` is installed at `~/.local/bin/uv` and is on the login-shell `PATH` (added to `~/.bashrc`).
  The project virtualenv lives at `.venv` and is created/updated by `uv sync` (run by the
  startup update script). Run everything through `uv run ...`.
- Python is pinned to 3.12.12 via `.python-version`; `uv` provisions it automatically.

### Running things
- Execute any Python against the project env with `uv run python ...`, e.g.
 `uv run python run_scout.py` (needs Tidy3D credentials for the final cloud submit step).
- Jupyter is intentionally **not** a project dependency. Run it on demand without editing
 `pyproject.toml`, e.g. `uv run --with jupyterlab jupyter lab` (interactive) or
 `uv run --with jupyterlab jupyter nbconvert --to notebook --execute examples/NanobeamCavity.ipynb`.

### Tidy3D cloud (important gotcha)
- Tidy3D is a **cloud** FDTD solver. Any `tidy3d.web` call (`web.test`, `web.upload`,
 `web.run`) requires authentication and consumes paid FlexCredits. The `run_scout.py` cloud
 step and any notebook cells that call `web.*` (and everything downstream that uses
 `sim_data`) will fail without credentials.
- Provide credentials via the `TIDY3D_API_KEY` environment variable (add it as a secret) or by
 writing `~/.tidy3d/config` with `apikey = '...'` (e.g. `uv run tidy3d configure`).
 Note: `run_scout.py` reads a **project-local `.env`** via `siv_cavity.config.load_project_env`
 before importing tidy3d and maps `TIDY3D_API_KEY` → `TIDY3D_WEB__APIKEY`, so an `.env` at the
 repo root also works.
- Everything else — geometry assembly, GDS export, building/validating the `td.Simulation`,
 and plotting with `sim.plot(...)` — runs fully offline and is the best way to sanity-check
 changes without spending credits.

### Lint / tests
- There is no lint config and no automated test suite in this repo.
