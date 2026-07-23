# AGENTS.md

## Cursor Cloud specific instructions

This repo is a small [`uv`](https://docs.astral.sh/uv/)-managed Python 3.12 project for
Tidy3D photonic-crystal (nanobeam) cavity FDTD simulations. Dependencies are `tidy3d` +
`gdstk` (+ scientific stack) declared in `pyproject.toml` and pinned in `uv.lock`.

Main pieces (note: the notebooks under `examples/` and `SiV_cavity_base_notebook.ipynb`
are git-ignored — they exist only in a local working copy, not in a fresh clone.
`SiV_cavity_base_notebook.ipynb` in particular carries a stale inline copy of
`SiVNanobeamSimulationSetup` that has diverged from the package; do not treat it
as a reference for how the code works):
- `siv_cavity/` package — the reusable SiV-cavity code: `geometry.py` (nanobeam + GDS),
  `materials.py` (diamond/air media), `simulation.py` (`SiVNanobeamSimulationSetup`, triangular
  core, monitors, symmetry), `analysis.py` (`extract_resonance` via `ResonanceFinder`), `config.py`.
- `run_scout.py` — single broadband scout run. `run_lockin.py` — two-stage scout→resonance→
  narrowband lock-in Q workflow, mirroring sections 5-6 of the Community Library
  notebook this project derives from (see README, Origin). That notebook is
  git-ignored under `examples/`, so it is not present in a clone.

### Environment
- `uv` is installed at `~/.local/bin/uv` and is on the login-shell `PATH` (added to `~/.bashrc`).
  The project virtualenv lives at `.venv` and is created/updated by `uv sync` (run by the
  startup update script). Run everything through `uv run ...`.
- Python is pinned to 3.12.12 via `.python-version`; `uv` provisions it automatically.

### Running things
- Execute any Python against the project env with `uv run python ...`, e.g.
 `uv run python run_scout.py` (needs Tidy3D credentials for the final cloud submit step).
- Jupyter is intentionally **not** a project dependency. Run it on demand without editing
 `pyproject.toml`, e.g. `uv run --with jupyterlab jupyter lab` (interactive). Notebook
 paths under `examples/` are not present in a clone — see the note above.

### Tidy3D cloud (important gotcha)
- Tidy3D is a **cloud** FDTD solver. Any `tidy3d.web` call (`web.test`, `web.upload`,
  `web.run`) requires authentication and consumes paid FlexCredits. The notebook cells that
  call `web.*` (and everything downstream that uses `sim_data`) will fail without credentials.
- The `siv_cavity` scripts read the key from a **project-local `.env`** (git-ignored) via
  `siv_cavity.config.load_project_env()`, which must run *before* `import tidy3d`; it maps the
  legacy `TIDY3D_API_KEY` to `TIDY3D_WEB__APIKEY` (what Tidy3D 2.11+ reads). So put
  `TIDY3D_API_KEY=...` in `/workspace/.env` (or add it as a secret). `~/.tidy3d/config` also works.
- Everything else — geometry assembly, building/validating the `td.Simulation`, and plotting
  with `sim.plot(...)` — runs fully offline and is the best way to sanity-check without credits.
  `td.Simulation(...)` validation (monitor placement, symmetry, etc.) runs at construction, so
  build sims locally before spending credits on `web.run`.
- FDTD notes: use `symmetry=(1,-1,0)` for the centred Ey dipole with the apex-down triangular
  core (x even, y odd, z broken by the triangle) — 4x cheaper + selects the mode. High-Q cavities
  never fully decay in a short `run_time` (expect the "field decay" shutoff warning); extract Q
  from the ringdown with `ResonanceFinder` instead of requiring full decay. Large `.hdf5` results
  under `.../data/results/` are git-ignored.

### Lint / tests
- There is no lint config and no automated test suite in this repo.
