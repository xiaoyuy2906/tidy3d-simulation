# AGENTS.md

## Cursor Cloud specific instructions

This repo is a small [`uv`](https://docs.astral.sh/uv/)-managed Python 3.12 project. Its
"application" is a single Tidy3D photonics notebook, `NanobeamCavity.ipynb`, which builds a
1D photonic-crystal (nanobeam) cavity, runs an FDTD simulation, and analyzes Q-factor and
mode volume. Dependencies are `tidy3d` (+ its scientific stack) declared in `pyproject.toml`
and pinned in `uv.lock`.

### Environment
- `uv` is installed at `~/.local/bin/uv` and is on the login-shell `PATH` (added to `~/.bashrc`).
  The project virtualenv lives at `.venv` and is created/updated by `uv sync` (run by the
  startup update script). Run everything through `uv run ...`.
- Python is pinned to 3.12.12 via `.python-version`; `uv` provisions it automatically.

### Running things
- Execute any Python against the project env with `uv run python ...`.
- Jupyter is intentionally **not** a project dependency. Run it on demand without editing
  `pyproject.toml`, e.g. `uv run --with jupyterlab jupyter lab` (interactive) or
  `uv run --with jupyterlab jupyter nbconvert --to notebook --execute NanobeamCavity.ipynb`.

### Tidy3D cloud (important gotcha)
- Tidy3D is a **cloud** FDTD solver. Any `tidy3d.web` call (`web.test`, `web.upload`,
  `web.run`) requires authentication and consumes paid FlexCredits. The notebook cells that
  call `web.*` (and everything downstream that uses `sim_data`) will fail without credentials.
- Provide credentials via the `TIDY3D_API_KEY` environment variable (add it as a secret) or by
  writing `~/.tidy3d/config` with `apikey = '...'` (e.g. `uv run tidy3d configure`).
- Everything else — geometry assembly, building/validating the `td.Simulation`, and plotting
  with `sim.plot(...)` / `geometry.plot(...)` — runs fully offline and is the best way to
  sanity-check changes without spending credits.

### Lint / tests
- There is no lint config and no automated test suite in this repo.
