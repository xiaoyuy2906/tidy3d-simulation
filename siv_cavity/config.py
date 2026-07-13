import os
from pathlib import Path

# Plot colors
RED = "#840032"
BLUE = "#002642"
WHITE = "#FFFAF2"
BLACK = "#02040f"

# Physical / scout defaults
C0_M_PER_S = 299_792_458.0
WAVELENGTH_SCOUT_UM = 0.737
SIDEWALL_ANGLE_DEG = 25.0
SCOUT_RUN_TIME_PS = 6.0
MIN_STEPS_PER_WVL = 18
SCOUT_BANDWIDTH_REL = 0.12
PERIOD_UM = 0.260

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_ROOT = PROJECT_ROOT / "tidy3d_SiV_diamond_photonic_cavity_runtime"
GDS_DIR = RUNTIME_ROOT / "gds"
RESULTS_DIR = RUNTIME_ROOT / "data" / "results"

def load_project_env() -> bool:
    """Load API key from project-local .env only (never ~/.tidy3d).

    Must be called before ``import tidy3d`` so the key is picked up.
    Accepts either TIDY3D_WEB__APIKEY (Tidy3D 2.11+) or legacy TIDY3D_API_KEY.
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return False

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value

    # Tidy3D reads TIDY3D_WEB__APIKEY, not TIDY3D_API_KEY.
    legacy = os.environ.get("TIDY3D_API_KEY")
    if legacy and not os.environ.get("TIDY3D_WEB__APIKEY"):
        os.environ["TIDY3D_WEB__APIKEY"] = legacy

    return bool(os.environ.get("TIDY3D_WEB__APIKEY") or os.environ.get("SIMCLOUD_APIKEY"))


def ensure_runtime_dirs() -> None:
    GDS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
