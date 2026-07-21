"""Diamond / cladding media and GeV:C n[,k] dispersion fitting."""

from pathlib import Path

import numpy as np
import tidy3d as td
from tidy3d.plugins.dispersion import AdvancedFastFitterParam, FastDispersionFitter

_MATERIAL_DIR = Path(__file__).resolve().parent / "material"

# Two GeV:C tables: lossy (n,k) and lossless (n only)
GEV_CARBON_WITH_K = _MATERIAL_DIR / "GeV_Carbon_interp.txt"
GEV_CARBON_NO_K = _MATERIAL_DIR / "GeV_Carbon_interp_noExtinction.txt"


def n_diamond(wavelength_um):
    """Diamond refractive index via the two-term Sellmeier equation."""
    lam2 = np.asarray(wavelength_um, dtype=float) ** 2
    b1, c1 = 0.3306, 0.175**2
    b2, c2 = 4.3356, 0.106**2
    n2 = 1.0 + b1 * lam2 / (lam2 - c1) + b2 * lam2 / (lam2 - c2)
    return np.sqrt(n2)


diamond_medium = td.Sellmeier(coeffs=[(0.3306, 0.175**2), (4.3356, 0.106**2)])
air_medium = td.Medium(permittivity=1.0)


def fit_gev_carbon(
    fname: Path | str | None = None,
    *,
    with_k: bool = True,
    max_points: int = 1000,
    max_num_poles: int = 3,
    tolerance_rms: float = 2e-2,
    show: bool = True,
):
    """Fit a dispersive GeV:C medium from n[,k] data.

    Parameters
    ----------
    with_k :
        True  -> use ``GeV_Carbon_interp.txt`` (wavelength, n, k).
        False -> use ``GeV_Carbon_interp_noExtinction.txt`` (wavelength, n).
        Ignored if ``fname`` is given explicitly.
    """
    import matplotlib.pyplot as plt

    if fname is None:
        path = GEV_CARBON_WITH_K if with_k else GEV_CARBON_NO_K
    else:
        path = Path(fname)

    # Whitespace-delimited: wavelength (um), n[, k]
    data = np.loadtxt(path, skiprows=1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected columns wl, n[, k] in {path}, got shape {data.shape}")

    if data.shape[0] > max_points:
        idx = np.linspace(0, data.shape[0] - 1, max_points, dtype=int)
        data = data[idx]

    wvl_um = data[:, 0]
    n_data = data[:, 1]
    # Only use k when the table actually has a 3rd column
    k_data = data[:, 2] if data.shape[1] >= 3 else None

    fitter = FastDispersionFitter(wvl_um=wvl_um, n_data=n_data, k_data=k_data)
    label = "with k" if k_data is not None else "no k (lossless)"

    if show:
        fitter.plot()
        plt.suptitle(f"GeV:C data ({label})")
        plt.show()

    advanced_param = AdvancedFastFitterParam(weights=(1, 1))
    medium, rms_error = fitter.fit(
        max_num_poles=max_num_poles,
        advanced_param=advanced_param,
        tolerance_rms=tolerance_rms,
    )
    print(
        f"GeV:C fit [{label}] RMS={rms_error:.3e}  "
        f"({path.name}, {len(wvl_um)} pts, poles<={max_num_poles})"
    )

    if show:
        fitter.plot(medium)
        plt.suptitle(f"GeV:C fit ({label})")
        plt.show()

    return medium, rms_error, fitter


def make_gev_carbon_box(medium: td.Medium, size=(1.0, 1.0, 1.0), name: str | None = None):
    """Demo unit box structure filled with a fitted GeV:C medium."""
    return td.Structure(
        geometry=td.Box(size=size),
        medium=medium,
        name=name,
    )


if __name__ == "__main__":
    # Case 1: carbon with extinction (n, k)
    medium_with_k, _, _ = fit_gev_carbon(with_k=True)
    structure_with_k = make_gev_carbon_box(medium_with_k, name="gev_carbon_with_k")

    # Case 2: carbon without extinction (n only)
    medium_no_k, _, _ = fit_gev_carbon(with_k=False)
    structure_no_k = make_gev_carbon_box(medium_no_k, name="gev_carbon_no_k")

    print(structure_with_k)
    print(structure_no_k)
