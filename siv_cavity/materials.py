"""Diamond / cladding media and carbon thin-film n[,k] dispersion fitting."""

from pathlib import Path

import numpy as np
import tidy3d as td
from tidy3d.plugins.dispersion import AdvancedFastFitterParam, FastDispersionFitter

_MATERIAL_DIR = Path(__file__).resolve().parent / "material"

# Two carbon tables: lossy (n,k) and lossless (n only)
CARBON_WITH_K = _MATERIAL_DIR / "Carbon_interp.txt"
CARBON_NO_K = _MATERIAL_DIR / "Carbon_interp_noExtinction.txt"


def n_diamond(wavelength_um):
    """Constant diamond index used in FDTD (non-dispersive)."""
    return np.full_like(np.asarray(wavelength_um, dtype=float), N_DIAMOND, dtype=float)


# Non-dispersive diamond — avoids Sellmeier-in-PML divergence on the nanobeam ends.
N_DIAMOND = 2.4064
diamond_medium = td.Medium(permittivity=N_DIAMOND**2)
air_medium = td.Medium(permittivity=1.0)


def fit_carbon(
    fname: Path | str | None = None,
    *,
    with_k: bool = True,
    max_points: int = 1000,
    max_num_poles: int = 3,
    tolerance_rms: float = 2e-2,
    show: bool = True,
):
    """Fit a dispersive carbon thin-film medium from n[,k] data.

    Parameters
    ----------
    with_k :
        True  -> use ``Carbon_interp.txt`` (wavelength, n, k).
        False -> use ``Carbon_interp_noExtinction.txt`` (wavelength, n).
        Ignored if ``fname`` is given explicitly.
    """
    import matplotlib.pyplot as plt

    if fname is None:
        path = CARBON_WITH_K if with_k else CARBON_NO_K
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
        plt.suptitle(f"Carbon data ({label})")
        plt.show()

    advanced_param = AdvancedFastFitterParam(weights=(1, 1))
    medium, rms_error = fitter.fit(
        max_num_poles=max_num_poles,
        advanced_param=advanced_param,
        tolerance_rms=tolerance_rms,
    )
    print(
        f"Carbon fit [{label}] RMS={rms_error:.3e}  "
        f"({path.name}, {len(wvl_um)} pts, poles<={max_num_poles})"
    )

    if show:
        fitter.plot(medium)
        plt.suptitle(f"Carbon fit ({label})")
        plt.show()

    return medium, rms_error, fitter


def n_carbon(wavelength_um: float = 0.737, *, with_k: bool = False):
    """Interpolated carbon index at a single wavelength (linear interp of the
    raw n[,k] table -- no dispersive pole fit).

    Returns ``n`` (with_k=False) or ``(n, k)`` (with_k=True).
    """
    path = CARBON_WITH_K if with_k else CARBON_NO_K
    data = np.loadtxt(path, skiprows=1)
    wl_um = data[:, 0]
    n = float(np.interp(wavelength_um, wl_um, data[:, 1]))
    if with_k:
        k = float(np.interp(wavelength_um, wl_um, data[:, 2]))
        return n, k
    return n


def carbon_medium_fixed(
    wavelength_um: float = 0.737, *, with_k: bool = False, freq0: float | None = None
) -> td.Medium:
    """Non-dispersive carbon medium: constant n[,k] evaluated at ``wavelength_um``.

    Same treatment as ``diamond_medium`` (constant permittivity, no Sellmeier /
    pole-residue fit) -- avoids the "dispersive medium into PML" divergence a
    fitted PoleResidue carbon medium triggers when the film reaches the FDTD
    domain's PML on the nanobeam ends.
    """
    if with_k:
        n, k = n_carbon(wavelength_um, with_k=True)
        if freq0 is None:
            freq0 = td.C_0 / wavelength_um
        return td.Medium.from_nk(n=n, k=k, freq=freq0)
    n = n_carbon(wavelength_um, with_k=False)
    return td.Medium(permittivity=n**2)


def make_carbon_box(medium: td.Medium, size=(1.0, 1.0, 1.0), name: str | None = None):
    """Demo unit box structure filled with a fitted carbon medium."""
    return td.Structure(
        geometry=td.Box(size=size),
        medium=medium,
        name=name,
    )


_CARBON_CACHE: dict[str, td.Medium] | None = None


def get_carbon_media(*, max_num_poles: int = 3, tolerance_rms: float = 2e-2):
    """Fit and cache both carbon media (with k and without k).

    Returns
    -------
    medium_with_k, medium_no_k : td.Medium
    """
    global _CARBON_CACHE
    if _CARBON_CACHE is not None:
        return _CARBON_CACHE["with_k"], _CARBON_CACHE["no_k"]

    medium_with_k, _, _ = fit_carbon(
        with_k=True,
        max_num_poles=max_num_poles,
        tolerance_rms=tolerance_rms,
        show=False,
    )
    medium_no_k, _, _ = fit_carbon(
        with_k=False,
        max_num_poles=max_num_poles,
        tolerance_rms=tolerance_rms,
        show=False,
    )
    _CARBON_CACHE = {"with_k": medium_with_k, "no_k": medium_no_k}
    return medium_with_k, medium_no_k


if __name__ == "__main__":
    # Case 1: carbon with extinction (n, k)
    medium_with_k, _, _ = fit_carbon(with_k=True)
    structure_with_k = make_carbon_box(medium_with_k, name="carbon_with_k")

    # Case 2: carbon without extinction (n only)
    medium_no_k, _, _ = fit_carbon(with_k=False)
    structure_no_k = make_carbon_box(medium_no_k, name="carbon_no_k")

    print(structure_with_k)
    print(structure_no_k)
