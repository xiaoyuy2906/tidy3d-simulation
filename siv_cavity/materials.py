import numpy as np
import tidy3d as td


def n_diamond(wavelength_um):
    """Diamond refractive index via the two-term Sellmeier equation."""
    lam2 = np.asarray(wavelength_um, dtype=float) ** 2
    b1, c1 = 0.3306, 0.175**2
    b2, c2 = 4.3356, 0.106**2
    n2 = 1.0 + b1 * lam2 / (lam2 - c1) + b2 * lam2 / (lam2 - c2)
    return np.sqrt(n2)


diamond_medium = td.Sellmeier(coeffs=[(0.3306, 0.175**2), (4.3356, 0.106**2)])
diamond_medium = td.Medium(permittivity=2.404**2)
air_medium = td.Medium(permittivity=1.0)
