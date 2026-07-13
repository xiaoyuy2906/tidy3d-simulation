# Standalone 3D preview for the ideal SiV triangular nanobeam cavity.

from typing import Any

import numpy as np
import tidy3d as td

try:
    from tidy3d_lambda import entrypoint
except ImportError:
    # ``tidy3d_lambda`` is available in the Lambda environment, but it is not
    # required for local ``td.Scene.plot_3d()`` previews.
    def entrypoint(function):
        return function


def _parameter(param: Any, name: str, default: Any) -> Any:
    # Read a parameter from a mapping/object, or return its default.
    if param is None:
        return default
    if isinstance(param, dict):
        return param.get(name, default)
    return getattr(param, name, default)


def _hole_positions(period: float, n_hole: int, n_taper: int) -> np.ndarray:
    # Return the symmetric x coordinates of the ideal cavity holes.
    taper_index = np.arange(n_taper, dtype=float)
    taper_fraction = taper_index / (n_taper - 1)
    taper_periods = period * (0.9 + 0.1 * taper_fraction**2)

    taper_positions = taper_periods[0] / 2 + np.concatenate(
        [np.array([0.0]), np.cumsum(taper_periods[1:])]
    )

    n_mirror = n_hole - n_taper
    mirror_positions = taper_positions[-1] + period * np.arange(
        1, n_mirror + 1, dtype=float
    )
    positive_positions = np.concatenate([taper_positions, mirror_positions])
    return np.concatenate([-positive_positions[::-1], positive_positions])


@entrypoint
def siv_nanobeam_cavity(param=None):
    # Build the diamond core and ideal circular air holes for 3D preview.
    period = float(_parameter(param, "period", 0.260))
    hole_radius = float(_parameter(param, "hole_radius", 0.075))
    wg_width = float(_parameter(param, "wg_width", 0.365))
    sidewall_angle_deg = float(_parameter(param, "sidewall_angle_deg", 25.0))
    n_hole = int(_parameter(param, "n_hole", 20))
    n_taper = int(_parameter(param, "n_taper", 8))
    end_wg_length = float(_parameter(param, "end_wg_length", 4.0))
    etch_overshoot = float(_parameter(param, "etch_overshoot", 0.05))

    if n_taper < 2:
        raise ValueError("n_taper must be at least 2.")
    if n_hole < n_taper:
        raise ValueError("n_hole must be greater than or equal to n_taper.")
    if not 0 < sidewall_angle_deg < 90:
        raise ValueError("sidewall_angle_deg must be between 0 and 90 degrees.")

    thickness = wg_width / (2 * np.tan(np.deg2rad(sidewall_angle_deg)))
    hole_x_positions = _hole_positions(period, n_hole, n_taper)
    beam_half_length = (
        float(np.max(np.abs(hole_x_positions))) + period / 2 + end_wg_length
    )

    diamond_medium = td.Sellmeier(coeffs=[(0.3306, 0.175**2), (4.3356, 0.106**2)])
    air_medium = td.Medium(permittivity=1.0)

    # Triangle vertices are in the y-z plane; axis=0 extrudes along x.
    triangle_yz = [
        (-wg_width / 2, thickness / 2),
        (wg_width / 2, thickness / 2),
        (0.0, -thickness / 2),
    ]
    core = td.PolySlab(
        vertices=triangle_yz,
        axis=0,
        slab_bounds=(-beam_half_length, beam_half_length),
    )
    structures = [td.Structure(geometry=core, medium=diamond_medium)]

    # The current ideal design uses circular holes, so Cylinder is sufficient.
    # Air cylinders extend slightly beyond the beam to guarantee a through etch.
    hole_length = thickness + 2 * etch_overshoot
    holes = [
        td.Cylinder(
            axis=2,
            radius=hole_radius,
            center=(float(x_um), 0.0, 0.0),
            length=hole_length,
        )
        for x_um in hole_x_positions
    ]
    structures.append(
        td.Structure(
            geometry=td.GeometryGroup(geometries=holes),
            medium=air_medium,
        )
    )
    return structures


if __name__ == "__main__":
    preview_structures = siv_nanobeam_cavity()
    preview_scene = td.Scene(
        structures=preview_structures,
        medium=td.Medium(permittivity=1.0),
    )
    preview_scene.plot_3d(width=1200, height=700)
