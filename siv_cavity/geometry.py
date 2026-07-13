from pathlib import Path

import gdstk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from siv_cavity.config import BLACK, BLUE, RED, WHITE


class build_nanobeam_cavity:
    def __init__(
        self,
        period=0.260,
        hole_radius=0.075,
        wg_width=0.365,
        ang=25,
        n_hole=20,
        n_taper=8,
        end_wg_length=5.0,
        ellipse_tolerance=0.001,
    ):
        self.period = float(period)
        self.hole_radius = float(hole_radius)
        self.wg_width = float(wg_width)
        self.ang = float(ang)
        self.n_hole = int(n_hole)
        self.n_taper = int(n_taper)
        self.end_wg_length = float(end_wg_length)
        self.ellipse_tolerance = float(ellipse_tolerance)

        if self.n_taper < 2:
            raise ValueError("n_taper must be at least 2.")
        if self.n_hole < self.n_taper:
            raise ValueError("n_hole must be >= n_taper.")
        if self.end_wg_length < 0:
            raise ValueError("end_wg_length must be non-negative.")

        self.thickness = self.wg_width / (2 * np.tan(np.deg2rad(self.ang)))
        self._precompute_cavity_params()
        self._precompute_geometry()

    def _precompute_cavity_params(self):
        i = np.arange(self.n_taper, dtype=float)
        frac = i / (self.n_taper - 1)
        self.a_list = 0.9 * self.period + 0.1 * self.period * frac**2
        self.a_cumsum = np.cumsum(self.a_list)
        self.a_total = float(self.a_cumsum[-1])
        self.n_mirror_eff = self.n_hole - self.n_taper

    def _precompute_geometry(self):
        taper_positions = self.a_list[0] / 2 + np.r_[0.0, np.cumsum(self.a_list[1:])]
        mirror_positions = taper_positions[-1] + self.period * np.arange(
            1, self.n_mirror_eff + 1, dtype=float
        )
        positive_positions = np.r_[taper_positions, mirror_positions]
        self.HOLE_X_POSITIONS_UM = np.r_[-positive_positions[::-1], positive_positions]
        self.HOLE_RADIUS_X_UM = np.full_like(
            self.HOLE_X_POSITIONS_UM, self.hole_radius, dtype=float
        )
        self.HOLE_CENTER_Y_UM = 0.0
        self.HOLE_RADIUS_Y_UM = self.hole_radius
        self.ELLIPSE_TOLERANCE_UM = self.ellipse_tolerance
        self.outer_hole_x_um = float(positive_positions[-1])
        self.beam_half_length_um = (
            self.outer_hole_x_um + self.period / 2 + self.end_wg_length
        )
        self.cav_len = 2 * self.beam_half_length_um

    def get_hole_specs(self):
        return {
            "HOLE_X_POSITIONS_UM": self.HOLE_X_POSITIONS_UM,
            "HOLE_RADIUS_X_UM": self.HOLE_RADIUS_X_UM,
            "HOLE_CENTER_Y_UM": self.HOLE_CENTER_Y_UM,
            "HOLE_RADIUS_Y_UM": self.HOLE_RADIUS_Y_UM,
            "ELLIPSE_TOLERANCE_UM": self.ELLIPSE_TOLERANCE_UM,
        }


def cavity_bbox_um(cavity: build_nanobeam_cavity):
    return (
        -cavity.cav_len / 2,
        -cavity.wg_width / 2,
        cavity.cav_len / 2,
        cavity.wg_width / 2,
    )


def generate_local_gds_from_specs(
    gds_dir: Path,
    cavity_bbox_um: tuple[float, float, float, float],
    hole_x_positions_um: np.ndarray,
    hole_radius_x_um: np.ndarray,
    hole_center_y_um: float,
    hole_radius_y_um: float,
    ellipse_tolerance_um: float = 0.001,
    force: bool = False,
    subtract_holes: bool = True,
):
    cavity_path = gds_dir / "Cavity_Ideal.gds"
    holes_path = gds_dir / "Holes_Ideal.gds"
    if cavity_path.exists() and holes_path.exists() and not force:
        return cavity_path, holes_path

    holes = [
        gdstk.ellipse(
            (float(x_um), float(hole_center_y_um)),
            (float(rx_um), float(hole_radius_y_um)),
            tolerance=float(ellipse_tolerance_um),
            layer=0,
            datatype=0,
        )
        for x_um, rx_um in zip(hole_x_positions_um, hole_radius_x_um)
    ]
    holes_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    holes_top = gdstk.Cell("TOP")
    for hole in holes:
        holes_top.add(hole)
    holes_lib.add(holes_top)
    holes_lib.write_gds(str(holes_path))

    xmin, ymin, xmax, ymax = cavity_bbox_um
    rect = gdstk.rectangle((float(xmin), float(ymin)), (float(xmax), float(ymax)))
    cavity_polys = gdstk.boolean([rect], holes, "not") if subtract_holes else [rect]
    cavity_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cavity_top = gdstk.Cell("TOP")
    for poly in cavity_polys or [rect]:
        cavity_top.add(poly)
    cavity_lib.add(cavity_top)
    cavity_lib.write_gds(str(cavity_path))
    return cavity_path, holes_path


def plot_top_view_geometry(holes_gds, cavity_bbox_um, source_xy_um):
    print("\n[Step 3] Creating top-view GDS geometry preview...")
    holes_lib = gdstk.read_gds(str(holes_gds))
    scale = holes_lib.unit / 1e-6
    cell = holes_lib.top_level()[0]
    xmin, ymin, xmax, ymax = cavity_bbox_um
    source_x, source_y = source_xy_um
    polys = [p for p in cell.polygons if p.layer == 0 and p.datatype == 0]

    print(f"  - Holes GDS   : {holes_gds}")
    print(f"  - Footprint   : {xmax - xmin:.3f} × {ymax - ymin:.3f} µm")
    print(f"  - Air holes   : {len(polys)}")
    print(f"  - Dipole      : ({source_x:.3f}, {source_y:.3f}) µm")

    fig, ax = plt.subplots(figsize=(13.2, 4.8))
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fc=BLUE, ec="none", alpha=0.85))
    for poly in polys:
        ax.add_patch(plt.Polygon(poly.points * scale, closed=True, fc=WHITE, ec=BLACK, lw=0.4))
    ax.plot(source_x, source_y, "x", color=RED, ms=8, mew=2)
    ax.set_xlim(xmin - 0.4, xmax + 0.4)
    ax.set_ylim(ymin - 0.4, ymax + 0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("SiV Nanobeam Cavity – GDS Top View")
    plt.tight_layout()
    plt.show()
    print("  [OK] Top-view preview complete")
