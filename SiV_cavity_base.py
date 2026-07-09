import tidy3d.web as web  # if needed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler
from scipy.optimize import curve_fit
from scipy.signal import hilbert

import tidy3d as td
from tidy3d.plugins.resonance import ResonanceFinder
import gdstk


class build_nanobeam_cavity:
    # default values are for the SiV cavity
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
        # Parameters
        # ----------
        # period        : mirror-hole lattice period [um]
        # hole_radius   : ideal circular hole radius [um]
        # wg_width      : nanobeam full width [um]
        # ang           : triangle half-angle / sidewall-related angle [deg]
        # n_hole        : number of holes on each side
        # n_taper       : number of taper holes on each side
        # end_wg_length : straight waveguide length added beyond the holes on each end [um]

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

        # Kalish taper:
        # a_i = 0.9a + 0.1a * (i / (N - 1))^2

        self.a_list = 0.9 * self.period + 0.1 * self.period * frac**2
        # self.a_list: [a0, a1, ..., a_{n_taper-1}]

        self.a_cumsum = np.cumsum(self.a_list)
        # self.a_cumsum:  [a0, a0+a1, a0+a1+a2, ...]
        self.a_total = float(self.a_cumsum[-1])

        self.n_mirror_eff = self.n_hole - self.n_taper

    def _precompute_geometry(self):
        # Positive-x taper hole centers.
        # Equivalent to:
        # positions[0] = a_taper[0] / 2
        # positions[i] = positions[i - 1] + a_taper[i]
        taper_positions = (
            self.a_list[0] / 2
            + np.r_[
                0.0,
                np.cumsum(self.a_list[1:]),
            ]
        )

        # Positive-x mirror hole centers after taper.
        mirror_positions = taper_positions[-1] + self.period * np.arange(
            1,
            self.n_mirror_eff + 1,
            dtype=float,
        )

        positive_positions = np.r_[taper_positions, mirror_positions]

        # Full symmetric hole array.
        self.HOLE_X_POSITIONS_UM = np.r_[
            -positive_positions[::-1],
            positive_positions,
        ]

        # Perfect circular holes: no fabrication error.
        self.HOLE_RADIUS_X_UM = np.full_like(
            self.HOLE_X_POSITIONS_UM,
            self.hole_radius,
            dtype=float,
        )
        self.HOLE_CENTER_Y_UM = 0.0
        self.HOLE_RADIUS_Y_UM = self.hole_radius

        # GDS ellipse polygon approximation tolerance, not fabrication error.
        self.ELLIPSE_TOLERANCE_UM = self.ellipse_tolerance

        self.positive_hole_positions_um = positive_positions
        self.outer_hole_x_um = float(positive_positions[-1])

        # Beam extends past the last hole by half a period plus straight wg.
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
) -> tuple[Path, Path]:
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
    rect = gdstk.rectangle(
        (float(xmin), float(ymin)),
        (float(xmax), float(ymax)),
        layer=0,
        datatype=0,
    )

    cavity_polys = gdstk.boolean([rect], holes, "not") if subtract_holes else [rect]

    cavity_lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cavity_top = gdstk.Cell("TOP")
    for poly in cavity_polys or [rect]:
        cavity_top.add(poly)
    cavity_lib.add(cavity_top)
    cavity_lib.write_gds(str(cavity_path))

    return cavity_path, holes_path


WAVELENGTH_SCOUT = 0.737  # Broadband centre wavelength [µm]  (737 nm)
SIDEWALL_ANGLE_DEG = 25  # Fabrication sidewall angle [degrees]
# NA = 0.65  # Collection numerical aperture
N_BG = 1.0  # Background refractive index (air)
C0 = 299_792_458.0  # Speed of light [m/s]

############################################################################
PERIOD_UM = 0.260

SiV_Cavity = build_nanobeam_cavity(
    period=PERIOD_UM,
    hole_radius=0.075,
    wg_width=0.365,
    ang=SIDEWALL_ANGLE_DEG,
    n_hole=20,
    n_taper=8,
    end_wg_length=4.0,
    ellipse_tolerance=0.001,
)

HOLE_X_POSITIONS_UM = SiV_Cavity.get_hole_specs()["HOLE_X_POSITIONS_UM"]
HOLE_RADIUS_X_UM = SiV_Cavity.get_hole_specs()["HOLE_RADIUS_X_UM"]
HOLE_CENTER_Y_UM = SiV_Cavity.get_hole_specs()["HOLE_CENTER_Y_UM"]
HOLE_RADIUS_Y_UM = SiV_Cavity.get_hole_specs()["HOLE_RADIUS_Y_UM"]
ELLIPSE_TOLERANCE_UM = SiV_Cavity.get_hole_specs()["ELLIPSE_TOLERANCE_UM"]


CAVITY_BBOX_UM = (
    -SiV_Cavity.cav_len / 2,
    -SiV_Cavity.wg_width / 2,
    SiV_Cavity.cav_len / 2,
    SiV_Cavity.wg_width / 2,
)


RUNTIME_ROOT = Path.cwd().resolve() / "tidy3d_SiV_diamond_photonic_cavity_runtime"
GDS_DIR = RUNTIME_ROOT / "gds"
RESULTS_DIR = RUNTIME_ROOT / "data" / "results"
GDS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


CAVITY_GDS, HOLES_GDS = generate_local_gds_from_specs(
    gds_dir=GDS_DIR,
    cavity_bbox_um=CAVITY_BBOX_UM,
    hole_x_positions_um=HOLE_X_POSITIONS_UM,
    hole_radius_x_um=HOLE_RADIUS_X_UM,
    hole_center_y_um=HOLE_CENTER_Y_UM,
    hole_radius_y_um=HOLE_RADIUS_Y_UM,
    ellipse_tolerance_um=ELLIPSE_TOLERANCE_UM,
    force=True,
    subtract_holes=True,
)


def n_diamond(wavelength_um):
    """
    Diamond refractive index via the two-term Sellmeier equation.

    Reference: Zaitsev, Optical Properties of Diamond (2001).
    Valid range: 0.23 – 5 µm.

    Parameters
    ----------
    wavelength_um : float  — free-space wavelength [µm]

    Returns
    -------
    float  — refractive index at the given wavelength
    """
    lam2 = np.asarray(wavelength_um, dtype=float) ** 2
    B1, C1 = 0.3306, 0.175**2  # first oscillator
    B2, C2 = 4.3356, 0.106**2  # second oscillator
    n2 = 1.0 + B1 * lam2 / (lam2 - C1) + B2 * lam2 / (lam2 - C2)
    return np.sqrt(n2)


# Tidy3D dispersive medium (Sellmeier coefficients)
diamond_medium = td.Sellmeier(coeffs=[(0.3306, 0.175**2), (4.3356, 0.106**2)])
air_medium = td.Medium(permittivity=1.0)

n_scout = float(n_diamond(WAVELENGTH_SCOUT))

from typing import Dict, List, Tuple


class SiVNanobeamSimulationSetup:
    """SiV diamond nanobeam cavity with a TRIANGULAR cross-section.

    Analogous to the rectangular-slab ``NotebookSimulationSetup`` in
    ``examples/DiamondPhotonicCrystalCavity.ipynb``, but the diamond core is a
    triangular prism (isosceles triangle in the y-z plane, extruded along x).

    The width, thickness and hole sizes are taken EXACTLY as configured upstream
    in ``build_nanobeam_cavity`` — this class never recomputes or overrides them.
    The triangle is simply the beam's ``wg_width`` (top) pulled to a downward
    apex over the given ``thickness``. The simulation domain is derived from the
    same design-space cavity bbox that was used to write the GDS.
    """

    def __init__(
        self,
        cavity_gds,
        holes_gds,
        cavity_bbox_um: tuple[float, float, float, float],
        wg_width_um: float,
        thickness_um: float,
        wavelength_um: float,
        source_bandwidth_rel: float = 0.12,
        pad_x_um: float = 0.6,
        pad_y_um: float = 0.8,
        pad_z_um: float = 0.8,
        hole_layer: Tuple[int, int] = (0, 0),
        chunk_max: int = 100,
        *,
        diamond_medium: td.Medium,
        clad_medium: td.Medium,
        c0_m_per_s: float = 299_792_458.0,
    ):
        self.cavity_gds = str(cavity_gds)
        self.holes_gds = str(holes_gds)
        self.cavity_bbox_um = tuple(float(v) for v in cavity_bbox_um)
        self.wg_width_um = float(wg_width_um)
        self.thickness_um = float(thickness_um)
        self.wavelength_um = float(wavelength_um)
        self.source_bandwidth_rel = float(source_bandwidth_rel)
        self.pad_x_um = float(pad_x_um)
        self.pad_y_um = float(pad_y_um)
        self.pad_z_um = float(pad_z_um)
        self.hole_layer = tuple(hole_layer)
        self.chunk_max = int(chunk_max)
        self.diamond_medium = diamond_medium
        self.clad_medium = clad_medium
        self.f0_center = c0_m_per_s / (self.wavelength_um * 1e-6)

    def geometry_params(self) -> Dict:
        """Derive the simulation domain from the design-space cavity bbox."""
        xmin, ymin, xmax, ymax = self.cavity_bbox_um
        size_x = (xmax - xmin) + 2 * self.pad_x_um
        size_y = self.wg_width_um + 2 * self.pad_y_um
        size_z = self.thickness_um + 2 * self.pad_z_um
        cx = 0.5 * (xmin + xmax)
        return {
            "xmin": float(xmin),
            "xmax": float(xmax),
            "ymin": float(ymin),
            "ymax": float(ymax),
            "size_x": float(size_x),
            "size_y": float(size_y),
            "size_z": float(size_z),
            "cx": float(cx),
            "cy": 0.0,
            "cz": 0.0,
        }

    def check_gds_matches_design(
        self,
        hole_x_positions_um: np.ndarray,
        hole_radius_x_um: np.ndarray,
        hole_center_y_um: float,
        hole_radius_y_um: float,
        atol_um: float = 2e-3,
        raise_on_fail: bool = False,
        verbose: bool = True,
    ) -> Dict:
        """Compare the written GDS geometry against the design values.

        This is a sanity check for the GDS export path, not part of simulation
        construction. Hole radii are inferred from each hole polygon bbox, so
        the tolerance should be no tighter than the ellipse polygon tolerance.
        """
        cavity_lib = gdstk.read_gds(self.cavity_gds)
        cavity_scale = cavity_lib.unit / 1e-6
        cavity_tops = cavity_lib.top_level()
        if not cavity_tops:
            raise RuntimeError("No top-level cells found in cavity GDS.")
        (actual_min, actual_max) = np.array(cavity_tops[0].bounding_box()) * cavity_scale
        actual_cavity_bbox = np.array(
            [actual_min[0], actual_min[1], actual_max[0], actual_max[1]], dtype=float
        )
        expected_cavity_bbox = np.array(self.cavity_bbox_um, dtype=float)
        cavity_bbox_error = actual_cavity_bbox - expected_cavity_bbox
        cavity_ok = bool(np.allclose(actual_cavity_bbox, expected_cavity_bbox, atol=atol_um))

        core_geo = td.PolySlab(
            vertices=self.triangle_vertices_yz(),
            axis=0,
            slab_bounds=(expected_cavity_bbox[0], expected_cavity_bbox[2]),
        )
        core_min, core_max = core_geo.bounds
        actual_core_bounds = np.array([*core_min, *core_max], dtype=float)
        expected_core_bounds = np.array(
            [
                expected_cavity_bbox[0],
                -self.wg_width_um / 2,
                -self.thickness_um / 2,
                expected_cavity_bbox[2],
                self.wg_width_um / 2,
                self.thickness_um / 2,
            ],
            dtype=float,
        )
        core_bounds_error = actual_core_bounds - expected_core_bounds
        core_ok = bool(np.allclose(actual_core_bounds, expected_core_bounds, atol=atol_um))

        holes_lib = gdstk.read_gds(self.holes_gds)
        holes_scale = holes_lib.unit / 1e-6
        holes_tops = holes_lib.top_level()
        if not holes_tops:
            raise RuntimeError("No top-level cells found in holes GDS.")

        layer, datatype = self.hole_layer
        hole_polygons = [
            poly
            for poly in holes_tops[0].polygons
            if poly.layer == layer and poly.datatype == datatype
        ]
        actual_holes = []
        for poly in hole_polygons:
            bbox = np.array(poly.bounding_box()) * holes_scale
            (xmin, ymin), (xmax, ymax) = bbox
            actual_holes.append(
                (
                    0.5 * (xmin + xmax),
                    0.5 * (ymin + ymax),
                    0.5 * (xmax - xmin),
                    0.5 * (ymax - ymin),
                )
            )
        actual_holes = np.array(sorted(actual_holes, key=lambda row: row[0]), dtype=float)

        expected_x = np.asarray(hole_x_positions_um, dtype=float)
        expected_rx = np.asarray(hole_radius_x_um, dtype=float)
        expected = np.column_stack(
            [
                expected_x,
                np.full_like(expected_x, float(hole_center_y_um)),
                expected_rx,
                np.full_like(expected_x, float(hole_radius_y_um)),
            ]
        )
        expected = expected[np.argsort(expected[:, 0])]

        holes_count_ok = actual_holes.shape[0] == expected.shape[0]
        if holes_count_ok:
            hole_errors = actual_holes - expected
            max_abs_hole_error = float(np.max(np.abs(hole_errors))) if expected.size else 0.0
            holes_ok = bool(np.allclose(actual_holes, expected, atol=atol_um))
        else:
            hole_errors = np.empty((0, 4), dtype=float)
            max_abs_hole_error = float("nan")
            holes_ok = False

        ok = cavity_ok and core_ok and holes_count_ok and holes_ok
        report = {
            "ok": ok,
            "cavity_ok": cavity_ok,
            "cavity_bbox_expected_um": expected_cavity_bbox,
            "cavity_bbox_actual_um": actual_cavity_bbox,
            "cavity_bbox_error_um": cavity_bbox_error,
            "core_ok": core_ok,
            "core_bounds_expected_um": expected_core_bounds,
            "core_bounds_actual_um": actual_core_bounds,
            "core_bounds_error_um": core_bounds_error,
            "holes_ok": holes_ok,
            "holes_count_ok": holes_count_ok,
            "holes_count_expected": int(expected.shape[0]),
            "holes_count_actual": int(actual_holes.shape[0]),
            "max_abs_hole_error_um": max_abs_hole_error,
            "hole_errors_um": hole_errors,
            "atol_um": float(atol_um),
        }

        if verbose:
            print("GDS geometry check:")
            print(f"  - cavity bbox ok : {cavity_ok}")
            print(f"  - core bounds ok : {core_ok}")
            print(f"  - holes count ok : {holes_count_ok}")
            print(f"  - holes values ok: {holes_ok}")
            print(f"  - max hole error : {max_abs_hole_error:.3e} µm")

        if raise_on_fail and not ok:
            raise AssertionError("Geometry does not match design values.")

        return report

    def triangle_vertices_yz(self):
        """Isosceles-triangle vertices in the (y, z) plane."""
        w = self.wg_width_um
        t = self.thickness_um
        return [(-w / 2, +t / 2), (+w / 2, +t / 2), (0.0, -t / 2)]

    def create_core_structure(self, geom: Dict) -> td.Structure:
        """Triangular diamond prism: flat top at +t/2, apex down at -t/2."""
        w = self.wg_width_um
        t = self.thickness_um
        # Vertices in the (y, z) plane (the plane perpendicular to axis=0).
        triangle_yz = self.triangle_vertices_yz()
        core_geo = td.PolySlab(
            vertices=triangle_yz,
            axis=0,  # extrude along x (the beam axis)
            slab_bounds=(geom["xmin"], geom["xmax"]),
        )
        print("Creating triangular core structure...")
        print(f"  - Top width   (y): {w:.3f} µm")
        print(f"  - Height      (z): {t:.3f} µm")
        print(f"  - Beam length (x): {geom['xmax'] - geom['xmin']:.3f} µm")
        return td.Structure(geometry=core_geo, medium=self.diamond_medium)

    def create_hole_structures(self, geom: Dict) -> List[td.Structure]:
        """Import the holes GDS as vertical air cylinders that pierce the beam."""
        lib = gdstk.read_gds(self.holes_gds)
        gds_scale = lib.unit / 1e-6
        tops = lib.top_level()
        if not tops:
            raise RuntimeError("No top-level cells found in holes GDS.")
        t = self.thickness_um
        holes_geo = td.Geometry.from_gds(
            gds_cell=tops[0],
            gds_layer=self.hole_layer[0],
            gds_dtype=self.hole_layer[1],
            axis=2,  # vertical cylinders along z
            # z-extent only sets the etch depth (holes pierce the whole beam);
            # the in-plane hole radii come straight from the GDS and are unchanged.
            slab_bounds=(-t / 2 - 0.05, t / 2 + 0.05),
            reference_plane="middle",
            gds_scale=gds_scale,
        )
        geoms = getattr(holes_geo, "geometries", [holes_geo])
        hole_structs = []
        for i in range(0, len(geoms), self.chunk_max):
            chunk = geoms[i : i + self.chunk_max]
            geom_grp = (
                td.GeometryGroup(geometries=chunk) if len(chunk) > 1 else chunk[0]
            )
            hole_structs.append(
                td.Structure(geometry=geom_grp, medium=self.clad_medium)
            )
        print(
            f"  - Imported {len(geoms)} hole polygon(s) -> {len(hole_structs)} structure(s)"
        )
        return hole_structs

    def _source(self, geom: Dict) -> td.PointDipole:
        return td.PointDipole(
            center=(geom["cx"], geom["cy"], 0.0),
            source_time=td.GaussianPulse(
                freq0=self.f0_center,
                fwidth=self.f0_center * self.source_bandwidth_rel,
            ),
            polarization="Ey",
        )

    def create_minimal_q_probe(
        self, geom: Dict
    ) -> Tuple[td.PointDipole, List[td.Monitor]]:
        probe = td.FieldTimeMonitor(
            center=(geom["cx"], geom["cy"], 0.0),
            size=(0, 0, 0),
            name="probe",
            interval=5,
        )
        return self._source(geom), [probe]

    def create_sources_and_monitors(
        self, geom: Dict
    ) -> Tuple[td.PointDipole, List[td.Monitor]]:
        probe = td.FieldTimeMonitor(
            center=(geom["cx"], geom["cy"], 0.0),
            size=(0, 0, 0),
            name="probe",
            interval=5,
        )
        flux = td.FluxMonitor(
            center=(geom["cx"], geom["cy"], 0.0),
            size=(geom["size_x"] * 0.8, geom["size_y"] * 0.8, 0),
            freqs=[self.f0_center],
            name="flux",
        )
        field_near = td.FieldMonitor(
            center=(geom["cx"], geom["cy"], 0.0),
            size=(geom["size_x"] * 0.8, geom["size_y"] * 0.8, 0),
            freqs=[self.f0_center],
            name="field_near",
        )
        return self._source(geom), [probe, flux, field_near]

    def create_mode_volume_monitor(self, geom: Dict) -> td.Monitor:
        return td.FieldMonitor(
            name="fld_3d_box",
            center=(geom["cx"], geom["cy"], 0.0),
            size=(min(geom["size_x"], 6.0), geom["size_y"], geom["size_z"]),
            fields=["Ex", "Ey", "Ez"],
            freqs=[self.f0_center],
            interval_space=(1, 1, 1),
        )

    def _grid_spec(self, min_steps_per_wvl: int) -> td.GridSpec:
        return td.GridSpec.auto(
            min_steps_per_wvl=min_steps_per_wvl, wavelength=self.wavelength_um
        )

    def _build(
        self, geom, structures, source, monitors, run_time_ps, min_steps_per_wvl
    ):
        return td.Simulation(
            size=(geom["size_x"], geom["size_y"], geom["size_z"]),
            center=(geom["cx"], geom["cy"], geom["cz"]),
            grid_spec=self._grid_spec(min_steps_per_wvl),
            structures=structures,
            sources=[source],
            monitors=monitors,
            run_time=run_time_ps * 1e-12,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
            medium=self.clad_medium,  # suspended in air
        )

    def create_q_scout_simulation(
        self, run_time_ps=6.0, min_steps_per_wvl=14
    ) -> td.Simulation:
        print("\nCreating minimal scout simulation (Q-only, triangular core)...")
        geom = self.geometry_params()
        structures = [self.create_core_structure(geom)] + self.create_hole_structures(
            geom
        )
        source, monitors = self.create_minimal_q_probe(geom)
        sim = self._build(
            geom, structures, source, monitors, run_time_ps, min_steps_per_wvl
        )
        print("✓ Scout simulation ready")
        return sim

    def create_simulation(self, run_time_ps=8.0, min_steps_per_wvl=14) -> td.Simulation:
        print("\nCreating full simulation (triangular core)...")
        geom = self.geometry_params()
        structures = [self.create_core_structure(geom)] + self.create_hole_structures(
            geom
        )
        source, monitors = self.create_sources_and_monitors(geom)
        monitors = monitors + [self.create_mode_volume_monitor(geom)]
        sim = self._build(
            geom, structures, source, monitors, run_time_ps, min_steps_per_wvl
        )
        print("✓ Full simulation ready")
        return sim


TRI_SETUP = SiVNanobeamSimulationSetup(
    cavity_gds=CAVITY_GDS,
    holes_gds=HOLES_GDS,
    cavity_bbox_um=CAVITY_BBOX_UM,
    wg_width_um=SiV_Cavity.wg_width,
    thickness_um=SiV_Cavity.thickness,
    wavelength_um=WAVELENGTH_SCOUT,
    source_bandwidth_rel=0.12,
    diamond_medium=diamond_medium,
    clad_medium=air_medium,
)

print(f"Width  (wg_width)  : {TRI_SETUP.wg_width_um:.4f} µm")
print(f"Height (thickness) : {TRI_SETUP.thickness_um:.4f} µm")
print(f"Source centre frequency : {TRI_SETUP.f0_center / 1e12:.2f} THz")


# ── Build the scout simulation and preview the triangular geometry ────────────
sim_tri_scout = TRI_SETUP.create_q_scout_simulation(
    run_time_ps=6.0,
    min_steps_per_wvl=14,
)

grid = sim_tri_scout.grid.num_cells
print("\n── Triangular SiV scout simulation ──────────────────────────────────")
print(
    f"  Domain     : {sim_tri_scout.size[0]:.3f} × {sim_tri_scout.size[1]:.3f} × {sim_tri_scout.size[2]:.3f} µm"
)
print(f"  Grid cells : {grid[0]} × {grid[1]} × {grid[2]} = {int(np.prod(grid)):,}")
print(f"  Run time   : {sim_tri_scout.run_time * 1e12:.1f} ps")
print(f"  Excitation : Ey dipole @ {WAVELENGTH_SCOUT * 1e3:.0f} nm")

geom = TRI_SETUP.geometry_params()
cx = 0.5 * (geom["xmin"] + geom["xmax"])

fig, (ax_top, ax_cs) = plt.subplots(2, 1, figsize=(12, 5))
sim_tri_scout.plot(z=0.0, ax=ax_top)
ax_top.set_title("Top view (x–y, z = 0): triangular SiV nanobeam with air holes")

sim_tri_scout.plot(x=cx, ax=ax_cs)
ax_cs.set_title(
    f"Cross-section (y–z, x = {cx:.2f} µm): triangular diamond core (apex down)"
)

plt.tight_layout()
plt.show()
