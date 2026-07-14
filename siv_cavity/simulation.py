from typing import Dict, List, Tuple

import gdstk
import numpy as np
import tidy3d as td

import os
import sys

from siv_cavity.config import C0_M_PER_S


class SiVNanobeamSimulationSetup:
    """SiV diamond nanobeam cavity with a triangular cross-section."""

    def __init__(
        self,
        cavity_gds,
        holes_gds,
        cavity_bbox_um: tuple[float, float, float, float],
        wg_width_um: float,
        thickness_um: float,
        wavelength_um: float,
        hole_x_positions_um: np.ndarray,
        hole_radius_x_um: np.ndarray,
        hole_center_y_um: float,
        hole_radius_y_um: float,
        ellipse_tolerance_um: float = 0.001,
        source_bandwidth_rel: float = 0.12,
        end_wg_length_um: float = 5.0,
        pad_x_neg: float | None = None,
        pad_x_pos: float | None = None,
        pad_y_um: float = 1.2,
        pad_z_um: float = 1.2,
        hole_layer: Tuple[int, int] = (0, 0),
        chunk_max: int = 100,
        *,
        diamond_medium: td.Medium,
        clad_medium: td.Medium,
        c0_m_per_s: float = C0_M_PER_S,
    ):
        self.cavity_gds = str(cavity_gds)
        self.holes_gds = str(holes_gds)
        self.cavity_bbox_um = tuple(float(v) for v in cavity_bbox_um)
        self.wg_width_um = float(wg_width_um)
        self.thickness_um = float(thickness_um)
        self.wavelength_um = float(wavelength_um)
        self.hole_x_positions_um = np.asarray(hole_x_positions_um, dtype=float)
        self.hole_radius_x_um = np.asarray(hole_radius_x_um, dtype=float)
        self.hole_center_y_um = float(hole_center_y_um)
        self.hole_radius_y_um = float(hole_radius_y_um)
        self.ellipse_tolerance_um = float(ellipse_tolerance_um)
        self.source_bandwidth_rel = float(source_bandwidth_rel)
        self.end_wg_length_um = float(end_wg_length_um)
        x_trunc_um = self.end_wg_length_um / 2
        self.pad_x_neg = float(pad_x_neg if pad_x_neg is not None else -x_trunc_um)
        self.pad_x_pos = float(pad_x_pos if pad_x_pos is not None else -x_trunc_um)
        self.pad_y_um = float(pad_y_um)
        self.pad_z_um = float(pad_z_um)
        self.hole_layer = tuple(hole_layer)
        self.chunk_max = int(chunk_max)
        self.diamond_medium = diamond_medium
        self.clad_medium = clad_medium
        self.f0_center = c0_m_per_s / (self.wavelength_um * 1e-6)

    def geometry_params(self) -> Dict:
        xmin, ymin, xmax, ymax = self.cavity_bbox_um
        left = xmin - self.pad_x_neg
        right = xmax + self.pad_x_pos
        size_x = right - left
        size_y = self.wg_width_um + 2 * self.pad_y_um
        size_z = self.thickness_um + 2 * self.pad_z_um
        cx = 0.5 * (left + right)
        return {
            "xmin": float(xmin),
            "xmax": float(xmax),
            "ymin": float(ymin),
            "ymax": float(ymax),
            "sim_left": float(left),
            "sim_right": float(right),
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
        cavity_lib = gdstk.read_gds(self.cavity_gds)
        cavity_scale = cavity_lib.unit / 1e-6
        cavity_tops = cavity_lib.top_level()
        if not cavity_tops:
            raise RuntimeError("No top-level cells found in cavity GDS.")
        (actual_min, actual_max) = (
            np.array(cavity_tops[0].bounding_box()) * cavity_scale
        )
        actual_cavity_bbox = np.array(
            [actual_min[0], actual_min[1], actual_max[0], actual_max[1]], dtype=float
        )
        expected_cavity_bbox = np.array(self.cavity_bbox_um, dtype=float)
        cavity_bbox_error = actual_cavity_bbox - expected_cavity_bbox
        cavity_ok = bool(
            np.allclose(actual_cavity_bbox, expected_cavity_bbox, atol=atol_um)
        )

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
        core_ok = bool(
            np.allclose(actual_core_bounds, expected_core_bounds, atol=atol_um)
        )

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
        actual_holes = np.array(
            sorted(actual_holes, key=lambda row: row[0]), dtype=float
        )

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
            max_abs_hole_error = (
                float(np.max(np.abs(hole_errors))) if expected.size else 0.0
            )
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
        w = self.wg_width_um
        t = self.thickness_um
        return [(-w / 2, +t / 2), (+w / 2, +t / 2), (0.0, -t / 2)]

    def create_core_structure(self, geom: Dict) -> td.Structure:
        w = self.wg_width_um
        t = self.thickness_um
        triangle_yz = self.triangle_vertices_yz()
        core_geo = td.PolySlab(
            vertices=triangle_yz,
            axis=0,
            slab_bounds=(geom["xmin"], geom["xmax"]),
        )
        print("Creating triangular core structure...")
        print(f"  - Top width   (y): {w:.3f} µm")
        print(f"  - Height      (z): {t:.3f} µm")
        print(f"  - Beam length (x): {geom['xmax'] - geom['xmin']:.3f} µm")
        return td.Structure(geometry=core_geo, medium=self.diamond_medium)

    def create_hole_structures(self, geom: Dict) -> List[td.Structure]:
        t = self.thickness_um
        slab_bounds = (-t / 2 - 0.05, t / 2 + 0.05)
        holes = [
            gdstk.ellipse(
                (float(x_um), self.hole_center_y_um),
                (float(rx_um), self.hole_radius_y_um),
                tolerance=self.ellipse_tolerance_um,
                layer=self.hole_layer[0],
                datatype=self.hole_layer[1],
            )
            for x_um, rx_um in zip(self.hole_x_positions_um, self.hole_radius_x_um)
        ]
        geoms = [
            td.PolySlab(vertices=hole.points, axis=2, slab_bounds=slab_bounds)
            for hole in holes
        ]
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
            f"  - Built {len(geoms)} hole polygon(s) -> {len(hole_structs)} structure(s)"
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

    def create_farfield_monitors(self, geom: Dict) -> List[td.Monitor]:
        """Upward far-field projection monitors (Cartesian / k-space / angular).

        The near-field sampling plane is placed above the beam but inside the
        simulation domain (not in the PML)."""
        z_mon = 0.35 * geom["size_z"]
        size_x = 0.8 * geom["size_x"]
        size_y = 0.8 * geom["size_y"]

        cartesian = td.FieldProjectionCartesianMonitor(
            center=(geom["cx"], geom["cy"], z_mon),
            size=(size_x, size_y, 0.0),
            freqs=[self.f0_center],
            name="farfield_cartesian",
            x=list(np.linspace(-4, 4, 50)),
            y=list(np.linspace(-4, 4, 50)),
            proj_axis=2,
            proj_distance=1e6,
        )
        kspace = td.FieldProjectionKSpaceMonitor(
            center=(geom["cx"], geom["cy"], z_mon),
            size=(size_x, size_y, 0.0),
            freqs=[self.f0_center],
            name="farfield_kspace",
            ux=list(np.linspace(-0.95, 0.95, 40)),
            uy=list(np.linspace(-0.95, 0.95, 40)),
            proj_axis=2,
        )
        angles = td.FieldProjectionAngleMonitor(
            center=(geom["cx"], geom["cy"], z_mon),
            size=(size_x, size_y, 0.0),
            freqs=[self.f0_center],
            name="farfield_angles",
            theta=list(np.linspace(0.0, np.pi / 2, 100)),
            phi=list(np.linspace(0.0, 2 * np.pi, 200)),
            proj_distance=1e6,
        )
        return [cartesian, kspace, angles]

    def _grid_spec(self, min_steps_per_wvl: int) -> td.GridSpec:
        return td.GridSpec.auto(
            min_steps_per_wvl=min_steps_per_wvl, wavelength=self.wavelength_um
        )

    def default_symmetry(self) -> Tuple[int, int, int]:
        """Symmetry planes for an Ey point dipole at the cavity centre.

        - x: the cavity (holes + taper) is mirror-symmetric about x = 0, and the
          fundamental Ey mode is even in x  -> +1.
        - y: an Ey dipole is odd under y -> -y                              -> -1.
        - z: the apex-down triangular cross-section is NOT symmetric about
          z = 0, so no z-symmetry is available                             ->  0.

        Using (1, -1, 0) cuts the domain 4x and selects the target mode parity,
        which also cleans up the ringdown used for the Q extraction.
        """
        return (1, -1, 0)

    def _build(
        self,
        geom,
        structures,
        source,
        monitors,
        run_time_ps,
        min_steps_per_wvl,
        symmetry: Tuple[int, int, int] = (0, 0, 0),
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
            medium=self.clad_medium,
            symmetry=symmetry,
        )

    def create_q_scout_simulation(
        self, run_time_ps=6.0, min_steps_per_wvl=14, symmetry=None
    ) -> td.Simulation:
        print("\nCreating minimal scout simulation (Q-only, triangular core)...")
        if symmetry is None:
            symmetry = self.default_symmetry()
        geom = self.geometry_params()
        structures = [self.create_core_structure(geom)] + self.create_hole_structures(
            geom
        )
        source, monitors = self.create_minimal_q_probe(geom)
        sim = self._build(
            geom, structures, source, monitors, run_time_ps, min_steps_per_wvl, symmetry
        )
        print(f"[OK] Scout simulation ready (symmetry={symmetry})")
        return sim

    def create_simulation(
        self,
        run_time_ps=8.0,
        min_steps_per_wvl=14,
        symmetry=None,
        with_farfield: bool = True,
    ) -> td.Simulation:
        """Full lock-in simulation with the 7-monitor characterization suite.

        Monitors: probe, flux, field_near, fld_3d_box, and (optionally) the
        three far-field projection monitors (Cartesian / k-space / angular).
        """
        print("\nCreating full simulation (triangular core)...")
        if symmetry is None:
            symmetry = self.default_symmetry()
        geom = self.geometry_params()
        structures = [self.create_core_structure(geom)] + self.create_hole_structures(
            geom
        )
        source, monitors = self.create_sources_and_monitors(geom)
        monitors = monitors + [self.create_mode_volume_monitor(geom)]
        if with_farfield:
            monitors = monitors + self.create_farfield_monitors(geom)
        sim = self._build(
            geom, structures, source, monitors, run_time_ps, min_steps_per_wvl, symmetry
        )
        print(
            f"[OK] Full simulation ready (symmetry={symmetry}, monitors={len(monitors)})"
        )
        return sim


def _terminal_color(text: str, ansi_code: str) -> str:
    if not sys.stdout.isatty() or os.getenv("NO_COLOR") is not None:
        return text
    return f"\033[{ansi_code}m{text}\033[0m"


def print_fdtd_summary(sim, setup, stage_label, min_steps_per_wvl):
    """Print geometry-to-domain sizing and the important FDTD settings."""
    geom = setup.geometry_params()
    core_size = np.array(
        [geom["xmax"] - geom["xmin"], setup.wg_width_um, setup.thickness_um],
        dtype=float,
    )
    domain_size = np.asarray(sim.size, dtype=float)
    x_trunc = 0.5 * (core_size[0] - domain_size[0])
    y_pad = 0.5 * (domain_size[1] - core_size[1])
    z_pad = 0.5 * (domain_size[2] - core_size[2])
    ratio = domain_size / core_size
    grid = np.asarray(sim.grid.num_cells, dtype=int)
    total = int(np.prod(grid))
    monitors = ", ".join(f"{m.name} ({type(m).__name__})" for m in sim.monitors)

    print(_terminal_color(f"\n── {stage_label}: Geometry and FDTD Settings ──", "1;36"))
    print(_terminal_color("  Geometry:", "1;33"))
    print(
        _terminal_color(
            f"    Core size        : {core_size[0]:.3f} × {core_size[1]:.3f} × {core_size[2]:.3f} µm",
            "32",
        )
    )
    print(
        _terminal_color(
            f"    FDTD domain      : {domain_size[0]:.3f} × {domain_size[1]:.3f} × {domain_size[2]:.3f} µm",
            "35",
        )
    )
    print(_terminal_color(f"    X truncation     : {x_trunc:.3f} µm/side", "34"))
    print(
        _terminal_color(
            f"    Y/Z padding      : y={y_pad:.3f}, z={z_pad:.3f} µm/side", "34"
        )
    )
    print(
        f"    Domain/core ratio: x={ratio[0]:.2f}, y={ratio[1]:.2f}, z={ratio[2]:.2f}"
    )

    print(_terminal_color("  FDTD:", "1;33"))
    print(
        _terminal_color(
            f"    Grid cells       : {grid[0]} × {grid[1]} × {grid[2]} = {total:,}",
            "34",
        )
    )
    print(f"    Grid target      : ≥ {min_steps_per_wvl} steps/wavelength")
    print(f"    Wavelength       : {setup.wavelength_um * 1e3:.1f} nm")
    print(f"    Time step        : {sim.dt * 1e18:.3f} as")
    print(f"    Time steps       : {sim.num_time_steps:,}")
    print(f"    Run time         : {sim.run_time * 1e12:.1f} ps")
    print(f"    Boundary         : PML all sides ({td.PML().num_layers} layers)")

    print(_terminal_color("  Source / monitors:", "1;33"))
    print(f"    Ey dipole @ ({geom['cx']:.3f}, {geom['cy']:.3f}, 0) µm")
    print(f"    Bandwidth        : {setup.source_bandwidth_rel * 100:.1f}%")
    print(f"    Monitors         : {monitors}")
