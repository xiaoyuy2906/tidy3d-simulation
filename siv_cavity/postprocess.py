"""Post-processing of saved cavity simulations: ringdown, near-field, Purcell.

Ports sections 5 (ringdown visualisation), 7 (near-field mode analysis) and
8 (mode volume / Purcell enhancement) of
``examples/DiamondPhotonicCrystalCavity.ipynb`` into reusable functions that
operate on ``td.SimulationData`` — either fresh from ``web.run`` or reloaded
from disk with ``td.SimulationData.from_file``.

Monitor names expected (see ``SiVNanobeamSimulationSetup.create_simulation``):
``probe`` (FieldTimeMonitor), ``field_near`` (2-D FieldMonitor at z=0) and
``fld_3d_box`` (3-D FieldMonitor).
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import gdstk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit
from scipy.signal import hilbert

import tidy3d as td

from siv_cavity.config import BLACK, BLUE, C0_M_PER_S, RED, WHITE
from siv_cavity.materials import n_diamond

# Monopolar colormap: black -> blue -> red -> yellow -> white (notebook theme)
YELLOW = "#e59500"
mono_cmap = LinearSegmentedColormap.from_list(
    "mono", [BLACK, BLUE, RED, YELLOW, WHITE], N=256
)


def _coords_to_um(arr: np.ndarray) -> np.ndarray:
    """Auto-convert a coordinate array from metres to um if needed."""
    return arr * 1e6 if np.max(np.abs(arr)) < 1e-3 else arr


# ── Ringdown visualisation (notebook section 5) ───────────────────────────────


def plot_ringdown(
    data: td.SimulationData,
    resonance: Dict,
    monitor_name: str = "probe",
    field: str = "Ey",
    wavelength_window_nm: Tuple[float, float] = (550.0, 750.0),
    title_prefix: str = "",
    save_path: Optional[Path] = None,
):
    """Two-panel figure: log-scale |E(t)| ringdown + FFT resonance spectrum.

    ``resonance`` is the dict returned by ``extract_resonance`` (needs
    ``wavelength_um`` and ``Q``) and is used to annotate the spectrum.
    """
    probe = data[monitor_name]
    sig = getattr(probe, field)
    ey = sig.values.squeeze()
    t = sig.coords["t"].values * 1e12  # ps
    env = np.abs(hilbert(ey))

    dt = np.diff(t[:2])[0] * 1e-12
    freqs_fft = np.fft.rfftfreq(len(ey), d=dt)
    spec = np.abs(np.fft.rfft(ey))
    wl_spec = np.where(freqs_fft[1:] > 0, C0_M_PER_S / freqs_fft[1:] * 1e9, np.nan)
    mask = (wl_spec > wavelength_window_nm[0]) & (wl_spec < wavelength_window_nm[1])
    wl_plot = wl_spec[mask] if np.any(mask) else wl_spec
    spec_plot = spec[1:][mask] if np.any(mask) else spec[1:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    signal_abs = np.abs(ey) / np.max(np.abs(ey))
    env_n = env / env.max()
    ringdown_floor = 1e-6

    ax1.semilogy(
        t,
        np.clip(signal_abs, ringdown_floor, None),
        color=BLUE,
        lw=0.9,
        alpha=0.55,
        label=f"$|E_y(t)|$",
    )
    ax1.semilogy(
        t, np.clip(env_n, ringdown_floor, None), color=RED, lw=1.8, label="Envelope |E|"
    )
    ax1.set_xlabel("Time (ps)")
    ax1.set_ylabel("Normalised amplitude (log scale)")
    ax1.set_title(f"{title_prefix}Time-Domain Ringdown")
    ax1.set_ylim(ringdown_floor, 1.2)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend()

    ax2.plot(wl_plot, spec_plot / spec_plot.max(), color=RED, lw=1.8)
    ax2.axvline(
        resonance["wavelength_um"] * 1e3,
        color=BLUE,
        ls="--",
        lw=1.8,
        label=(
            f"$\\lambda_0$ = {resonance['wavelength_um'] * 1e3:.2f} nm\n"
            f"Q = {resonance['Q']:.0f}"
        ),
    )
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Normalised PSD")
    ax2.set_title(f"{title_prefix}Resonance Spectrum")
    ax2.legend()

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved figure -> {save_path}")
    return fig


# ── Near-field mode analysis (notebook section 7) ─────────────────────────────


def extract_nearfield_intensity(
    data: td.SimulationData, monitor_name: str = "field_near"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the 2-D |E|^2 map (shape (ny, nx)) and x/y coordinates in um."""
    mon = data[monitor_name]
    Ex = mon.Ex.isel(f=0).values.squeeze()
    Ey = mon.Ey.isel(f=0).values.squeeze()
    Ez = mon.Ez.isel(f=0).values.squeeze() if "Ez" in mon.field_components else 0.0
    I = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2
    x = _coords_to_um(mon.Ey.coords["x"].values)
    y = _coords_to_um(mon.Ey.coords["y"].values)
    if I.shape == (len(x), len(y)):
        I = I.T
    return I, x, y


def crop_field(
    I: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_lim: Tuple[float, float] = (-2.0, 2.0),
    y_lim: Tuple[float, float] = (-0.7, 0.7),
):
    """Crop a 2-D intensity map to the specified window (um)."""
    xi = (x >= x_lim[0]) & (x <= x_lim[1])
    yi = (y >= y_lim[0]) & (y <= y_lim[1])
    return I[np.ix_(yi, xi)], x[xi], y[yi]


def overlay_gds_outline(
    ax,
    x_window: Tuple[float, float],
    y_window: Tuple[float, float],
    cavity_bbox_um: Tuple[float, float, float, float],
    holes_gds: Path,
) -> None:
    """Overlay the cavity rectangle and hole outlines on a near-field map.

    ``cavity_bbox_um`` is (xmin, ymin, xmax, ymax) as returned by
    ``siv_cavity.geometry.cavity_bbox_um``.
    """
    x0, y0, x1, y1 = cavity_bbox_um
    rect_kw_outer = dict(fill=False, ec=BLACK, lw=1.4, alpha=0.55, zorder=4)
    rect_kw_inner = dict(fill=False, ec=WHITE, lw=0.8, alpha=0.95, zorder=5)
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, **rect_kw_outer))
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, **rect_kw_inner))

    lib = gdstk.read_gds(str(holes_gds))
    scale = lib.unit / 1e-6
    cell = lib.top_level()[0]

    for poly in cell.polygons:
        if poly.layer != 0 or poly.datatype != 0:
            continue
        pts = poly.points * scale
        if (
            pts[:, 0].max() < x_window[0]
            or pts[:, 0].min() > x_window[1]
            or pts[:, 1].max() < y_window[0]
            or pts[:, 1].min() > y_window[1]
        ):
            continue
        ax.add_patch(
            plt.Polygon(
                pts, closed=True, fill=False, ec=BLACK, lw=1.0, alpha=0.5,
                zorder=4, joinstyle="round",
            )
        )
        ax.add_patch(
            plt.Polygon(
                pts, closed=True, fill=False, ec=WHITE, lw=0.4, alpha=0.95,
                zorder=5, joinstyle="round",
            )
        )


def plot_nearfield(
    I: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    wavelength_nm: float,
    cavity_bbox_um: Tuple[float, float, float, float],
    holes_gds: Path,
    cmap=None,
    save_path: Optional[Path] = None,
):
    """Display the near-field intensity map with the GDS outline overlaid."""
    cmap = cmap or mono_cmap
    I_n = I / I.max()
    vmin, vmax = np.percentile(I_n, [0.1, 99.9])

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    im = ax.pcolormesh(x, y, I_n, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    overlay_gds_outline(
        ax, (x.min(), x.max()), (y.min(), y.max()), cavity_bbox_um, holes_gds
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("$|\\mathbf{E}|^2$ (normalised)")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(f"Near-Field Intensity  –  $\\lambda_0$ = {wavelength_nm:.2f} nm")
    ax.set_aspect("equal")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved figure -> {save_path}")
    return fig


def gaussian_1d(x, A, x0, sigma, bg):
    return A * np.exp(-2 * (x - x0) ** 2 / sigma**2) + bg


def compute_confinement(
    I: np.ndarray, x: np.ndarray, y: np.ndarray, wavelength_um: float
) -> Dict:
    """Quantify the lateral confinement from 1-D Gaussian fits.

    Returns a dict with 1/e^2 widths, FWHM, effective mode area
    A_eff = (sum I)^2 / sum I^2 * dx dy, and A_eff normalised to lambda^2.
    """
    results = {}
    for axis_label, coords, profile in [
        ("x", x, I.sum(axis=0)),
        ("y", y, I.sum(axis=1)),
    ]:
        pn = profile / profile.max()
        try:
            p0 = [1.0, coords[np.argmax(pn)], 0.25, 0.0]
            popt, _ = curve_fit(gaussian_1d, coords, pn, p0=p0, maxfev=8000)
            w = abs(popt[2])  # 1/e^2 half-width
        except RuntimeError:
            above = coords[pn > np.exp(-2)]
            w = (above[-1] - above[0]) / 2 if len(above) > 1 else np.nan
        results[f"w_{axis_label}_um"] = w
        results[f"fwhm_{axis_label}_nm"] = 2 * np.sqrt(2 * np.log(2)) * w * 1e3

    dx = abs(np.diff(x[:2])[0])
    dy = abs(np.diff(y[:2])[0])
    A_eff = float(np.sum(I) ** 2 / np.sum(I**2) * dx * dy)
    results["A_eff_um2"] = A_eff
    results["A_eff_lambda2"] = A_eff / wavelength_um**2
    return results


# ── Mode volume and Purcell factor (notebook section 8) ───────────────────────


def compute_mode_volume(
    data: td.SimulationData,
    monitor_name: str = "fld_3d_box",
    wavelength_um: float = 0.737,
    thickness_um: float = 0.220,
) -> Tuple[float, float]:
    """Compute the effective mode volume V_eff and its normalisation (lambda/2n)^3.

    V_eff = integral(eps |E|^2 dV) / max(eps |E|^2), with the permittivity
    estimated analytically: n_diamond^2 inside the slab (|z| < thickness/2)
    and 1 elsewhere (holes and the triangular sidewalls are not resolved,
    matching the notebook's approximation).

    Returns (V_eff [um^3], V_eff / (lambda/2n)^3).
    """
    mon = data[monitor_name]
    Ex = mon.Ex.isel(f=0).values.squeeze()
    Ey = mon.Ey.isel(f=0).values.squeeze()
    Ez = (
        mon.Ez.isel(f=0).values.squeeze()
        if "Ez" in mon.field_components
        else np.zeros_like(Ex)
    )

    x = _coords_to_um(mon.Ex.coords["x"].values)
    y = _coords_to_um(mon.Ex.coords["y"].values)
    z = _coords_to_um(mon.Ex.coords["z"].values)

    # Ensure field shape (nz, ny, nx)
    I = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2
    if I.shape == (len(x), len(y), len(z)):
        I = np.transpose(I, (2, 1, 0))

    # Analytic permittivity map
    n = float(n_diamond(wavelength_um))
    Z3d = z[:, None, None] * np.ones_like(I)
    eps = np.where(np.abs(Z3d) < thickness_um / 2, n**2, 1.0)

    eps_I = eps * I
    dV = abs(np.diff(x[:2])[0]) * abs(np.diff(y[:2])[0]) * abs(np.diff(z[:2])[0])
    V_eff = float(np.sum(eps_I) * dV / np.max(eps_I))

    lambda_n = wavelength_um / n
    V_norm = V_eff / (lambda_n / 2) ** 3
    return V_eff, V_norm


def compute_purcell(Q: float, V_eff_um3: float, wavelength_um: float) -> float:
    """Purcell enhancement factor F_P = (3 / 4 pi^2) (lambda/n)^3 Q / V_eff."""
    n = float(n_diamond(wavelength_um))
    lam_n3 = (wavelength_um / n) ** 3  # (lambda/n)^3 [um^3]
    return float((3.0 / (4.0 * np.pi**2)) * lam_n3 / V_eff_um3 * Q)


# ── Orchestration ─────────────────────────────────────────────────────────────


def run_full_analysis(
    data_scout: td.SimulationData,
    data_lockin: td.SimulationData,
    res_scout: Dict,
    res_lockin: Dict,
    cavity_bbox: Tuple[float, float, float, float],
    holes_gds: Path,
    thickness_um: float,
    fig_dir: Path,
    show: bool = True,
) -> Dict:
    """Run ringdown, near-field, mode-volume and Purcell analyses.

    Saves ringdown_scout.png / ringdown_lockin.png / nearfield.png into
    ``fig_dir`` and returns a dict with the confinement and Purcell numbers.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    wavelength_um = res_lockin["wavelength_um"]

    # -- Ringdown visualisation
    print("\n-- Ringdown visualisation ------------------------------------")
    wl_scout_nm = res_scout["wavelength_um"] * 1e3
    plot_ringdown(
        data_scout,
        res_scout,
        wavelength_window_nm=(wl_scout_nm - 60.0, wl_scout_nm + 60.0),
        title_prefix="Scout: ",
        save_path=fig_dir / "ringdown_scout.png",
    )
    wl_nm = wavelength_um * 1e3
    plot_ringdown(
        data_lockin,
        res_lockin,
        wavelength_window_nm=(wl_nm - 10.0, wl_nm + 10.0),
        title_prefix="Lock-in: ",
        save_path=fig_dir / "ringdown_lockin.png",
    )

    # -- Near-field mode analysis
    print("\n-- Near-field analysis ---------------------------------------")
    I_raw, x_raw, y_raw = extract_nearfield_intensity(data_lockin)
    I_nf, x_nf, y_nf = crop_field(I_raw, x_raw, y_raw)
    plot_nearfield(
        I_nf, x_nf, y_nf, wl_nm, cavity_bbox, holes_gds,
        save_path=fig_dir / "nearfield.png",
    )
    conf = compute_confinement(I_nf, x_nf, y_nf, wavelength_um)
    print(
        f"  1/e^2 width  w_x   = {conf['w_x_um'] * 1e3:.0f} nm"
        f"  (FWHM = {conf['fwhm_x_nm']:.0f} nm)"
    )
    print(
        f"  1/e^2 width  w_y   = {conf['w_y_um'] * 1e3:.0f} nm"
        f"  (FWHM = {conf['fwhm_y_nm']:.0f} nm)"
    )
    print(
        f"  Eff. mode area     = {conf['A_eff_um2']:.4f} um^2"
        f"  = {conf['A_eff_lambda2']:.3f} lambda^2"
    )

    # -- Mode volume and Purcell factor
    print("\n-- Mode volume and Purcell factor ----------------------------")
    V_eff, V_norm = compute_mode_volume(
        data_lockin, wavelength_um=wavelength_um, thickness_um=thickness_um
    )
    Q_val = res_lockin["Q"]
    F_P = compute_purcell(Q_val, V_eff, wavelength_um)
    print(f"  Effective mode volume  V_eff  = {V_eff:.4f} um^3")
    print(f"                                = {V_norm:.3f} x (lambda/2n)^3")
    print(f"  Quality factor         Q      = {Q_val:.0f}")
    print(f"  Purcell factor         F_P    = {F_P:.1f}")

    if show:
        plt.show()

    return {
        "confinement": conf,
        "V_eff_um3": V_eff,
        "V_norm": V_norm,
        "Q": Q_val,
        "F_P": F_P,
    }
