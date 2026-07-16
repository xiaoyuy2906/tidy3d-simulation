"""Cavity Q-factor extraction from the time-domain ringdown.

Implements the Stage-5 workflow of ``examples/DiamondPhotonicCrystalCavity.ipynb``:
the Tidy3D ``ResonanceFinder`` (matrix-pencil / Prony method) decomposes the
probe signal E_y(t) into a sum of decaying sinusoids

    E_y(t) ~ sum_k  A_k  exp(-(omega_k / 2 Q_k) t)  cos(omega_k t + phi_k)

and returns the dominant cavity resonance (frequency, wavelength, Q, decay time).
"""

from typing import Dict, Optional

import numpy as np
import tidy3d as td
from tidy3d.plugins.resonance import ResonanceFinder

from siv_cavity.config import C0_M_PER_S


def extract_resonance(
    data: td.SimulationData,
    monitor_name: str = "probe",
    wavelength_centre_um: Optional[float] = None,
    bandwidth_rel: float = 0.12,
    q_min: float = 1e3,
    q_max: float = 1e9,
    init_num_freqs: int = 200,
    rcond: float = 1e-6,
    max_error: float = 1.0,
    field: str = "Ey",
) -> Dict:
    """Extract the dominant cavity resonance from a ringdown time monitor.

    Parameters
    ----------
    data : td.SimulationData
        Simulation data containing a ``FieldTimeMonitor`` named ``monitor_name``.
    wavelength_centre_um : float, optional
        Source centre wavelength [um]; sets the frequency search window.
    bandwidth_rel : float
        Relative width of the frequency search window (+/- bandwidth * f0).
    q_min, q_max : float
        Q-factor acceptance window (filters spurious low-Q source transients).
    init_num_freqs, rcond : ResonanceFinder hyper-parameters.
    max_error : float
        Maximum ResonanceFinder fit error to accept.

    Returns
    -------
    dict with keys: freq_Hz, wavelength_um, wavelength_nm, Q, decay_time_ps,
    amplitude, error, and the full (filtered) dataframe under 'dataframe'.
    """
    if wavelength_centre_um is None:
        raise ValueError("wavelength_centre_um must be provided.")

    mon = data[monitor_name]
    signal = getattr(mon, field).values.squeeze().astype(complex)
    t = getattr(mon, field).coords["t"].values
    dt = float(t[1] - t[0])

    # Trim the source transient before fitting: keep t >= 2 x source decay
    # (offset + pulse time, cf. examples/NanobeamCavity.ipynb) so the
    # ResonanceFinder only sees the clean exponential ringdown.
    if data.simulation.sources:
        src_time = data.simulation.sources[0].source_time
        t_off = src_time.offset / (2 * np.pi * src_time.fwidth)
        t_start = 2.0 * (t_off + 0.44 / src_time.fwidth)
        keep = t >= t_start
        if 16 < np.count_nonzero(keep) < len(t):
            signal = signal[keep]
            t = t[keep]

    f0 = C0_M_PER_S / (wavelength_centre_um * 1e-6)
    fwidth = f0 * bandwidth_rel
    freq_window = (f0 - fwidth, f0 + fwidth)

    rf = ResonanceFinder(
        freq_window=freq_window, init_num_freqs=init_num_freqs, rcond=rcond
    )
    res = rf.run_raw_signal(signal=signal, time_step=dt)
    df = res.to_dataframe()

    # Keep physical, well-fit, high-Q modes only.
    df = df[(df.index > 0) & (df["Q"] > q_min) & (df["Q"] < q_max)]
    if "error" in df.columns:
        df = df[df["error"] < max_error]
    df = df.sort_values("amplitude", ascending=False)

    if df.empty:
        tail = float(np.abs(signal[-1]))
        peak = float(np.max(np.abs(signal)))
        raise RuntimeError(
            "ResonanceFinder found no valid modes in the Q window "
            f"[{q_min:.1e}, {q_max:.1e}]. tail/peak={tail / max(peak, 1e-30):.2e}. "
            "Try a longer run_time, a different bandwidth, or widen the Q window."
        )

    freq = float(df.index[0])
    Q = float(df["Q"].iloc[0])
    wl_um = (C0_M_PER_S / freq) * 1e6
    tau_ps = Q / (np.pi * freq) * 1e12
    return {
        "freq_Hz": freq,
        "wavelength_um": wl_um,
        "wavelength_nm": wl_um * 1e3,
        "Q": Q,
        "decay_time_ps": tau_ps,
        "amplitude": float(df["amplitude"].iloc[0]),
        "error": float(df["error"].iloc[0]) if "error" in df.columns else float("nan"),
        "dataframe": df,
    }


def print_resonance(res: Dict, label: str = "Q-factor extraction") -> None:
    print(f"\n-- {label} --")
    print(f"  Resonance wavelength : {res['wavelength_nm']:.3f} nm")
    print(f"  Quality factor  Q    : {res['Q']:.3e}")
    print(f"  Amplitude decay time : {res['decay_time_ps']:.2f} ps (energy: {res['decay_time_ps'] / 2:.2f} ps)")
    print(f"  Fit error            : {res['error']:.2e}")
