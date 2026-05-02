"""Signal generation service.

Produces the 10-vector sinusoid corpus per HOMEWORK_BRIEF § 3 and
PRD_signal_generation.md.  All public functions are pure (no I/O); the RNG
is the only source of nondeterminism and is seeded explicitly.

Public surface:
    SignalConfig, Corpus  (dataclasses)
    make_clean(config)              → ndarray (4, N) float32
    make_noisy(config, rng)         → ndarray (4, N) float32
    generate_corpus(config, seed)   → Corpus
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from signal_extraction_rnn_lstm.constants import N_SINUSOIDS

_DistName = Literal["gaussian", "uniform"]
_VALID_DISTRIBUTIONS = ("gaussian", "uniform")


@dataclass(frozen=True)
class SignalConfig:
    """Parsed and validated form of ``config['signal']``.

    Input:  individual fields (see PRD_signal_generation.md § 3.1).
    Output: a frozen, validated config; raises ``ValueError`` on invalid input.
    Setup:  validation runs in ``__post_init__``; phases and beta are floats
            (already evaluated upstream by ``shared.config.parse_angle``).
    """

    fs: int
    duration_s: int
    frequencies_hz: tuple[float, ...]
    amplitudes: tuple[float, ...]
    phases_rad: tuple[float, ...]
    noise_alpha: float
    noise_beta: float
    noise_distribution: _DistName

    def __post_init__(self) -> None:
        if self.fs <= 0:
            raise ValueError(f"fs must be > 0, got {self.fs}")
        if self.duration_s <= 0:
            raise ValueError(f"duration_s must be > 0, got {self.duration_s}")
        for name, seq in (("frequencies_hz", self.frequencies_hz),
                          ("amplitudes", self.amplitudes),
                          ("phases_rad", self.phases_rad)):
            if len(seq) != N_SINUSOIDS:
                raise ValueError(f"{name} must have exactly {N_SINUSOIDS} entries")
        nyquist = self.fs / 2
        for f in self.frequencies_hz:
            if f <= 0:
                raise ValueError(f"frequencies must be > 0, got {f}")
            if f >= nyquist:
                raise ValueError(f"frequency {f} >= Nyquist {nyquist}")
        for a in self.amplitudes:
            if a <= 0:
                raise ValueError(f"amplitudes must be > 0, got {a}")
        if self.noise_alpha < 0:
            raise ValueError(f"noise_alpha must be >= 0, got {self.noise_alpha}")
        if self.noise_beta < 0:
            raise ValueError(f"noise_beta must be >= 0, got {self.noise_beta}")
        if self.noise_distribution not in _VALID_DISTRIBUTIONS:
            raise ValueError(
                f"noise_distribution must be one of {_VALID_DISTRIBUTIONS}, "
                f"got {self.noise_distribution!r}"
            )


@dataclass(frozen=True)
class Corpus:
    """The 10-vector signal corpus consumed by ``services.dataset``.

    Input:  produced by ``generate_corpus(config, seed)``.
    Output: this object.
    Setup:  carries SignalConfig + seed for provenance.
    """

    fs: int
    n_samples: int
    frequencies_hz: tuple[float, ...]
    clean: np.ndarray            # (4, N), float32
    noisy: np.ndarray            # (4, N), float32
    clean_sum: np.ndarray        # (N,), float32
    noisy_sum: np.ndarray        # (N,), float32
    config: SignalConfig
    seed: int


def _time_grid(config: SignalConfig) -> np.ndarray:
    n = config.fs * config.duration_s
    return np.arange(n, dtype=np.float64) / config.fs


def _draw_noise(rng: np.random.Generator, dist: _DistName,
                shape: tuple[int, int]) -> np.ndarray:
    if dist == "gaussian":
        return rng.standard_normal(shape)
    half_range = math.sqrt(3.0)  # uniform[-√3, √3] has unit variance
    return rng.uniform(-half_range, half_range, shape)


def make_clean(config: SignalConfig) -> np.ndarray:
    """Return the (4, N) float32 array of clean base sinusoids.

    Input:  SignalConfig.
    Output: ndarray shape (4, N), float32. Pure, deterministic.
    Setup:  no RNG; same config → same output.
    """
    t = _time_grid(config)
    out = np.empty((N_SINUSOIDS, t.size), dtype=np.float64)
    for i in range(N_SINUSOIDS):
        out[i] = config.amplitudes[i] * np.sin(
            2 * np.pi * config.frequencies_hz[i] * t + config.phases_rad[i]
        )
    return out.astype(np.float32)


def make_noisy(config: SignalConfig, rng: np.random.Generator) -> np.ndarray:
    """Return the (4, N) float32 array of noisy base sinusoids.

    Input:  SignalConfig and a numpy ``Generator``.
    Output: ndarray shape (4, N), float32.
    Setup:
        Two unit-variance noise tensors of shape (4, N) are drawn in fixed
        order — amplitude first, phase second.  Same RNG state ⇒ bitwise-
        identical output.  Per-sample formula::

            s_tilde[i, t] = (A_i + alpha*eps_A[i, t])
                            * sin(2π f_i t + phi_i + beta*eps_phi[i, t])
    """
    t = _time_grid(config)
    shape = (N_SINUSOIDS, t.size)
    eps_a = _draw_noise(rng, config.noise_distribution, shape)
    eps_phi = _draw_noise(rng, config.noise_distribution, shape)
    out = np.empty(shape, dtype=np.float64)
    for i in range(N_SINUSOIDS):
        amp = config.amplitudes[i] + config.noise_alpha * eps_a[i]
        phase = (2 * np.pi * config.frequencies_hz[i] * t
                 + config.phases_rad[i] + config.noise_beta * eps_phi[i])
        out[i] = amp * np.sin(phase)
    return out.astype(np.float32)


def parse_signal_config(d: dict) -> SignalConfig:
    """Build a validated ``SignalConfig`` from ``config['signal']`` dict.

    Phase strings (``"pi/2"`` etc.) and ``noise.beta`` are evaluated through
    ``shared.config.parse_angle``; numeric values pass through unchanged.
    """
    from signal_extraction_rnn_lstm.shared.config import parse_angle
    noise = d["noise"]
    beta = parse_angle(noise["beta"]) if isinstance(noise["beta"], str) else float(noise["beta"])
    phases = tuple(
        parse_angle(p) if isinstance(p, str) else float(p) for p in d["phases_rad"]
    )
    return SignalConfig(
        fs=int(d["fs"]),
        duration_s=int(d["duration_s"]),
        frequencies_hz=tuple(float(f) for f in d["frequencies_hz"]),
        amplitudes=tuple(float(a) for a in d["amplitudes"]),
        phases_rad=phases,
        noise_alpha=float(noise["alpha"]),
        noise_beta=beta,
        noise_distribution=noise["distribution"],
    )


def generate_corpus(config: SignalConfig, seed: int) -> Corpus:
    """Build the full 10-vector ``Corpus``.

    Input:  SignalConfig, integer seed (None forbidden — seeding is mandatory).
    Output: Corpus with clean, noisy, and both summed signals.
    Setup:  ``np.random.default_rng(seed)``; ``make_clean`` then ``make_noisy``;
            sums; bundles provenance.
    """
    if seed is None or not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be a non-None int, got {type(seed).__name__}")
    rng = np.random.default_rng(seed)
    clean = make_clean(config)
    noisy = make_noisy(config, rng)
    return Corpus(
        fs=config.fs,
        n_samples=clean.shape[1],
        frequencies_hz=tuple(config.frequencies_hz),
        clean=clean,
        noisy=noisy,
        clean_sum=clean.sum(axis=0),
        noisy_sum=noisy.sum(axis=0),
        config=config,
        seed=seed,
    )
