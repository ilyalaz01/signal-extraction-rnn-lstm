"""RED-phase tests for services.signal_gen — T-SG-01 through T-SG-14.

All tests fail until services/signal_gen.py and shared/config.py are
implemented. Covers: shapes, dtype, determinism, seed sensitivity,
noise model correctness, sum identities, FFT frequency content,
config validation, provenance, per-sample independence, and parse_angle.
"""

import math

import numpy as np
import pytest

from signal_extraction_rnn_lstm.services.signal_gen import (
    Corpus,
    SignalConfig,
    generate_corpus,
    make_clean,
    make_noisy,
)
from signal_extraction_rnn_lstm.shared.config import parse_angle

_FREQS = (2.0, 10.0, 50.0, 200.0)
_AMPS = (1.0, 1.0, 1.0, 1.0)
_PHASES = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)


def _cfg_with(**kw) -> SignalConfig:
    """Build a SignalConfig from defaults, overriding any field via kw."""
    defaults: dict = {
        "fs": 1000, "duration_s": 10, "frequencies_hz": _FREQS,
        "amplitudes": _AMPS, "phases_rad": _PHASES,
        "noise_alpha": 0.05, "noise_beta": 2 * math.pi,
        "noise_distribution": "gaussian",
    }
    return SignalConfig(**{**defaults, **kw})


@pytest.fixture
def cfg() -> SignalConfig:
    return _cfg_with()


@pytest.fixture
def corpus(cfg: SignalConfig) -> Corpus:
    return generate_corpus(cfg, seed=42)


# ---------------------------------------------------------------------------
# T-SG-01  Shapes and dtype
# ---------------------------------------------------------------------------

def test_t_sg_01_shapes_and_dtype(corpus: Corpus) -> None:
    assert corpus.clean.shape == (4, 10_000)
    assert corpus.noisy.shape == (4, 10_000)
    assert corpus.clean_sum.shape == (10_000,)
    assert corpus.noisy_sum.shape == (10_000,)
    for arr in (corpus.clean, corpus.noisy, corpus.clean_sum, corpus.noisy_sum):
        assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# T-SG-02  Determinism — same (config, seed) → bitwise-identical corpus
# ---------------------------------------------------------------------------

def test_t_sg_02_deterministic(cfg: SignalConfig) -> None:
    c1, c2 = generate_corpus(cfg, seed=42), generate_corpus(cfg, seed=42)
    np.testing.assert_array_equal(c1.clean, c2.clean)
    np.testing.assert_array_equal(c1.noisy, c2.noisy)


# ---------------------------------------------------------------------------
# T-SG-03  Seed sensitivity — different seeds → different noisy arrays
# ---------------------------------------------------------------------------

def test_t_sg_03_seed_sensitivity(cfg: SignalConfig) -> None:
    c1, c2 = generate_corpus(cfg, seed=42), generate_corpus(cfg, seed=99)
    assert float(np.mean(np.abs(c1.noisy - c2.noisy))) > 0


# ---------------------------------------------------------------------------
# T-SG-04  alpha=0 ⇒ amplitude envelope bounded by A_i exactly
# ---------------------------------------------------------------------------

def test_t_sg_04_alpha_zero_bounded() -> None:
    c = generate_corpus(_cfg_with(noise_alpha=0.0), seed=42)
    for i in range(4):
        assert float(np.abs(c.noisy[i]).max()) <= _AMPS[i] + 1e-5


# ---------------------------------------------------------------------------
# T-SG-05  alpha=0, beta=0 ⇒ noisy == clean (fp-exact)
# ---------------------------------------------------------------------------

def test_t_sg_05_zero_noise_equals_clean() -> None:
    c = generate_corpus(_cfg_with(noise_alpha=0.0, noise_beta=0.0), seed=42)
    np.testing.assert_array_equal(c.noisy, c.clean)


# ---------------------------------------------------------------------------
# T-SG-06  Clean sum identity  S = s1 + s2 + s3 + s4
# ---------------------------------------------------------------------------

def test_t_sg_06_clean_sum_identity(corpus: Corpus) -> None:
    np.testing.assert_array_equal(corpus.clean_sum, corpus.clean.sum(axis=0))


# ---------------------------------------------------------------------------
# T-SG-07  Noisy sum identity  S̃ = s̃1 + s̃2 + s̃3 + s̃4  (brief's key invariant)
# ---------------------------------------------------------------------------

def test_t_sg_07_noisy_sum_identity(corpus: Corpus) -> None:
    np.testing.assert_array_equal(corpus.noisy_sum, corpus.noisy.sum(axis=0))


# ---------------------------------------------------------------------------
# T-SG-08  Default config: noisy ≠ clean
# ---------------------------------------------------------------------------

def test_t_sg_08_noisy_differs_from_clean(corpus: Corpus) -> None:
    assert not np.allclose(corpus.noisy, corpus.clean)


# ---------------------------------------------------------------------------
# T-SG-09  FFT smoke — dominant peak within 1 bin of target frequency
# ---------------------------------------------------------------------------

def test_t_sg_09_frequency_content(corpus: Corpus) -> None:
    n = corpus.clean.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / 1000)
    bin_hz = 1000.0 / n
    for i, f_target in enumerate(_FREQS):
        spectrum = np.abs(np.fft.rfft(corpus.clean[i]))
        assert abs(float(freqs[np.argmax(spectrum)]) - f_target) <= bin_hz


# ---------------------------------------------------------------------------
# T-SG-10  Phases stored as floats, not strings; round-trip through sin
# ---------------------------------------------------------------------------

def test_t_sg_10_phases_are_floats(corpus: Corpus) -> None:
    for p in corpus.config.phases_rad:
        assert isinstance(p, float)
    assert math.isclose(math.sin(corpus.config.phases_rad[1]), 1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# parse_angle unit tests (feeds T-SG-10 / T-SG-11)
# ---------------------------------------------------------------------------

def test_parse_angle_basic() -> None:
    assert parse_angle("0") == pytest.approx(0.0)
    assert parse_angle("pi/2") == pytest.approx(math.pi / 2)
    assert parse_angle("2*pi") == pytest.approx(2 * math.pi)
    assert parse_angle("3*pi/2") == pytest.approx(3 * math.pi / 2)


def test_parse_angle_rejects_unsafe() -> None:
    with pytest.raises(ValueError):
        parse_angle("__import__('os')")
    with pytest.raises(ValueError):
        parse_angle("open('/etc/passwd')")
    with pytest.raises(ValueError):
        parse_angle(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        parse_angle("((1+2)")  # passes whitelist, fails eval (unbalanced parens)


# ---------------------------------------------------------------------------
# T-SG-11  Config validation — invalid inputs raise ValueError / TypeError
# ---------------------------------------------------------------------------

def test_t_sg_11_config_validation() -> None:
    with pytest.raises(ValueError):
        _cfg_with(frequencies_hz=(2.0, 10.0, 50.0))              # only 3
    with pytest.raises(ValueError):
        _cfg_with(noise_alpha=-0.1)                               # negative
    with pytest.raises(ValueError):
        _cfg_with(noise_distribution="laplace")                   # unknown
    with pytest.raises(ValueError):
        _cfg_with(frequencies_hz=(2.0, 10.0, 50.0, 600.0))       # > Nyquist
    with pytest.raises(ValueError):
        _cfg_with(frequencies_hz=(-1.0, 10.0, 50.0, 200.0))      # negative freq
    with pytest.raises(ValueError):
        _cfg_with(fs=0)                                            # fs <= 0
    with pytest.raises(ValueError):
        _cfg_with(duration_s=0)                                    # duration <= 0
    with pytest.raises(ValueError):
        _cfg_with(amplitudes=(-1.0, 1.0, 1.0, 1.0))               # negative amp
    with pytest.raises(ValueError):
        _cfg_with(amplitudes=(1.0, 1.0, 1.0))                     # only 3 amps
    with pytest.raises(ValueError):
        _cfg_with(noise_beta=-0.5)                                 # negative beta


def test_uniform_distribution_smoke() -> None:
    """Exercises the rng.uniform branch in _draw_noise (T-SG-11 § 9 edge case)."""
    c = generate_corpus(_cfg_with(noise_distribution="uniform"), seed=0)
    assert c.noisy.shape == (4, 10_000)
    assert not np.allclose(c.noisy, c.clean)


def test_seed_none_raises(cfg: SignalConfig) -> None:
    with pytest.raises(TypeError):
        generate_corpus(cfg, seed=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# T-SG-12  Provenance — corpus echoes its inputs
# ---------------------------------------------------------------------------

def test_t_sg_12_provenance(cfg: SignalConfig) -> None:
    c = generate_corpus(cfg, seed=42)
    assert c.seed == 42
    assert c.config == cfg


# ---------------------------------------------------------------------------
# T-SG-13  Per-sample independence — amplitude residual std matches theory
# ---------------------------------------------------------------------------

def test_t_sg_13_per_sample_noise() -> None:
    # beta=0 isolates amplitude noise; expected: std(residual) ≈ alpha/√2
    # because eps_A ~ N(0,1), sin^2 time-average = 1/2 over complete cycles
    c = generate_corpus(_cfg_with(noise_alpha=0.05, noise_beta=0.0), seed=42)
    residual = c.noisy.astype(np.float64) - c.clean.astype(np.float64)
    for i in range(4):
        expected_std = 0.05 * _AMPS[i] / math.sqrt(2)
        np.testing.assert_allclose(float(residual[i].std()), expected_std, rtol=0.15)


# ---------------------------------------------------------------------------
# T-SG-14  make_noisy with zero noise returns clean signal byte-identical
# ---------------------------------------------------------------------------

def test_t_sg_14_make_noisy_zero_noise_identical() -> None:
    cfg_zero = _cfg_with(noise_alpha=0.0, noise_beta=0.0)
    rng = np.random.default_rng(42)
    np.testing.assert_array_equal(make_noisy(cfg_zero, rng), make_clean(cfg_zero))


# ---------------------------------------------------------------------------
# Property tests (§ 8.2) — FFT peak holds for multiple seeds and frequencies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("i,f_hz", list(enumerate(_FREQS)))
def test_prop_fft_peak_any_seed(i: int, f_hz: float, seed: int) -> None:
    c = generate_corpus(_cfg_with(noise_alpha=0.0, noise_beta=0.0), seed=seed)
    freqs = np.fft.rfftfreq(c.clean.shape[1], d=1.0 / 1000)
    peak = float(freqs[np.argmax(np.abs(np.fft.rfft(c.clean[i])))])
    assert abs(peak - f_hz) <= 1000.0 / c.clean.shape[1]
