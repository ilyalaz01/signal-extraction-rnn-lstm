"""Project-wide physical constants fixed by HOMEWORK_BRIEF.md § 3.3.

These values MUST NOT be read from config. Config-mutable parameters
(frequencies, amplitudes, phases, noise strengths) live in config/setup.json
and are loaded by shared.config.
"""

# Sampling parameters
SAMPLING_RATE_HZ: int = 1000
DURATION_S: int = 10
N_SAMPLES: int = SAMPLING_RATE_HZ * DURATION_S  # 10 000

# Signal model
N_SINUSOIDS: int = 4

# Dataset — fixed by HOMEWORK_BRIEF.md § 4.2
WINDOW_SIZE: int = 10

# Derived I/O sizes (per HOMEWORK_BRIEF.md § 5.1 and ADR-003)
FC_INPUT_SIZE: int = N_SINUSOIDS + WINDOW_SIZE  # 14 = 4 selector + 10 window
SEQ_FEATURE_SIZE: int = 1 + N_SINUSOIDS         # 5  = 1 sample  + 4 broadcast selector
OUTPUT_SIZE: int = WINDOW_SIZE                   # 10 predicted samples
