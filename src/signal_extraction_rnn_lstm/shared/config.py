"""Configuration loader.

Loads and validates ``config/setup.json``; parses angle-expression strings;
enforces schema version compatibility. See PLAN.md § 9.

Public surface:
    load_config(config_path) → dict
    parse_angle(expr)        → float
    ConfigVersionMismatchError    (exception)
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

_EXPECTED_VERSION = "1.00"
_ANGLE_SAFE_RE = re.compile(r"^[\d.\*\/\+\-\(\)\s]+$")


class ConfigVersionMismatchError(Exception):
    """Raised when the JSON config version is incompatible with the code."""


def parse_angle(expr: str) -> float:
    """Parse an angle expression to a float in radians.

    Accepted tokens: numeric literals, the literal ``pi``, ``+ - * /``,
    parentheses, and whitespace.  Examples::

        parse_angle("0")       → 0.0
        parse_angle("pi/2")    → 1.5707963...
        parse_angle("2*pi")    → 6.2831853...
        parse_angle("3*pi/2")  → 4.7123889...

    Safety: ``pi`` is replaced with ``str(math.pi)`` and the resulting string
    is matched against a strict whitelist that admits only digits, the
    decimal point, the four arithmetic operators, parentheses, and
    whitespace.  An expression that survives the whitelist contains no
    identifiers, no function calls, and no attribute access — so passing it
    to ``eval`` cannot invoke arbitrary code.  The whitelist regex is the
    security boundary; ``eval`` is run only on a verified-safe expression.

    Raises:
        ValueError: if expr is not a string, contains unsupported tokens,
            or fails to evaluate to a number.
    """
    if not isinstance(expr, str):
        raise ValueError(f"angle expression must be a string, got {type(expr).__name__}")
    cleaned = expr.replace("pi", str(math.pi))
    if not _ANGLE_SAFE_RE.match(cleaned):
        raise ValueError(f"unsafe or unsupported tokens in angle expression: {expr!r}")
    try:
        result = eval(cleaned)  # noqa: S307 — input passed strict whitelist above
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"failed to evaluate angle expression {expr!r}") from exc
    return float(result)


def load_config(config_path: Path | str | None = None) -> dict:
    """Load and validate ``setup.json``.

    Args:
        config_path: path to setup.json. Defaults to
            ``<project_root>/config/setup.json`` (resolved relative to this file).

    Returns:
        Raw config dict — angle strings remain strings; consumers (typically
        ``services.signal_gen``) evaluate them via ``parse_angle``.

    Raises:
        ConfigVersionMismatchError: if ``config["version"]`` is incompatible.
        FileNotFoundError: if ``config_path`` does not exist.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[3] / "config" / "setup.json"
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    version = cfg.get("version")
    if version != _EXPECTED_VERSION:
        raise ConfigVersionMismatchError(
            f"config version {version!r} != expected {_EXPECTED_VERSION!r}"
        )
    return cfg
