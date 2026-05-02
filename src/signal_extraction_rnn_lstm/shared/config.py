"""Configuration loader.

Loads and validates config/setup.json; parses angle-expression strings;
enforces schema version compatibility. See PLAN.md § 9.

Public surface:
    load_config(config_path) → dict
    parse_angle(expr)        → float
    ConfigVersionMismatch    (exception)
"""


class ConfigVersionMismatchError(Exception):
    """Raised when the JSON config version is incompatible with the code."""


def load_config(config_path: object = None) -> dict:
    """Load and validate setup.json.

    Args:
        config_path: Path to setup.json. Defaults to config/setup.json
            relative to the project root.

    Returns:
        Validated config dict with angle strings resolved to floats.

    Raises:
        ConfigVersionMismatchError: if ``config["version"]`` is incompatible.
        ValueError: if required fields are missing or out of range.
        FileNotFoundError: if config_path does not exist.
    """
    raise NotImplementedError("M2")


def parse_angle(expr: str) -> float:
    """Parse a string angle expression to a float in radians.

    Accepted tokens: numeric literals, 'pi', integer/decimal coefficients,
    '*', '/', and parentheses. Examples::

        parse_angle("0")       → 0.0
        parse_angle("pi/2")    → 1.5707963...
        parse_angle("2*pi")    → 6.2831853...
        parse_angle("3*pi/2")  → 4.7123889...

    See PLAN.md § 9 for the full grammar.

    Raises:
        ValueError: if expr contains unsupported tokens.
    """
    raise NotImplementedError("M2")
