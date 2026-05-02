#!/usr/bin/env bash
# check.sh — local CI gate.
# Usage: bash scripts/check.sh
#
# Runs ruff (zero violations required) then pytest + coverage.
# When no test_*.py files exist yet (M1 scaffold), the coverage-fail-under
# threshold is skipped — pytest-cov still instruments the source but
# reports 0 %, which is expected on a stub-only tree.
set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Lint
# ---------------------------------------------------------------------------
echo "========================================"
echo "  ruff check"
echo "========================================"
uv run ruff check src/ tests/
echo "ruff: OK"
echo ""

# ---------------------------------------------------------------------------
# 2. Tests + coverage
# ---------------------------------------------------------------------------
echo "========================================"
echo "  pytest + coverage"
echo "========================================"

# Count real test files (test_*.py anywhere under tests/).
TEST_FILES=$(find tests/ -name "test_*.py" 2>/dev/null | wc -l)

if [ "$TEST_FILES" -eq 0 ]; then
    # M1 / scaffold mode: run pytest for structural sanity but override the
    # 85 % threshold (nothing was executed, so coverage would be 0 %).
    # --cov-fail-under=0 overrides [tool.coverage.report].fail_under from pyproject.toml.
    # set +e because pytest exits 5 when 0 tests are collected.
    set +e
    uv run pytest --cov=src --cov-fail-under=0 -q
    PYTEST_EXIT=$?
    set -e
    # Accept 0 (tests passed) or 5 (no tests collected); anything else is real failure.
    if [ "$PYTEST_EXIT" -ne 0 ] && [ "$PYTEST_EXIT" -ne 5 ]; then
        echo "pytest: unexpected error (exit $PYTEST_EXIT)" >&2
        exit "$PYTEST_EXIT"
    fi
    echo ""
    echo "No test files found — coverage threshold skipped (M1 scaffold)."
    echo "check.sh: PASSED"
    exit 0
fi

# M2+ mode: full enforcement.
uv run pytest --cov=src --cov-fail-under=85
echo "pytest: OK"
echo ""
echo "check.sh: PASSED"
