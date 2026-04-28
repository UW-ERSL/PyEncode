"""pytest configuration for the PyEncode test suite.

Registers custom markers used in test_pyencode.py and filters expected
``UserWarning`` messages so they do not clutter the ``pytest`` output.
"""


def pytest_configure(config):
    # ---- custom markers ----------------------------------------------------
    config.addinivalue_line(
        "markers",
        "slow: opt-in deep-coverage tests at large m (m >= 12).  "
        "Skipped by default; run with `pytest -m slow` to include them.",
    )

    # ---- expected-warning filters ------------------------------------------
    # The test suite intentionally exercises overlapping-support SUM cases
    # (the API contract documented in Section 5.1 of the paper requires
    # the SUM constructor to emit a UserWarning whenever components
    # overlap, since post-selection overhead is non-trivial).  These are
    # not test failures and not regressions; suppress them at the
    # conftest level so the run output stays clean.  The match is narrow
    # — only the exact "overlapping support" message is filtered, so any
    # *other* UserWarning emitted by future code changes will still be
    # surfaced.
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:SUM components have overlapping support.*:UserWarning",
    )
    # qiskit-ibm-runtime emits a deprecation warning every time the
    # stevedore plugin loader instantiates IBMFractionalTranslationPlugin,
    # which happens once per transpile() call.  Not from PyEncode code,
    # not actionable from this side; suppress to keep the output readable.
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:.*IBMFractionalTranslationPlugin is deprecated.*:DeprecationWarning",
    )