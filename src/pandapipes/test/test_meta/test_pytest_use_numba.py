"""Meta-tests for ``pytest_use_numba``.

Verify that ``--only=numba|numpy`` actually changes what is collected. These guard
against the plugin regressing to "collect both modes regardless of the flag", which
would leave every test green while the flag silently does nothing. That is the one
failure mode the normal suite cannot surface on its own, so ``--collect-only`` is used
to inspect parametrization without running the fixture.
"""
pytest_plugins = ["pytester"]

PLUGIN = "pandapipes.test.pytest_use_numba"


def _collect(pytester, *args):
    pytester.makepyfile("""
        def test_it(use_numba):
            pass
    """)
    result = pytester.runpytest("-p", PLUGIN, "--collect-only", "-q", *args)
    return result.stdout.str()


def test_default_collects_both(pytester):
    out = _collect(pytester)
    assert "test_it[numba]" in out
    assert "test_it[numpy]" in out


def test_only_numba_collects_one(pytester):
    out = _collect(pytester, "--only=numba")
    assert "test_it[numba]" in out
    assert "test_it[numpy]" not in out


def test_only_numpy_collects_one(pytester):
    out = _collect(pytester, "--only=numpy")
    assert "test_it[numpy]" in out
    assert "test_it[numba]" not in out
