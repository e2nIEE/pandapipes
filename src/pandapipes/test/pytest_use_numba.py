"""Parametrize any test requesting the `use_numba` fixture.

Loaded via ``addopts`` in ``.pytest.toml``. Every ``use_numba`` test runs twice by
default (ids ``[numba]`` / ``[numpy]``); ``--only=numba|numpy`` restricts to one mode.
"""
import pytest

_CASES = {
    "numba": pytest.param(True, id="numba"),
    "numpy": pytest.param(False, id="numpy"),
}


def pytest_addoption(parser):
    parser.addoption(
        "--only",
        choices=list(_CASES),
        default=None,
        help="run only the 'numba' (use_numba=True) or 'numpy' (use_numba=False) parametrization; "
        "omit to run both",
    )


def pytest_generate_tests(metafunc):
    """Parametrize the `use_numba` fixture indirectly, so the fixture body stays single."""
    if "use_numba" not in metafunc.fixturenames:
        return

    only = metafunc.config.getoption("--only")
    params = [_CASES[only]] if only else list(_CASES.values())
    metafunc.parametrize("use_numba", params, indirect=True)


@pytest.fixture
def use_numba(request):
    return request.param
