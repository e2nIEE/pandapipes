import pytest
import pandapipes.pf.friction_factor_model as fm

# Registered here rather than via addopts in .pytest.toml so the plugin is also active when the
# installed package is tested (``pytest --pyargs pandapipes.test``, ``run_tests()``).
pytest_plugins = ["pandapipes.test.pytest_use_numba"]


@pytest.fixture(
    params=[
        fm.Nikuradse,
        fm.SwameeJain,
        fm.Colebrook,
    ]
)
def model_class(request):
    """Fixture that yields each friction‑factor model class."""
    return request.param
