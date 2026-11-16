# Registered here rather than via addopts in .pytest.toml so the plugin is also active when the
# installed package is tested (``pytest --pyargs pandapipes.test``, ``run_tests()``).
pytest_plugins = ["pandapipes.test.pytest_use_numba"]
