# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.


import importlib.util
import os

import pytest
from pandapipes import pp_dir
try:
    from setuptools import find_packages
    SETUPTOOLS_AVAILABLE = True
except ImportError:
    SETUPTOOLS_AVAILABLE = False


@pytest.mark.skipif(not SETUPTOOLS_AVAILABLE, reason="setuptools not available")
def test_import_packages():
    all_packages = find_packages(pp_dir)
    for pck in all_packages:
        spec = importlib.util.find_spec(os.path.split(pp_dir)[-1] + "." + pck)
        new_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(new_module)
    assert True


if __name__ == '__main__':
    pytest.main(["test_imports.py"])
