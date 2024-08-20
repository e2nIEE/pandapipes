# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
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


def test_import_packages():
    if SETUPTOOLS_AVAILABLE:
        all_packages = find_packages(pp_dir)
    else:
        all_packages = []
        for root, dirs, files in os.walk(pp_dir):
            for file in files:
                if file != "__init__.py":
                    continue
                path = os.path.relpath(root, pp_dir)
                if path == ".":
                    continue
                pck = path.replace(os.sep, ".")
                all_packages.append(pck)

    for pck in all_packages:
        spec = importlib.util.find_spec(os.path.split(pp_dir)[-1] + "." + pck)
        new_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(new_module)
    assert True


if __name__ == '__main__':
    pytest.main(["test_imports.py"])
