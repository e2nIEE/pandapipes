# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import re

from setuptools import find_namespace_packages
from setuptools import setup

with open('README.rst', 'rb') as f:
    install = f.read().decode('utf-8')

with open('CHANGELOG.rst', 'rb') as f:
    changelog = f.read().decode('utf-8')

classifiers = [
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3']

with open('.github/workflows/run_tests_master.yml', 'rb') as f:
    lines = f.read().decode('utf-8')
    versions = set(re.findall('3.[8-9]', lines)) | set(re.findall('3.1[0-9]', lines))
    for version in versions:
        classifiers.append('Programming Language :: Python :: %s' % version)

long_description = '\n\n'.join((install, changelog))

setup(
    name='pandapipes',
    version='0.10.0',
    author='Simon Ruben Drauz-Mauel, Daniel Lohmeier, Jolando Marius Kisse',
    author_email='simon.ruben.drauz-mauel@iee.fraunhofer.de, daniel.lohmeier@retoflow.de, '
                 'jolando.kisse@uni-kassel.de',
    description='A pipeflow calculation tool that complements pandapower in the simulation of multi energy grids',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='http://www.pandapipes.org',
    license='BSD',
    install_requires=["pandapower>=2.14.6", "matplotlib", "shapely"],
    extras_require={"docs": ["numpydoc", "sphinx", "sphinx_rtd_theme", "sphinxcontrib.bibtex"],
                    "plotting": ["plotly", "igraph"],
                    "test": ["pytest", "pytest-xdist", "nbmake"],
                    "all": ["numpydoc", "sphinx", "sphinx_rtd_theme", "sphinxcontrib.bibtex",
                            "plotly", "igraph", "pytest", "pytest-xdist", "nbmake"]},
    packages=find_namespace_packages(where='src'),
    package_dir={"": "src"},
    include_package_data=True,
    classifiers=classifiers
)
