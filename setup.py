# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from setuptools import find_packages
from setuptools import setup
import re

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
    versions = set(re.findall('3.[7-9]', lines))
    for version in versions:
        classifiers.append('Programming Language :: Python :: 3.%s' % version[-1])

long_description = '\n\n'.join((install, changelog))

setup(
    name='pandapipes',
    version='0.7.0',
    author='Dennis Cronbach, Daniel Lohmeier, Simon Ruben Drauz, Jolando Marius Kisse',
    author_email='dennis.cronbach@iee.fraunhofer.de, daniel.lohmeier@iee.fraunhofer.de, '
                 'simon.ruben.drauz@iee.fraunhofer.de, jolando.kisse@uni-kassel.de',
    description='A pipeflow calculation tool that complements pandapower in the simulation of multi energy grids',
    long_description=long_description,
	long_description_content_type='text/x-rst',
    url='http://www.pandapipes.org',
    license='BSD',
    install_requires=["pandapower>=2.10.1", "matplotlib"],
    extras_require={"docs": ["numpydoc", "sphinx", "sphinx_rtd_theme", "sphinxcontrib.bibtex"],
                    "plotting": ["plotly", "python-igraph"],
                    "test": ["pytest", "pytest-xdist", "nbmake"],
                    "all": ["numpydoc", "sphinx", "sphinx_rtd_theme", "sphinxcontrib.bibtex",
                            "plotly", "python-igraph", "pytest", "pytest-xdist", "nbmake"]},
    packages=find_packages(),
    include_package_data=True,
    classifiers=classifiers
)
