# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import io
import os
import re

from setuptools import find_packages
from setuptools import setup

with open('README.rst', 'rb') as f:
    install = f.read().decode('utf-8')

with open('CHANGELOG.rst', 'rb') as f:
    changelog = f.read().decode('utf-8')

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6']

long_description = '\n\n'.join((install, changelog))

setup(
    name='pandapipes',
    version='1.0.1',
    author='Dennis Cronbach, Daniel Lohmeier, Simon Ruben Drauz',
    author_email='dennis.cronbach@iee.fraunhofer.de, daniel.lohmeier@iee.fraunhofer.de, '
                 'simon.ruben.drauz@iee.fraunhofer.de',
    description='Convenient Power System Modelling and Analysis based on PYPOWER and pandas',
    long_description=long_description,
    url='http://www.pandapipes.org',
    license='BSD',
    install_requires=["pandapower>=2.0"],
    extras_require={"docs": ["numpydoc", "sphinx", "sphinx_rtd_theme", "sphinxcontrib.bibtex"]},
    python_requires='>=3, <4',
    packages=find_packages(),
    include_package_data=True,
    classifiers=classifiers
)
