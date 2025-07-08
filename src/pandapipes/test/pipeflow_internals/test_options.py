# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import pytest

import pandapipes
import pandapipes.pf.pipeflow_setup
from pandapipes.pf.pipeflow_setup import PipeflowNotConverged
from pandapipes.test.pipeflow_internals.test_inservice import create_test_net


@pytest.mark.parametrize("use_numba", [True, False])
def test_set_user_pf_options(create_test_net, use_numba):
    """

    :param create_test_net:
    :type create_test_net:
    :return:
    :rtype:
    """
    net = copy.deepcopy(create_test_net)
    pandapipes.create_fluid_from_lib(net, "lgas")

    max_iter_hyd = 3 if use_numba else 3
    necessary_options = {'mode': 'hydraulics', 'use_numba': use_numba,
                         'max_iter_hyd': max_iter_hyd}
    pandapipes.pipeflow(net, **necessary_options)

    old_options = net._options.copy()
    test_options = {key: i for i, key in enumerate(old_options.keys())}

    pandapipes.pf.pipeflow_setup.set_user_pf_options(net, hello='bye', **test_options)
    test_options.update({'hello': 'bye'})
    test_options.update({'hyd_flag': True})

    assert net.user_pf_options == test_options

    # remove what is in user_pf_options and add hello=world
    pandapipes.pf.pipeflow_setup.set_user_pf_options(net, reset=True, hello='world')
    assert net.user_pf_options == {'hello': 'world'}

    # check if 'hello' is added to net._options, but other options are untouched
    pandapipes.pipeflow(net, **necessary_options)
    assert 'hello' in net._options.keys() and net._options['hello'] == 'world'
    net._options.pop('hello')
    assert net._options == old_options

    # check if user_pf_options can be deleted and net._options is as it was before
    pandapipes.pf.pipeflow_setup.set_user_pf_options(net, reset=True, hello='world')
    pandapipes.pf.pipeflow_setup.set_user_pf_options(net, reset=True)
    assert net.user_pf_options == {}
    pandapipes.pipeflow(net, **necessary_options)
    assert 'hello' not in net._options.keys()

    # see if user arguments overrule user_pf_options, but other user_pf_options still have the
    # priority
    pandapipes.pf.pipeflow_setup.set_user_pf_options(net, reset=True, tol_p=1e-6, tol_m=1e-6)
    max_iter_hyd = 4 if use_numba else 4
    necessary_options.update({'max_iter_hyd': max_iter_hyd})
    pandapipes.pipeflow(net, tol_p=1e-8, **necessary_options)
    assert net.user_pf_options['tol_p'] == 1e-6
    assert net._options['tol_p'] == 1e-8
    assert net._options['tol_m'] == 1e-6


@pytest.mark.parametrize("use_numba", [True, False])
def test_iter(create_test_net, use_numba):
    """

    :param create_test_net:
    :type create_test_net:
    :return:
    :rtype:
    """
    net = copy.deepcopy(create_test_net)
    pandapipes.create_fluid_from_lib(net, "water")

    max_iter_hyd = 3 if use_numba else 3
    max_iter_therm = 3 if use_numba else 3

    pandapipes.set_user_pf_options(net, iter=2)

    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, mode='sequential')

    pandapipes.pipeflow(net, mode='hydraulics', max_iter_hyd=max_iter_hyd)

    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, mode='sequential', max_iter_hyd=max_iter_hyd)

    pandapipes.pipeflow(net, mode='sequential', max_iter_hyd=max_iter_hyd, max_iter_therm=max_iter_therm)

    pandapipes.set_user_pf_options(net, max_iter_hyd=max_iter_hyd)
    pandapipes.pipeflow(net, mode='hydraulics')

    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, mode='sequential')

    pandapipes.set_user_pf_options(net, max_iter_therm=max_iter_therm)

    pandapipes.pipeflow(net, mode='sequential')

    pandapipes.set_user_pf_options(net, max_iter_hyd=max_iter_hyd, max_iter_therm=max_iter_therm)

    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, mode='sequential', iter=2)


if __name__ == '__main__':
    pytest.main(["test_options.py"])
