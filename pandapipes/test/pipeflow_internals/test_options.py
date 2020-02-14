# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes.pipeflow_setup
import pandapipes
import copy
import pytest
from pandapipes.test.pipeflow_internals.test_inservice import create_test_net


def test_set_user_pf_options(create_test_net):
    """

    :param create_test_net:
    :type create_test_net:
    :return:
    :rtype:
    """
    net = copy.deepcopy(create_test_net)
    pandapipes.create_fluid_from_lib(net, "lgas")

    necessary_options = {'mode': 'hydraulics'}
    pandapipes.pipeflow(net, **necessary_options)

    old_options = net._options.copy()
    test_options = {key: i for i, key in enumerate(old_options.keys())}

    pandapipes.pipeflow_setup.set_user_pf_options(net, hello='bye', **test_options)
    test_options.update({'hello': 'bye'})
    test_options.update({'hyd_flag': True})

    assert net.user_pf_options == test_options

    # remove what is in user_pf_options and add hello=world
    pandapipes.pipeflow_setup.set_user_pf_options(net, reset=True, hello='world')
    assert net.user_pf_options == {'hello': 'world'}

    # check if 'hello' is added to net._options, but other options are untouched
    pandapipes.pipeflow(net, **necessary_options)
    assert 'hello' in net._options.keys() and net._options['hello'] == 'world'
    net._options.pop('hello')
    assert net._options == old_options

    # check if user_pf_options can be deleted and net._options is as it was before
    pandapipes.pipeflow_setup.set_user_pf_options(net, reset=True, hello='world')
    pandapipes.pipeflow_setup.set_user_pf_options(net, reset=True)
    assert net.user_pf_options == {}
    pandapipes.pipeflow(net, **necessary_options)
    assert 'hello' not in net._options.keys()

    # see if user arguments overrule user_pf_options, but other user_pf_options still have the
    # priority
    pandapipes.pipeflow_setup.set_user_pf_options(net, reset=True, tol_p=1e-6, tol_v=1e-6)
    pandapipes.pipeflow(net, tol_p=1e-8, **necessary_options)
    assert net.user_pf_options['tol_p'] == 1e-6
    assert net._options['tol_p'] == 1e-8
    assert net._options['tol_v'] == 1e-6


if __name__ == '__main__':
    pytest.main(["test_options.py"])
