# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pytest
import pandapipes


def test_add_fluid():
    net = pandapipes.create_empty_network()
    fluid_old = pandapipes.call_lib("air")

    try:
        pandapipes.get_fluid(net)
        assert False, "should'nt get here"
    except UserWarning:
        pass

    pandapipes.add_fluid_to_net(net, fluid_old)
    fluid_new = pandapipes.create_constant_fluid("arbitrary_gas2", "gas", density=2,
                                                 compressibility=2)
    pandapipes.add_fluid_to_net(net, fluid_new, overwrite=False)
    assert pandapipes.get_fluid(net) == fluid_old

    pandapipes.add_fluid_to_net(net, fluid_new)
    assert pandapipes.get_fluid(net) == fluid_new

    net["fluid"] = "Hello"

    pandapipes.add_fluid_to_net(net, fluid_new, overwrite=False)
    assert pandapipes.get_fluid(net) == "Hello"

    pandapipes.add_fluid_to_net(net, fluid_new)
    assert pandapipes.get_fluid(net) == fluid_new


def test_property_adaptation():
    net = pandapipes.create_empty_network(fluid="hgas")
    fluid = pandapipes.get_fluid(net)

    density_old = fluid.all_properties["density"]
    pandapipes.create_constant_property(net, "density", 1, overwrite=False)
    assert pandapipes.get_fluid(net).all_properties["density"] == density_old

    pandapipes.create_constant_property(net, "density", 1, overwrite=True, warn_on_duplicates=False)
    density_new = pandapipes.create_constant_property(net, "density", 1, overwrite=False)
    assert pandapipes.get_fluid(net).all_properties["density"] == density_new


def test_fluid_exceptions():
    net = pandapipes.create_empty_network(fluid="hgas")
    fluid = pandapipes.get_fluid(net)

    try:
        fluid.get_property("xyz", 100)
        assert False, "Shouldn't find property xyz!"
    except UserWarning:
        pass

    prop = pandapipes.FluidProperty()
    try:
        prop.get_property(100)
        assert False, "Shouldn't have property defined!"
    except NotImplementedError:
        pass

    try:
        pandapipes.call_lib("natural_gas")
        assert False, "Shouldn't have fluid natural_gas defined in lib!"
    except AttributeError:
        pass


if __name__ == '__main__':
    pytest.main(["test_fluid_specials.py"])
