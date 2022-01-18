# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pytest
import pandapipes
from pandapipes.properties.fluids import _add_fluid_to_net


def test_add_fluid():
    net = pandapipes.create_empty_network()
    fluid_old = pandapipes.call_lib("air")

    with pytest.raises(AttributeError, match="no fluid"):
        pandapipes.get_fluid(net)

    _add_fluid_to_net(net, fluid_old)
    fluid_new = pandapipes.create_constant_fluid("arbitrary_gas2", "gas", density=2,
                                                 compressibility=2)
    _add_fluid_to_net(net, fluid_new, overwrite=False)
    assert fluid_old.name in net.fluid

    _add_fluid_to_net(net, fluid_new)
    assert fluid_new.name in net.fluid

    net["fluid"]['hello'] = "Hello"

    _add_fluid_to_net(net, fluid_new, overwrite=False)
    assert "hello" in net["fluid"]
    assert "Hello" == net["fluid"]["hello"]

    #_add_fluid_to_net(net, fluid_new)
    #assert pandapipes.get_fluid(net) == fluid_new


def test_property_adaptation():
    net = pandapipes.create_empty_network()
    _add_fluid_to_net(net, pandapipes.call_lib("hgas"))
    fluid = net.fluid['hgas']

    density_old = fluid.all_properties["density"]
    pandapipes.create_constant_property(net, "hgas", "density", 1, overwrite=False)
    assert fluid.all_properties["density"] == density_old

    pandapipes.create_constant_property(net, "hgas", "density", 1, overwrite=True, warn_on_duplicates=False)
    density_new = pandapipes.create_constant_property(net, "hgas", "density", 1, overwrite=False)
    assert fluid.all_properties["density"].equals(density_new)
    assert fluid.all_properties["density"] != density_new


def test_fluid_exceptions():
    net = pandapipes.create_empty_network()
    _add_fluid_to_net(net, pandapipes.call_lib("hgas"))
    fluid = net.fluid['hgas']

    with pytest.raises(UserWarning, match="property xyz was not defined for the fluid"):
        fluid.get_property("xyz", 100)

    prop = pandapipes.FluidProperty()
    with pytest.raises(NotImplementedError, match="Please implement a proper fluid property!"):
        prop.get_at_value(100)

    with pytest.raises(AttributeError, match="Fluid 'natural_gas' not found in the fluid library."):
        pandapipes.call_lib("natural_gas")


if __name__ == '__main__':
    pytest.main(["test_fluid_specials.py"])
