# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import pandapipes
import pytest
from pandas.testing import assert_frame_equal


# @pytest.fixture()
def load_net():
    # create test network

    net = pandapipes.create_empty_network("test_net", fluid="lgas")
    j1 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15,
                                    name="Connection to External Grid", geodata=(0, 0))
    j2 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 2",
                                    geodata=(2, 0))
    j3 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 3",
                                    geodata=(7, 4))
    j4 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 4",
                                    geodata=(7, -4))
    j5 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 5",
                                    geodata=(5, 3))
    j6 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 6",
                                    geodata=(5, -3))

    pandapipes.create_ext_grid(net, junction=j1, p_bar=1.1, t_k=293.15, name="Grid Connection")

    pandapipes.create_pipe_from_parameters(net, from_junction=j1, to_junction=j2, length_km=10,
                                           diameter_m=0.05, name="Pipe 1", geodata=[(0, 0), (2, 0)])
    pandapipes.create_pipe_from_parameters(net, from_junction=j2, to_junction=j3, length_km=2,
                                           diameter_m=0.05, name="Pipe 2",
                                           geodata=[(2, 0), (2, 4), (7, 4)])
    pandapipes.create_pipe_from_parameters(net, from_junction=j2, to_junction=j4, length_km=2.5,
                                           diameter_m=0.05, name="Pipe 3",
                                           geodata=[(2, 0), (2, -4), (7, -4)])
    pandapipes.create_pipe_from_parameters(net, from_junction=j3, to_junction=j5, length_km=1,
                                           diameter_m=0.05, name="Pipe 4",
                                           geodata=[(7, 4), (7, 3), (5, 3)])
    pandapipes.create_pipe_from_parameters(net, from_junction=j4, to_junction=j6, length_km=1,
                                           diameter_m=0.05, name="Pipe 5",
                                           geodata=[(7, -4), (7, -3), (5, -3)])

    pandapipes.create_valve(net, from_junction=j5, to_junction=j6, diameter_m=0.05,
                            opened=True)

    pandapipes.create_sink(net, junction=j4, mdot_kg_per_s=5.45e-5, name="Sink 1")

    pandapipes.create_source(net, junction=j3, mdot_kg_per_s=3.45e-5)

    return net


def test_pickle(tmp_path):
    """
    Checks if a network saved and reloaded as a pickle file is identical.
    :return:
    :rtype:
    """

    net = load_net()
    filename = os.path.abspath(str(tmp_path)) + "test_net_1.p"

    # save test network
    pandapipes.to_pickle(net, filename)

    # load test network
    net2 = pandapipes.from_pickle(filename)

    # check if saved and loaded versions are identical
    assert pandapipes.nets_equal(net, net2), "Error in comparison after saving to Pickle."


def test_json(tmp_path):
    """
    Checks if a network saved and reloaded as a json file is identical.
    :return:
    :rtype:
    """
    net = load_net()
    filename = os.path.abspath(str(tmp_path)) + "test_net_1.json"

    # save test network
    pandapipes.to_json(net, filename)

    # load test network
    net2 = pandapipes.from_json(filename)

    # check if saved and loaded versions are identical
    assert_frame_equal(net.pipe_geodata, net2.pipe_geodata)
    del net.pipe_geodata
    del net2.pipe_geodata

    assert pandapipes.nets_equal(net, net2), "Error in comparison after saving to JSON."


def test_json_string():
    """
    Checks if a network saved and reloaded as a json file is identical.
    :return:
    :rtype:
    """
    net = load_net()

    # save test network
    json_string = pandapipes.to_json(net)

    # load test network
    net2 = pandapipes.from_json_string(json_string)

    # check if saved and loaded versions are identical
    assert_frame_equal(net.pipe_geodata, net2.pipe_geodata)
    del net.pipe_geodata
    del net2.pipe_geodata

    assert pandapipes.nets_equal(net, net2), \
        "Error in comparison after saving to JSON string."


if __name__ == '__main__':
    pytest.main(["test_file_io.py"])
