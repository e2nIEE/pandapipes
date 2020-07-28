import pandapipes as pp


def test_reservoir():
    """

    :return:
    """
    net = pp.create_empty_network(fluid="water")

    junction1 = pp.create_junction(net, pn_bar=1.0, tfluid_k=293.15, name="Connection to External Grid")
    junction2 = pp.create_junction(net, pn_bar=1.0, tfluid_k=293.15, name="Junction 2")

    pp.create_reservoir(net, junction1, h_m=3, name="Grid reservoir")

    pp.create_pipe_from_parameters(net, from_junction=junction1, to_junction=junction2, length_km=10, diameter_m=0.3, name="Pipe 1")

    pp.create_sink(net, junction=junction2, mdot_kg_per_s=0.545, name="Sink 1")

    pp.pipeflow(net)

    assert 1 == 1
