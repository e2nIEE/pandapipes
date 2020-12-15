********
Multinet
********

A MultiNet is a frame (ADict) for different pandapipes & pandapower nets and coupling controllers.

Usually, a multinet is a multi energy net with one net per energy carrier.

Create Function
===============


.. _multinet:

An empty multinet is created with this function:

.. autofunction:: pandapipes.multinet.create_multinet.create_empty_multinet

An existing pandapipes gas net or pandapower net can be added with this function:

.. autofunction:: pandapipes.multinet.create_multinet.add_net_to_multinet

The nets are stored with a unique key in a dictionary in multinet['nets'].
It is also possible to add multiple nets in a single step to the multinet:

.. autofunction:: pandapipes.multinet.create_multinet.add_nets_to_multinet

:Example:
    >>> mn = create_empty_multinet()
    >>> add_net_to_multinet(mn, net, "first_net_in_multinet")
    >>> add_nets_to_multinet(mn, ("power_net1", net1), ("power_net2", net2), ("gas_net", net3)