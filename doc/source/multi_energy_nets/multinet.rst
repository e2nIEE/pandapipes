********
Multinet
********

Create Function
===============


.. _multinet:

An empty multinet is created with this function:

.. autofunction:: pandapipes.multinet.multinet.create_empty_multinet

An existing pandapipes gas net or pandapower net can be added with this function:

.. autofunction:: pandapipes.multinet.multinet.add_net_to_multinet

It is also possible to add multiple nets in a single step to the multinet:

.. autofunction:: pandapipes.multinet.multinet.add_nets_to_multinet

:Example:
    >>> mn = create_empty_multinet()
    >>> add_net_to_multinet(mn, net, "first_net_in_multinet")
    >>> add_nets_to_multinet(mn, ("power_net1", net1), ("power_net2", net2), ("gas_net", net3)