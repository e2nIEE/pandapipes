.. _me_controller_classes:

***************************************
Controller to couple different networks
***************************************

Basic Controller
================

These are some simple controllers that read values from one net, apply efficiency factors and unit
conversions, and write the results to another net.

.. autoclass:: pandapipes.multinet.control.controller.multinet_control.P2GControlMultiEnergy
    :members:

.. autoclass:: pandapipes.multinet.control.controller.multinet_control.G2PControlMultiEnergy
    :members:

.. autoclass:: pandapipes.multinet.control.controller.multinet_control.GasToGasConversion
    :members:


:Example:
    >>> p2g_id_el = pandapower.create_load(net_power, bus=3, p_mw=2)
    >>> p2g_id_gas = ppipes.create_source(net_gas, junction=1, mdot_kg_per_s=0)
    >>> p2g_ctrl = P2GControlMultiEnergy(multinet, p2g_id_el, p2g_id_gas, efficiency=0.7)

Coupling controller for time series simulation
==============================================

The sole purpose of the coupling controllers mentioned above is to connect generation and
consumption units in different grids. They are not able to read external input data, e.g. from
.csv files with profile data. Thus, for time series calculation, a ConstController is attached
to the input side of the coupled elements. The ConstController reads profile data from a
datasource and writes it to a dedicated element in the net (the input side of the coupled
elements). Afterwards, the P2G or G2P is called to calculate the corresponding output values to the
other net. With the following functions, matching pairs of ConstControllers and P2G/G2P
controllers can be created conveniently.

.. note:: The `order` tuple (0, 1) is important to ensure that the profile values are updated before
          the output is calculated, based on the updated values.

.. autofunction:: pandapipes.multinet.control.controller.coupled_p2g_const_control

.. autofunction:: pandapipes.multinet.control.controller.coupled_g2p_const_control
