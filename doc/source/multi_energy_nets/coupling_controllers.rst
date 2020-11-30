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
