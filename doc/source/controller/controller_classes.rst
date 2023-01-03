.. _controller_classes:

**********
Controller
**********

Basic Controller
================

In the following the basic class :code:`Controller` of pandapower
is listed, which should serve as a platform for user-defined control implementations.

.. _BasicController:
.. autoclass:: pandapower.control.basic_controller.Controller
    :members:

ConstControl
============

As already mentioned at the beginning of the chapter "Controller Simulation",
the :code:`ConstControl` controller is intended for use in time series simulation.
This is used to read the data from a DataSource and write it to a network.

.. _ConstControl:
.. autoclass:: pandapower.control.controller.const_control.ConstControl
    :members: