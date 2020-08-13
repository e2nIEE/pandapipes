.. _controller_classes:

**********
Controller
**********

Basic Controller
================

In the following the basic class :code:`Controller` of pandapower
is listed, which should serve as a platform for user-defined control implementations.

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

NonReturnValveController
========================

The NonReturnValveController makes it possible to implement a valve
which only allows flow in the connection direction. In the backward
direction the valve can be regarded as ideally closed.

.. _NonReturnValveController:
.. autoclass:: pandapipes.control.controller.non_return_valve_controller.NonReturnValveController
    :members:
