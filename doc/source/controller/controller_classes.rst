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

Leakage Controller
==================

With the help of the Leakage Controller it is possible to define
leaks with a given outlet area on valves, pipes, heat exchangers and junctions.
In the case of leaks at junctions, the current restriction is that they are not
only connected to a pump, they must have at least one connection to a pipe,
a valve or a heat exchanger. In addition, a leak at a junction can be realised
very simply with a sink, if a mass flow for the loss through the leak can be specified.

.. _LeakageController:
.. autoclass:: pandapipes.control.controller.leakage_controller.LeakageController
    :members:
