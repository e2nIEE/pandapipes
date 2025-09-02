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

Custom Controller Example: BadPointPressureController
====================================================

A practical example of a custom controller implementation is provided in the Jupyter Notebook
:ref:`BadPointPressureController <https://github.com/e2nIEE/pandapipes/tree/develop/tutorials/BadPointPressureController.ipynb>`.  
This notebook demonstrates how to create a controller that maintains a minimum pressure at the worst point in a thermal network, which is a common requirement in district heating systems.

The BadPointPressureController serves as both a template for user-defined controllers and as an important tool for operating thermal grids with pandapipes.