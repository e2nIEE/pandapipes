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

Custom Controller Example: BadPointPressureLiftController
==========================================================

A practical example of a custom controller implementation is provided in the Jupyter Notebook
`BadPointPressureLiftController.ipynb <https://github.com/e2nIEE/pandapipes/tree/develop/tutorials/BadPointPressureLiftController.ipynb>`_.
This notebook demonstrates how to create a controller that maintains a minimum pressure difference at the worst point in a district heating network, which is a common requirement in district heating systems.

The BadPointPressureLiftController is now available as a standalone controller class in :code:`pandapipes.control` and serves as both a template for user-defined controllers and as an important tool for operating thermal grids with pandapipes.