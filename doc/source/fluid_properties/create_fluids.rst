
.. _create_fluids_from_lib:

***************
Creating Fluids
***************


Fluid from Library
==================

In the fluid library some default fluids are already implemented. Currently it is
possible to work with the default fluids high or low caloric natural
gas (hgas or lgas), water and air. One of these default fluids can be created and added to an
existing network by calling the following function.

.. autofunction:: pandapipes.create_fluid_from_lib


If only the fluid shall be retrieved without adding it to a specific network, the
following function can be used.

.. autofunction:: pandapipes.call_lib



Fluid from Parameters
=====================

Apart from the default fluids in the fluid library, there is the possibility to
create a specific fluid from certain parameters, if they are all just constants.

.. note:: This functionality is currently not well implemented and probably buggy!

.. autofunction:: pandapipes.create_constant_fluid


Otherwise, it is also possible to add a number of properties (constant and linear)
to an existing fluid in the net with the help of auxiliary functions:

.. autofunction:: pandapipes.create_constant_property

.. autofunction:: pandapipes.create_linear_property

:Example:
    >>> prop1 = pandapipes.create_constant_property(net, "density", 1000)
    >>> prop2 = pandapipes.create_linear_property(net, "compressibility", -0.01, 1)
