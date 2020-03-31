******************************
Fluid Properties in pandapipes
******************************

The pipelines in the pandapipes network are run with a certain fluid. This can be
chosen individually for the net from the fluid library or by own creation. The fluids
are defined by certain properties. In the fluid library currently high or low caloric natural
gas (hgas and lgas), water and air are implemented with default properties.


The Fluid Class
===============

Inside this class, different properties with their values are implemented.
These properties can be called by different functions. There exists a general
function, which returns the values of the requested property. Furthermore, there
are different specified functions to return directly the value of the density,
viscosity and heat capacity.

In addition to the already existing properties there is a function, which
allows to add new properties. It also warns if the there is already a property
with the same name and can overwrite an existing property with a new value.


.. autoclass:: pandapipes.Fluid
    :members:


Properties
===========

The idea behind the properties is a functional correlation between physical
quantities, e. g. a linear correlation between pressure and temperature based on
the ideal gas law. This way the pandapipes components can ask for a specific
fluid property, such as the density, at a given operation point. Some classes
for different functional correlations have already been implemented. All properties
have to inherit from the Property base class in order to be used in a fluid. These
classes shall be introduced in the following. They all inherit from the pandapower
JSONSerializableClass, as they shall also be saved and reloaded savely (see also the chapter Save
and Load Networks).


Property Base Class
-------------------

.. autoclass:: pandapipes.FluidProperty


Property With Constant Value
----------------------------

.. autoclass:: pandapipes.FluidPropertyConstant
    :members:


Property With Linear Correlation
--------------------------------

.. autoclass:: pandapipes.FluidPropertyLinear
    :members:


Property With Interpolated Measure Points
-----------------------------------------

.. autoclass:: pandapipes.FluidPropertyInterExtra
    :members:
