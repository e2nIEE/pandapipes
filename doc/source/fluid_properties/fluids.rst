******************************
Fluid Properties in pandapipes
******************************

The pipelines in the pandapipes network are run with a certain fluid. This can be
chosen individually for the net from the fluid library or by own creation. The fluids
are defined by certain properties.

The following fluids currently exist in the library:

- `hgas` and `lgas` (high and low calorific natural gas),
- `hydrogen`,
- `methane`,
- `water`,
- `biomethane_pure` and `biomathane_treated`,
- `air`.


.. Note::
   **Biomethanes**:
   A particularity of injecting biomethane in the gas grid in Germany is the addition of LPG to enhance
   the Wobbe-Index and the heating value of the gas and thus make it equivalent to the (high-calorific) natural gas transported
   in the grid. This addition is done in a gas treatment unit (Biogaseinspeiseanlage) upstream of the biomethane
   feed-in junction.

..
   tabularcolumns:: |l|l|l|

.. csv-table::
   :file: composition.csv
   :delim: ;
   :widths: 10, 10, 10

..
   IDEALLY: embed the table in the box or decrease its size.

The Fluid Class
===============

Inside this class, different properties with their values are implemented.
These properties can be called by different functions. There exists a general
function, which returns the values of the requested property. Furthermore, there
are different specified functions to return directly the value of the density,
viscosity and heat capacity.

In addition to the already existing properties there is a function, which
allows to add new properties. It also warns if there is already a property
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
JSONSerializableClass, as they shall also be saved and reloaded savely (see also the chapter
:ref:`save_load`).


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
