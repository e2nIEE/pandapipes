*********************
Circulation Pump Mass
*********************


Create Function
===============

.. _create_circ_pump_const_mass:


.. autofunction:: pandapipes.create_circ_pump_const_mass_flow


Component Table Data
====================

*net.circ_pump_mass*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.40\linewidth}|

.. csv-table::
   :file: circ_pump_mass_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40


Physical Model
==============

The circulation pump is a component that controls the flow in a looped network. At its outlet (the flow side),
it induces a pressure and temperature level just like an external grid. At the inlet (the return side), it induces
a mass flow which should be the same as the mass flow into the system.


Result Table Data
=================

*net.circ_pump_mass*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.45\linewidth}|
.. csv-table::
   :file: circ_pump_mass_res.csv
   :delim: ;
   :widths: 10, 10, 45
