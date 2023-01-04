*************************
Circulation Pump Pressure
*************************


Create Function
===============

.. _create_circ_pump_const_pressure:


.. autofunction:: pandapipes.create_circ_pump_const_pressure


Component Table Data
====================

*net.circ_pump_pressure*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.40\linewidth}|

.. csv-table::
   :file: circ_pump_pressure_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40


Physical Model
==============

The circulation pump is a component that controls the flow in a looped network. At its outlet (the flow side),
it induces a pressure and temperature level just like an external grid. At the inlet (the return side), it also 
induces a certain pressure, which is defined by the pressure lift parameter. The resulting mass flow through the
loop is identified by the pipeflow algorithm.


Result Table Data
=================

*net.res_circ_pump_pressure*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.45\linewidth}|
.. csv-table::
   :file: circ_pump_pressure_res.csv
   :delim: ;
   :widths: 10, 10, 45
