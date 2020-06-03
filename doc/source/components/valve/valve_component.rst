.. _valve_model:

*****
Valve
*****

Create Function
===============

.. autofunction:: pandapipes.create_valve

Component Table Data
====================

*net.valve*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.40\linewidth}|
.. csv-table:: 
   :file: valve_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40


   
Physical Model
==============

A valve is an edge component which can be used to connect or disconnect network sections. The
valve can be modelled as ideal component, which is either fully open or closed causing no losses.
However, a pressure loss coefficient can be imposed as well. This property considers a specific
loss caused by the component when opened and models internal pressure losses. There are no losses
when the component is closed.

The valve is considered as branch element with no length and is modelled accordingly.



Result Table Data
=================

**For incompressible media:**

*net.res_valve*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.50\linewidth}|
.. csv-table::
   :file: valve_res_liquid.csv
   :delim: ;
   :widths: 10, 10, 50


**For compressible media:**

*net.res_valve*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.50\linewidth}|
.. csv-table::
   :file: valve_res_gas.csv
   :delim: ;
   :widths: 10, 10, 50