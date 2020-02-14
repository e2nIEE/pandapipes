******
Source
******

Create Function
===============

.. _create_source:


.. autofunction:: pandapipes.create_source


Component Table Data
====================

*net.source*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.1\linewidth}|p{0.25\linewidth}|p{0.4\linewidth}|
.. csv-table:: 
   :file: source_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40



Physical Model
==============

A source component injects a specified mass flow into the system. The source is connected to a specified node.

Sources are typically used to model producing units in hydraulic systems. Please note that sources cannot be used to
model heat sources for the pandapipes heat mode. In this case, to model a heat flow entering the system, the heat
exchanger component can be used.


Result Table Data
=================

*net.res_source*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.50\linewidth}|
.. csv-table:: 
   :file: source_res.csv
   :delim: ;
   :widths: 10, 10, 50
