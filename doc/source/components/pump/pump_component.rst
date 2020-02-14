****
Pump
****


Create Function
===============

.. _create_pump:


.. autofunction:: pandapipes.create_pump


Component Table Data
====================

*net.pump*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.40\linewidth}|

.. csv-table::
   :file: pump_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40


Physical Model
==============

The pump is used to lift the pressure of a fluid flowing therethrough. The pressure rise depends on
pump specific technical product properties setting flowrate and pressure in correlation. Thus, by
given flowrate the pressure lift can be calculated. There are three different standard types
defined by default (P1, P2, P3). However, the user can easily extend the standard type list.

In pandapipes the pump is considered as branch element with zero length (pumps are considered ideal).
Thus it behaves and is considered accordingly.


Result Table Data
=================

*net.res_pump*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.45\linewidth}|
.. csv-table::
   :file: pump_res.csv
   :delim: ;
   :widths: 10, 10, 45
