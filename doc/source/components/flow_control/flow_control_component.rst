==============
Flow Control
==============

Physical Model
==============
The flow control component enforces a specific mass flow between two junctions.
This is very helpful to control the mass flow in district heating networks.

.. warning::

    It is recommended to use the flow control components only in looped networks.
    Using the flow controller in non-looped networks, in particular on stubs, can
    likely lead to non-convergence of the pipeflow.

Create Function
===============

.. _create_flow_control:

For creating a single flow control unit:

.. autofunction:: pandapipes.create_flow_control

For creating multiple flow control units at once:

.. autofunction:: pandapipes.create_flow_controls


Component Table Data
====================

*net.flow_control*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.40\linewidth}|

.. csv-table::
   :file: flow_control_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40


Result Table Data
=================

*net.res_flow_control*

.. tabularcolumns:: |p{0.15\linewidth}|p{0.10\linewidth}|p{0.55\linewidth}|
.. csv-table::
   :file: flow_control_res.csv
   :delim: ;
   :widths: 15, 10, 55
