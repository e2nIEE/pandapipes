*************
External Grid
*************

Create Function
===============

.. _create_ext_grid:


For creating a single external grid:

.. autofunction:: pandapipes.create_ext_grid

For creating multiple external grids at once:

.. autofunction:: pandapipes.create_ext_grids


Component Table Data
====================

*net.ext_grid*

.. tabularcolumns:: |p{0.15\linewidth}|p{0.10\linewidth}|p{0.15\linewidth}|p{0.40\linewidth}|
.. csv-table:: 
   :file: ext_grid_par.csv
   :delim: ;
   :widths: 15, 10, 15, 40


   
Physical Model
===============

An external grid is used to denote nodes with fixed values of pressure or temperature, that shall
not be solved for anymore. In many cases, an external grid represents a connection to a
higher-level grid (e.g. representing the medium pressure level in a low pressure grid). Please note
the type naming convention, stating that "p" means that the pressure is fixed, "t" means that the
temperature is fixed and "pt" means that both values are fixed at the connected junction. For nodes
with fixed pressure, the mass flow into or out of the system is not known prior to calculation, but
is a result of the pipeflow calculation.

Also note that there has to be at least one fixed value of pressure for hydraulic calculations
and one fixed value for temperature in heat transfer calculations for each separate part of the
grid. This is also checked for in the :ref:`connectivity check <connectivity_check>`.


Result Table Data
=================

*net.res_ext_grid*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.1\linewidth}|p{0.50\linewidth}|
.. csv-table:: 
   :file: ext_grid_res.csv
   :delim: ;
   :widths: 10, 10, 50
