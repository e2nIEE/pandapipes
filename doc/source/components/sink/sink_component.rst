.. _sinks:

****
Sink
****

Create Function
===============

.. _create_sink:

For creating a single sink:

.. autofunction:: pandapipes.create_sink

For creating multiple sinks at once:

.. autofunction:: pandapipes.create_sinks


Component Table Data
====================

*net.sink*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.40\linewidth}|

.. csv-table::
   :file: sink_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40



Physical Model
==============

A sink draws a specified mass flow from the connected junction. Note that positive mass flow values
correspond to flows leaving the network system.

Sinks are typically used to model loads in hydraulic systems. Please note that sinks cannot be used to model loads for
the pandapipes heat mode. In this case, to model a heat flow drawn by a consumer, the heat exchanger component can be
used.



Result Table Data
=================
*net.res_sink*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.45\linewidth}|
.. csv-table:: 
   :file: sink_res.csv
   :delim: ;
   :widths: 10, 10, 45
