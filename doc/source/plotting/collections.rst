************************
Collections for plotting
************************


Collections are a number of different symbols that can be plotted as a unit in
matplotlib, which is very handy when plotting a large number of the same component,
such as junctions in pandapipes. In order to create such collections for different
components of a pandapipes network, there are some functions to be found in the
plotting module of pandapipes.


Creating collections
====================

The following functions can be used to create collections for the most important
components of a pandapipes network. Please note that some of these functions in
fact return more than just one collection:

.. autofunction:: pandapipes.plotting.create_junction_collection

.. autofunction:: pandapipes.plotting.create_pipe_collection

.. autofunction:: pandapipes.plotting.create_sink_collection

.. autofunction:: pandapipes.plotting.create_source_collection

.. autofunction:: pandapipes.plotting.create_ext_grid_collection

.. autofunction:: pandapipes.plotting.create_valve_collection

.. autofunction:: pandapipes.plotting.create_heat_exchanger_collection


Drawing collections
===================

In order to draw a number of created collections, they can be added to a list and
handed over to the function :code:`draw_collections()` which is actually part of pandapower.
It returns a matplotlib axes, which can be extended by using :code:`add_collections_to_axes()`.

.. autofunction:: pandapipes.plotting.draw_collections

.. autofunction:: pandapipes.plotting.add_collections_to_axes
