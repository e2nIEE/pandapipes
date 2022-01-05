###########
Toolbox
###########

The pandapipes toolbox is a collection of helper functions that are implemented for the pandapipes
framework. It is
designed for functions of common application that fit nowhere else. Have a look at the available functions to save
yourself the effort of maybe implementing something twice. If you develop some functionality which could be
interesting to other users as well and do not fit into one of the specialized packages, feel welcome to add your
contribution.

There are many similarities to the pandapower toolbox functions (c.f.
`this chapter <https://pandapower.readthedocs.io/en/latest/toolbox.html>`_ in the pandapower
documentation), but not all functions are transferred to pandapipes. If you want to extend the
toolbox, feel free to open a new pull request.

.. note::
    If you implement a function that might be useful for others, it is mandatory to add a short docstring to make browsing
    the toolbox practical. Ideally further comments if appropriate and a reference of authorship should be added as well.

====================================
General Issues
====================================

.. autofunction:: pandapipes.element_junction_tuples

.. autofunction:: pandapipes.pp_elements

====================================
Result and Net Information
====================================

.. autofunction:: pandapipes.nets_equal

====================================
Simulation Setup and Preparation
====================================

.. autofunction:: pandapipes.reindex_junctions

.. autofunction:: pandapipes.create_continuous_junction_index

.. autofunction:: pandapipes.reindex_elements

.. autofunction:: pandapipes.create_continuous_elements_index

====================================
Topology Modification
====================================

.. autofunction:: pandapipes.fuse_junctions

.. autofunction:: pandapipes.drop_junctions

.. autofunction:: pandapipes.drop_elements_at_junctions

.. autofunction:: pandapipes.drop_pipes

====================================
pandapower toolbox functions
====================================

Some toolbox functions can be used directly from pandapower, for example:

.. autofunction:: pandapower.clear_result_tables

.. autofunction:: pandapower.compare_arrays

.. autofunction:: pandapower.dataframes_equal

.. autofunction:: pandapower.drop_elements_simple

.. autofunction:: pandapower.get_element_index

.. autofunction:: pandapower.get_element_indices

.. autofunction:: pandapower.ensure_iterability

