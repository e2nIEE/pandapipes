.. _important_functions:

*********
Functions
*********

This section lists important functions for running a time series simulation.
It introduces the time series main function and the auxiliary methods for
preparing a time series simulation. Furthermore, various internal functions are listed.

Run Function
============

.. _run_timeseries_ppipe:
.. autofunction:: pandapipes.timeseries.run_time_series.run_timeseries

Functions for Preparation
=========================

To prepare a time series simulation, classes/ methods of pandapower are accessed.
``DFData`` is derived from the `DataSource <https://pandapower.readthedocs.io/en/latest/timeseries/data_source.html>`_
class. The controller ``ConstControl`` is also required and can be
found in chapter :ref:`controller_classes`. In the following all functions
for the preparation are listed:

- `DFData <https://pandapower.readthedocs.io/en/latest/timeseries/data_source.html#dataframe-data-source>`_
- `ConstControl <https://pandapower.readthedocs.io/en/latest/control/controller.html#constcontrol>`_
- `OutputWriter <https://pandapower.readthedocs.io/en/latest/timeseries/output_writer.html>`_

Further Functions
=================

The following functions are called within :code:`run_timeseries`.

.. _init_time_series_ppipe:
.. autofunction:: pandapipes.timeseries.run_time_series.init_time_series

.. _control_diagnostic:
.. autofunction:: pandapower.control.util.diagnostic.control_diagnostic

.. _print_progress:
.. autofunction:: pandapower.timeseries.run_time_series.print_progress

.. _run_loop:
.. autofunction:: pandapower.timeseries.run_time_series.run_loop

.. _run_time_step:
.. autofunction:: pandapower.timeseries.run_time_series.run_time_step

.. _cleanup:
.. autofunction:: pandapipes.timeseries.run_time_series.cleanup
