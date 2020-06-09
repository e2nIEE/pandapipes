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
.. autofunction:: pandapipes.timeseries.run_time_series.run_timeseries_ppipe

Preparatory Functions
=====================

.. _prepare_grid:
.. autofunction:: pandapipes.test.pipeflow_internals.test_time_series._prepare_grid


.. _output_writer:
.. autofunction:: pandapipes.test.pipeflow_internals.test_time_series._output_writer

.. note:: The previous method :code:`_output_writer()` returns an object of the `OutputWriter <https://pandapower.readthedocs.io/en/v2.2.2/timeseries/output_writer.html>`_ class of pandapower.

Further Functions
=================

The following functions are called within :code:`run_timeseries_ppipe`.

.. _init_time_series_ppipe:
.. autofunction:: pandapipes.timeseries.run_time_series.init_time_series_ppipe

.. _control_diagnostic:
.. autofunction:: pandapower.control.util.diagnostic.control_diagnostic

.. _print_progress:
.. autofunction:: pandapipes.timeseries.run_time_series.print_progress

.. _run_time_step:
.. autofunction:: pandapipes.timeseries.run_time_series.run_time_step

.. _cleanup:
.. autofunction:: pandapipes.timeseries.run_time_series.cleanup

Data Source Function
====================

The data sources resp. the given time series of the sinks and sources
are read out with the below method. It is called within :code:`_prepare_grid(net)`
and uses the class `DFData <https://pandapower.readthedocs.io/en/v2.2.2/timeseries/data_source.html#dataframe-data-source>`_
from pandapower.

.. _data_source:
.. autofunction:: pandapipes.test.pipeflow_internals.test_time_series._data_source