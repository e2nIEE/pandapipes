.. _timeseries:

######################
Time Series Simulation
######################

So far, only steady-state calculations of a network have been considered. With the
time series module it is possible to include the time component. For a given number
of time steps, one stationary calculation is performed for each time step, so that
a discrete value distribution, e.g. for pressures or flow velocities, can be considered along time.
Furthermore, the already existing `controller implementations from panderpower <https://pandapower.readthedocs.io/en/v2.2.2/control.html>`_
are accessed. These controllers are used to update the values of different variables for each
time step within a time series.

.. toctree:: 
    :maxdepth: 1

    timeseries/overview
    timeseries/example
    timeseries/functions
    timeseries/tutorials_ts
