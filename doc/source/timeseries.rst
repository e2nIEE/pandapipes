.. _timeseries:

######################
Time Series Simulation
######################

Using the time series module, it is possible to calculate quasi-stationary time-dependent problems. This means that for a given number
of time steps, one stationary calculation is performed for each one, so that
a discrete value distribution, e.g. for pressures or flow velocities, can be considered along time.
Furthermore, the already existing `controller implementations from panderpower <https://pandapower.readthedocs.io/en/latest/control.html>`_
can be accessed. This means that control strategies can also be applied for time-dependent problems.

.. toctree:: 
    :maxdepth: 1

    timeseries/overview
    timeseries/functions

