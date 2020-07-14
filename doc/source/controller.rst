.. _controller:

#####################
Controller Simulation
#####################

The control components of pandapipes are based on or directly used by `pandapower <https://pandapower.readthedocs.io/en/v2.2.2/control.html>`_.
pandapipes uses the controls to simulate the network elements based on mass flows
instead of power flows. As with pandapower, the BasicController can be used to
implement your own controllers. Currently only one predefined controller exists,
namely ConstControl, which can be found in the time series calculation.
ConstControl has no control property and is only intended to update the values during a
time series simulation. The structure of the `control loop <https://pandapower.readthedocs.io/en/v2.2.2/control/control_loop.html>`_
can be taken from pandapower.

.. toctree:: 
    :maxdepth: 1

    controller/run
    controller/controller_classes
    controller/tutorials
