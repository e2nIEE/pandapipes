.. _controller:

#####################
Controller Simulation
#####################

The control components of pandapipes are based on the ones introduced by
`pandapower <https://pandapower.readthedocs.io/en/latest/control.html>`_.
Follow the link to learn more about the control loop.

Controllers can be used by pandapipes to implement control strategies for pipe networks.
As with pandapower, the :ref:`BasicController <BasicController>` can be used to
implement your own controllers. In addition, pandapower offers some predefined controllers.
Currently, it only makes sense to apply the ConstControl also for a pandapipes network.
ConstControl has no control property and is only intended to update the values during a
time series simulation.

.. toctree:: 
    :maxdepth: 1

    controller/run
    controller/controller_classes

