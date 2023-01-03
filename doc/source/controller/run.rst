.. _run:

********************
Running a Simulation
********************

Before a controller simulation can be carried out, at least one controller must
be defined for an element in the network. This is done using the base class :code:`Controller`.
With the command

.. code::

	print(net.controller)

a list of the controllers contained in :code:`net` can be displayed.

To perform a simulation of the pandapipes network using controllers, the function
:code:`run_control` must be called.
Internally, this function calls the control loop implemented in pandapower.


.. code::

	from pandapipes.control.run_control import run_control_ppipe
	run_control_ppipe(net)

.. _run_control_ppipe:
.. autofunction:: pandapipes.control.run_control.run_control
