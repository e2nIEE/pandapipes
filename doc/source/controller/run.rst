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
To perform a control of the pandapipes network, the following function must be called:

.. code::

	from pandapipes.control.run_control import run_control_ppipe
	run_control_ppipe(net)

.. _run_control_ppipe:
.. autofunction:: pandapipes.control.run_control.run_control_ppipe
