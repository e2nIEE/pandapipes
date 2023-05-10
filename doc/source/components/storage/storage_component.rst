************
Mass Storage
************

Physical Model
==============

A mass storage unit draws/injects a specified mass flow from/into the connected junction.
It is implemented similar to a :ref:`sink <sinks>`, i.e. positive mass flow values correspond to charging of the
storage unit, negative mass flows means discharge of the storage unit.

The mass storage component is meant for all storage applications that do not involve relevant
temperature changes or changes in the fluid's composition, e.g., a water tower.
It is always assumed that the discharged fluid is of the same temperature and composition as the
fluid at the junction where the unit is connected to.

In stationary calculations (c.f. :ref:`pipeflow_function`), it acts as a simple sink/source,
without changing :code:`m_stored_kg` (since the duration of the mass flow is unknown).
For dynamic behaviour across time steps, an accompaning :ref:`BasicController <BasicController>` has to be implemented,
in which the charging / discharging behaviour and duration of timesteps is defined.
The dynamic calculation can then be started by :ref:`run_control <run_control_ppipe>` (for one time step) or
:ref:`run_timeseries <run_timeseries_ppipe>` for time series.

.. tip:: An example storage controller can be found in the `respective tutorial <https://github.com/e2nIEE/pandapipes/blob/develop/tutorials/building_a_storage_controller.ipynb>`_.

.. warning:: This component is only intended for mass storage applications and will likely lead to
   wrong results if it is used for district heating network applications.

Create Function
===============

.. _create_mass_storage:

For creating a single mass storage:

.. autofunction:: pandapipes.create_mass_storage

..
    # commented out, since not yet implemented:
    For creating multiple sinks at once:

    .. autofunction:: pandapipes.create_mass_storages


Component Table Data
====================

*net.mass_storage*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.40\linewidth}|

.. csv-table::
   :file: mass_storage_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40


Result Table Data
=================
*net.res_mass_storage*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.45\linewidth}|
.. csv-table:: 
   :file: mass_storage_res.csv
   :delim: ;
   :widths: 10, 10, 45

More columns may be added by controller implemente by the user,
e.g., :code:`m_stored_kg`, :code:`filling_level_percent`, etc.
