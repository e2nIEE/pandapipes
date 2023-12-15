================
Pressure Control
================

Physical Model
==============
The pressure control component enforces a specific pressure at the :code:`controlled_junction`.
This is achieved by adding a required pressure lift or pressure drop between the
:code:`from_junction` and :code:`to_junction`, so that :code:`controlled_p_bar` will be reached
at the controlled junction. The controlled junction can be identical to
the :code:`to_junction` to control the pressure directly at the outlet.

Internally, the behaviour is achieved by fixing the pressure variable at
:code:`controlled_junction` in the system matrix and keeping the pressure drop of the
pressure control unit variable, so that is calculated during the Newton-Raphson-calculation.

.. note::

    The temperature at the inlet and outlet junction will not be adapted by the pressure control
    unit. Therefore, these components usually operate isothermal (inlet temperature =
    outlet temperature). It is assumed that temperature changes due to compression
    or expansion (Joule-Thomson-effect) are balanced internally by adding or removing heat.


.. warning::

    A sufficient hydraulic connection between :code:`from_junction`, :code:`to_junction` and
    :code:`controlled_junction` is crucial for proper operation of this component.
    Hydraulically impossible configurations (e.g., if the controlled junction is on a different
    stub) or contradicting other pressure control units will lead to non-convergence of the pipeflow.


Create Function
===============

.. _create_press_control:

For creating a single pressure control unit:

.. autofunction:: pandapipes.create_pressure_control

For creating multiple pressure control units at once:

.. autofunction:: pandapipes.create_pressure_controls


Component Table Data
====================

*net.press_control*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.40\linewidth}|

.. csv-table::
   :file: press_control_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40


Result Table Data
=================

*net.res_press_control*

.. tabularcolumns:: |p{0.15\linewidth}|p{0.10\linewidth}|p{0.55\linewidth}|
.. csv-table::
   :file: press_control_res.csv
   :delim: ;
   :widths: 15, 10, 55
