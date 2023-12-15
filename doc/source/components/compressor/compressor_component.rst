**********
Compressor
**********

Physical Model
==============

For a compressor between :code:`from_junction` *i* and :code:`to_junction` *j*, the pressure lift
is calculated based on the absolute pressure (i.e., relative pressure as shown in pandapipes +
ambient pressure) and the :code:`pressure_ratio` :math:`\Pi`.

.. math::

    p_j = \begin{cases}
    (p_i + p_{amb}) \cdot \Pi - p_{amb},    & \text{if } \dot{m}_{ij} > 0 \text{ kg/s} \\
    p_i,              & \text{otherwise}
    \end{cases}

The required compression power for compressors with :math:`\dot{m}_{ij} > 0` in Megawatt is
estimated by the ideal adiabatic change in enthalpy by applying this equation
:cite:`Schmidt.2015`:

.. math::

    \begin{equation}
    \label{eq:compr_power}
    P_{\text{compr},ij} = \dot{m}_{ij}\frac{\kappa}{\kappa - 1} R_\text{s} z(p_i) T_i \cdot \left(\Pi ^{\frac{\kappa -1}{\kappa}}-1\right)\cdot 10^{-6}
    \end{equation}

Where :math:`\kappa` is the isentropic exponent (assumed as 1.4), :math:`R_\text{s}` is the
specific gas constant, :math:`z(p)` the compressibility and :math:`T_i` the temperature.

Create Function
===============

.. _create_compressor:


.. autofunction:: pandapipes.create_compressor


Component Table Data
====================

*net.compressor*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.40\linewidth}|

.. csv-table::
   :file: compressor_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40


Result Table Data
=================

*net.res_compressor*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.45\linewidth}|
.. csv-table::
   :file: compressor_res.csv
   :delim: ;
   :widths: 10, 10, 45
