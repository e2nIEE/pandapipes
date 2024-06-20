.. _calculation_modes:

*****************
Calculation Modes
*****************

When running a pipeflow, you can choose between different calculation modes. This page explains the
differences between the modes and also describes necessary input parameters.

In summary, there are two modes available to determine the pressure and velocity distribution in the
examined network:

One suitable for incompressible and one suitable for compressible media, respectively. A user does
not have to define the calculation mode manually. Instead, the fluid used for the calculation
contains a parameter specifying if the fluid behaves compressible or incompressible. pandapipes
automatically chooses the appropriate calculation mode based on this fluid parameter.

In addition, there is one mode available which calculates the temperature distribution in the
network. It is important to note that with the current version of pandapipes, the temperature
calculation is only intended to be used in combination with incompressible fluids, although the
program will not throw a warning or error message if temperatures are calculated in combination with
gases. In this case, a user has to be aware that the temperature calculation is done sequentially
with the current pandapipes version.

.. image:: modes.png
	:width: 15em
	:alt: alternate Text
	:align: center

This means that a calculated temperature does not have an influence on the hydraulic fluid properties, such as the density. This approach is only
valid if the fluid properties do not show a strong dependence on the temperature or if temperature
variations are very small.

The user has to decide if these assumptions are suitable for his purposes. In future, pandapipes
will be extended in order to make sure that calculated temperatures also effect hydraulic fluid
properties. To activate temperature calculation, the pipe flow option "mode" has to be set
to "sequential", "bidirectional" or "heat". If heat is chosen, the user has to provide a solution vector of the hydraulics calculation manually.


Hydraulic calculations for incompressible media
===================================================================

Important parameters of the network main components (junctions and pipes) needed for the calculation
are listed in the following table. Note that some parameters which can be specified are not needed
for the calculation of incompressible media. The :ref:`component section <components>` of this
manual contains an extensive overview of all component parameters.


.. tabularcolumns:: |p{0.12\linewidth}|p{0.25\linewidth}|p{0.30\linewidth}|
.. csv-table::
   :file: incompressible_par.csv
   :delim: ;
   :widths: 10, 25, 40


The main effects of pressure loss accounted for during the calculation of incompressible media are
losses due to friction and losses due to bendings or assets. In addition, height differences can
influence the pressure and velocity distribution.

Because the fluid is incompressible, the velocity along a pipe is constant and less inputs are needed in comparison with
the calculation of compressible fluids.


Hydraulic calculations for compressible media
===================================================================

Important parameters of the network main components (junctions and pipes) needed for the calculation
are listed in the following table. Note that some parameters which can be specified are not needed
for the calculation of compressible media. The :ref:`component section <components>` of this manual
contains an extensive overview of all component parameters.

.. tabularcolumns:: |p{0.12\linewidth}|p{0.10\linewidth}|p{0.30\linewidth}|
.. csv-table::
   :file: compressible_par.csv
   :delim: ;
   :widths: 10, 25, 40

The law of ideal gases and the comparison with a prescribed reference state are part of the internal
calculation, which means that pandapipes also makes use of internal constants, e.g. the normal
pressure and normal temperature, to calculate pressure drops.

As mentioned in the introduction of this chapter, temperatures used for hydraulic calculations
cannot be calculated for compressible media. Instead, temperature values at junctions are parameters
required as input values to calculate pressure losses. The corresponding junction variables "tn_k"
will be considered constant throughout the simulation.

Other temperature values than the ones listed in the table are not needed for hydraulic calculations
of compressible media. Especially the parameter "text_k", which can be defined for pipes, does not
have an effect in hydraulic calculations.

In gas flows, the velocity is typically not constant along a pipeline. For this reason, result
tables for pipes show more entries in comparison with the result tables for incompressible media.


Temperature calculations (pipeflow option: mode = "sequential", mode = "bidrectional" or mode = "heat")
=======================================================================================================

Important parameters of the network main components (junctions and pipes) needed for the calculation
are listed in the following table. The :ref:`component section <components>` of this manual contains
an extensive overview of all component parameters.

.. tabularcolumns:: |p{0.12\linewidth}|p{0.25\linewidth}|p{0.30\linewidth}|
.. csv=table::
   :file: temperature_par.csv
   :delim: ;
   :widths: 10, 25, 40

Note that temperature values at junctions, the tn_k variables, have a different function than in the
hydraulic calculation mode for gases. For temperature calculations, tn_k specifies only the initial
temperature value for the calculation. Calculated temperatures will usually differ from the assumed
input.

To calculate heat losses along pipelines it is necessary to specifiy the temperature of the
surroundings. The temperature needed for loss calculation is stored in the parameter text_k which
can be specified for :ref:`pipe components<pipe_component>`.

Summary of temperature values
===================================================================

Because a lot of temperature values are needed for the different pandapipes calculations, the
following table summarizes available temperatures and their purpose:

.. tabularcolumns:: |p{0.12\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.30\linewidth}|
.. csv-table::
   :file: temperature_overview.csv
   :delim: ;
   :widths: 10, 10, 25, 40
