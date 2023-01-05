.. _stanet_converter:

****************
Stanet Converter
****************

This converter can be used for converting `STANETⓇ <`https://www.stafu.de/en/home.html>`_ networks to pandapipes networks.
Currently, the converter only works for gas grids and has been tested with a number of real and synthetic gas grids.
For heating grids, some components might be missing and no tests have been performed.

The conversion has been validated with STANET Version 10.2.33 (in German).

Hints for the conversion procedure:

#. Please make sure that you perform a simulation in STANET before exporting the data,
   as this will determine a few simulation parameters needed for the converter and the
   pipeflow as well (see the notes below for more info).
#. Then, the network needs to be exported as CSV-file from STANET (export everything into
   one file).
#. The pandapipes net can be created by passing the path to the CSV-file to the
   :code:`stanet_to_pandapipes` function.

.. autofunction:: pandapipes.converter.stanet.stanet2pandapipes.stanet_to_pandapipes

.. note:: In pandapipes, pipes and valves are usually modelled as separate components.
          In STANET, however, a valve can be 'integrated' into a pipe, as a valve always contains a
          length parameter. By default, the converter splits this into a pipe and an attached valve
          in the pandapipes network. This is recommended.

          If required, the (not recommended, not documented) :code:`valve_pipe` pandapipes component
          can be used alternatively, which represents the pipe with integrated valve from STANET.
          To do so, the :code:`stanet_like_valves` parameter has to be set to :code:`True`.
          The :code:`valve_pipe` is impleted directly in the converter directory
          (:code:`pandapipes.converter.stanet.valve_pipe_component`).

.. note:: In STANET, meters can be connected through a household piping system or directly
          to a node in the main piping system, and the user can switch between these
          simulation options upon performing a simulation run. The converter will identify
          whether for the last run the household system was successfully considered. If not,
          all meters are directly connected to the main system and the household system is
          set out of service. There is unfortunately no simple toolbox function yet to switch
          between these two options. N.B. in some cases, the household piping system is set
          out of service, although this option was not chosen in the STANET run. This can
          happen, if the whole simulation failed in STANET. In this case, the converter might
          show unexpected behavior.
