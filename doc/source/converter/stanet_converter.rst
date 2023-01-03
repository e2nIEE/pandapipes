.. _stanet_converter:

****************
Stanet Converter
****************

This converter can be used for converting (`STANETⓇ <`https://www.stafu.de/en/home.html>`_) networks to pandapipes networks.
Currently, the converter only works for gas grids.
Heating grids are not implemented yet.
The conversion has been validated with STANET Version 10.2.33 (in German).

First, the network needs to be exported as CSV-file from STANET.
Then, pandapipes net can be created by passing the path to the CSV-file to the :code:`stanet_to_pandapipes` function.

.. autofunction:: pandapipes.converter.stanet.stanet2pandapipes.stanet_to_pandapipes

.. note:: In pandapipes, pipes and valves are usually modelled as separate components.
          In STANET, however, a valve can be 'integrated' into a pipe. By default,
          the converter splits this into a pipe and an attached valve in the pandapiepes network.
          This is recommended.

          If required, the (not recommended, not documented) :code:`valve_pipe` pandapipes component
          can be used alternatively, which represents the pipe with integrated valve from STANET.
          To do so, the :code:`stanet_like_valves` parameter has to be set to :code:`True`.
          The :code:`valve_pipe` is impleted directly in the converter directory
          (:code:`pandapipes.converter.stanet.valve_pipe_component`).

