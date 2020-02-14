.. _pipeflow:

########
Pipeflow
########

The pipeflow is the central functionality of pandapipes and is used to calculate the hydraulic
properties of a pipe bound network, i.e. the pressure at junctions and the velocity within pipes.
Furthermore it can calculate the temperatures at all junctions and the heat transfer between
components and the environment. In order to calculate the pressure, velocity and heat transfer
certain boundary conditions have to be set or assumed. In general it is possible for the user to
influence all the important boundary conditions within the network components or in the options.

The following chapters shall explain how the pipeflow works in general, which internal functions
are used and how it is possible to influence the calculation.


.. toctree::
    :maxdepth: 1
    
    pipeflow/run
    pipeflow/options
    pipeflow/pipeflow_procedure
    pipeflow/calculation_modes
    pipeflow/internal_functions


