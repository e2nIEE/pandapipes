.. _components:

############################
Datastructure and Components
############################


A pandapipes network consists of an element table for each type of element in the network.
Each element table consists of a column for each parameter and a row for each element.

pandapipes provides thermo-hydraulic models for 9 elements, for each of which you can find
detailed information about the definition and interpretation of the parameters in the
following documentation:


.. toctree:: 
    :maxdepth: 1

    components/empty_net/empty_network
    components/junction/junction_component
    components/pipe/pipe_component
    components/valve/valve_component
    components/sink/sink_component
    components/source/source_component
    components/ext_grid/ext_grid_component
    components/pump/pump_component
    components/circulation_pump_mass/circ_pump_mass_component
    components/circulation_pump_pressure/circ_pump_pressure_component
    components/heat_exchanger/heat_exchanger_component
