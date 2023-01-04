.. _components:

############################
Datastructure and Components
############################


In a pandapipes network, the elements are listed in tables (pandas-DataFrames). For each
type of elements in the network (e.g., sink, source, junction, pipe...) there exists one table.
Each element of these tables consists of one row for each element of the respective kind and one
column for each parameter.

All element tables are organized in a pandapipesNet-object. Like a pandapowerNet,
pandapipesNet-objects are of the data type 'ADict'. Thus, similar to dictionaries, values are stored
with a unique key (e.g., 'pipe': pd.DataFrame containing the pipe-table). In addition to the
dictionary syntax, the values are settable and gettable via attributes as well, so one can use e.g.
`net.pipe` instead of `net['pipe']` to get the pipe-DataFrame. Also, pandapipesNet-objects have a
`__repr__` method that prints a human-readable text summary of the network, including the number
and types of included elements and the fluid. Further explanation, also on additional data that
is stored in the pandapipesNet, can be found in the `pandapipes-paper <https://doi.org/10.3390/su12239899>`_
:cite:`Lohmeier2020`.

pandapipes provides thermo-hydraulic models for several component types, for each of which you can
find detailed information about the definition and interpretation of the parameters in the
following documentation:


.. toctree:: 
    :maxdepth: 1

    components/empty_net/empty_network
    components/junction/junction_component
    components/pipe/pipe_component
    components/valve/valve_component
    components/sink/sink_component
    components/source/source_component
    components/storage/storage_component
    components/ext_grid/ext_grid_component
    components/heat_exchanger/heat_exchanger_component
    components/pump/pump_component
    components/circulation_pump_mass/circ_pump_mass_component
    components/circulation_pump_pressure/circ_pump_pressure_component
    components/compressor/compressor_component
    components/press_control/press_control_component
    components/flow_control/flow_control_component
