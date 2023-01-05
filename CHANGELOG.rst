Change Log
=============

[0.8.0] - 2023-01-05
-------------------------------
- [ADDED] new component `flow controller`: a branch component that controls the flow through itself. The flow controller component is able to keep its mass flow fixed. It adapts the pressure drop between two junctions to reflect the desired flow situation.
- [ADDED] new component `mass_storage` and tutorial how to use it
- [ADDED] a stanet converter has been added incl. a stanet converter documentation
- [ADDED] in the course of the stanet converter release a component called valve-pipe was introduced. It is a combination of a valve and a pipe
- [ADDED] automated release process
- [ADDED] documentation added for circulation pumps of any kind, compressors, flow controller, pressure controller and mass storage
- [ADDED] adding property based on the sutherland model as additional FluidProperty
- [ADDED] besides regression models it is also possible to use interpolation models in case of StdTypes
- [ADDED] enable bulk creation of flow controls and ext grids
- [ADDED] toolbox function to extract the _pit (pandapipes internal tables) structure for nodes and branches as pandas tables with meaningful names for the stored columns as given in the node_idx and branch_idx files
- [ADDED] new global variable `__format_version__` that shall only be increased in case of API changes (i.e. if the convert_format function for JSON I/O must be called)
- [ADDED] documentation check which is able to throw errors in case of warnings
- [ADDED] example networks including new component types for the convert_format test
- [CHANGED] circ pump is now a branch component
- [CHANGED] default roughness parameter `k_mm` for pipes is now 0.2 mm instead of 1 mm (all create_pipe... functions)
- [CHANGED] instead of from and to junction, in case of circulation pumps it is called return and flow junction from now on
- [CHANGED] suffix 'flow' added to input variable p, t an mass in case of circulation pumps
- [CHANGED] ctrl/ts_variables dict in multinet gets an intermediate level 'nets', so that the structure is similar to Multinet ADict
- [CHANGED] order of the pump entries was adapted (v, p, degree)
- [CHANGED] by default p and t are set to None in case of ext grids. Based on the type selected, p and t must be adapted accordingly. If set to 'auto', based on the given p and t values, the type is set
- [CHANGED] 'auto' can also be set for type in case of circulation pumps. The behavior is the same as in case of ext grids
- [CHANGED] authors list adapted
- [CHANGED] as of now, not one but two example nets (water and gas) will be created for the convert_format test (ensure backward compatibility when loading nets from jsons)
- [FIXED] the references to pipeflow procedures have changed. Caused problems in the documentation
- [FIXED] removed unused import of 'progress_bar' from pandapower which caused import problems

[0.7.0] - 2022-08-02
-------------------------------
- [ADDED] automated test with Python 3.10 added to GitHub Actions CI (now Python 3.7 - 3.10)
- [ADDED] function to test tutorials / jupyter notebooks for raised errors
- [ADDED] add tests for tutorials to GitHub Actions
- [ADDED] some internal functions of the hydraulic calculation are also implemented with numba's Just-in-time compilation mode for speed-up (switch on/off with the use_numba flag)
- [ADDED] function for subnet selection
- [ADDED] functions for standard type changes
- [ADDED] added \__eq__ method for JSONSerializableClass using deepdiff library in pandapower. Required adjustments in property comparison test.
- [CHANGED] timeseries progress bar now shown with tqdm as in pandapower
- [CHANGED] some restructuring (the pf package now contains different modules for pipeflow internals)
- [CHANGED] for hydraulic calculation, the derivatives and some result extraction functions were made global (previously in component models). Components can influence the calculation beforehand/afterwards (e.g. for pressure lift) in pre-/ post derivative calculation functions.
- [CHANGED] standard types now under net.std_types instead of net.std_type
- [CHANGED] renaming extract_results to init_results in component_models
- [CHANGED] standard types are created, not added anymore
- [FIXED] bugfix to resolve problems with numpy indexing (especially with numpy.repeat) in some component models
- [FIXED] HHV/LHV for H2 corrected
- [FIXED] only considering external grids, which are in service
- [FIXED] preventing unexpected behavior of pressure control component or displaying logger warnings
- [FIXED] usage of tqdm for progress bar print
- [FIXED] individual run function can be passed in run_timeseries now (test added)
- [FIXED] converged flag set equals to False at the beginning of each pipeflow

[0.6.0] - 2022-02-07
-------------------------------
- [ADDED] Adding `pressure controller` as new component
- [ADDED] Adding `compressor` as new component
- [ADDED] Compressing power of a pump component are returned as result
- [ADDED] Adding polynomial fluids
- [CHANGED] Removing irrelevant results in branch models with zero length (mean velocity, lambda, reynolds)
- [FIXED] Only ext grids in service are considered
- [FIXED] Converting format of the nets in a multinet correctly
- [FIXED] Changes in pandas are considered
- [FIXED] Bug with multinet controller in run_control
- [FIXED] Bugfix in pandapower changed function cleanup in run_timeseries

[0.5.0] - 2021-07-29
-------------------------------
- [ADDED] Enabling encryption of pandapipes networks
- [CHANGED] Removing p_scale from default net options
- [FIXED] Input of get_compressibility in fluids.py is pressure, not temperature

[0.4.0] - 2021-03-09
-------------------------------
- [ADDED] Six new tutorials.
- [ADDED] Adding flag in run_control if controller convergence should be checked in each level or just at the end.
- [ADDED] Adding variables to change collection colors more specifically.
- [ADDED] Added flag in fluid for constant fluid properties if warning is displayed or not in case of several input variables.
- [ADDED] Added function in graph_searches.py to determine junction distances.
- [CHANGED] Deleted one tutorial for heating networks. Another one was updated.
- [CHANGED] Deleted tutorial sections from the documentation. These should be available only via the corresponding homepage.
- [CHANGED] compressibility and der_compr for hydrogen is now for 273.15 K (instead of 293.15 K before).
- [CHANGED] Small updates in the pandapipes documentation.
- [CHANGED] Adding a maximum number of iterations when using colebrook friction model.
- [CHANGED] In fluids changed function name get_property into get_at_value.
- [CHANGED] In std_type renaming attribute type into component.
- [FIXED] Bugfix in controller what to do by default in case on net divergence.

[0.3.0] - 2021-01-08
-------------------------------
- [ADDED] added bulk create functions for junctions, sinks, sources, pipes (from std_type and parameters) and valves (clean control)
- [ADDED] automated Testing for Python 3.8
- [ADDED] github action tests added
- [ADDED] LHV and HHV properties for fuel gases
- [ADDED] multinet functionality to couple a pandapower and pandapipes network
- [ADDED] example gas distribution grid with houses and geodata
- [ADDED] compressibility values for hydrogen
- [ADDED] create graph added to topology
- [CHANGED] bypassing for pumps, pressure lift = 0 for negative and very high volume flows
- [CHANGED] pressure lift in pumps now based on p_from (inlet-volume flow)
- [CHANGED] logger level for pipeflow messages is now "debug" instead of "info"
- [CHANGED] usage of generic functions in the create module which are mostly based on pandapower
- [CHANGED] renaming component_models.py into base_component.py
- [CHANGED] removing extract_results from pipe, heat_exchanger and valve up to the abstract file branch_models.py
- [CHANGED] adding initial_run to net.controller and removing initial_run and recycle from all controllers
- [CHANGED] updating run_control and run_timeseries in order to minimize duplicated code between pandapower and pandapipes
- [CHANGED] update of generic geodata creation in plotting
- [CHANGED] addding initial pressure and volume values as well as its chosen regression polynomial degree to each pump
- [FIXED] all tests pass with pandas > 1.x
- [FIXED] bug fix for ext_grid result extraction in case of unordered connected junctions
- [FIXED] problem of not converging pipeflow solved if there are no branches

[0.2.0] - 2020-09-03
-------------------------------
- [ADDED] added the pipeflow option "reuse_internal_data" which allows to reuse the system matrix from one pipeflow to the next in combination with "only_update_hydraulic_matrix" - useful for timeseries calculations
- [ADDED] hydrogen properties
- [ADDED] Swamee-Jain friction model
- [ADDED] test networks (water) for Swamee-Jain friction model
- [ADDED] further explanation in the documentation, e.g. on heating networks and time series / controller
- [ADDED] heating network and time series tutorials
- [ADDED] enable net loading built in different pandapipes versions
- [ADDED] carry over new artificial coordinate functions from pandapower
- [ADDED] functionality to create_networkx graph for pandapipes networks
- [ADDED] tests for connected components searches in pandapipes networks
- [CHANGED] property files for bi-atomic gases
- [CHANGED] make ppipe_hook serializable and inherit from pp_hook by using decorators
- [CHANGED] changed column "controller" in controller table to "object"
- [CHANGED] changes in run_control/run_time_series for better pandapower code reusability like using initial_run variable instead of initial_pipeflow
- [CHANGED] names of parameters for regression function in pump
- [CHANGED] pressure lift for pumps is now always >= 0
- [CHANGED] on reverse flow, the pressure lift for pumps is 0
- [CHANGED] add_fluid_to_net is now a private function (usually the wrapper create_fluid_from_lib should be used)
- [FIXED] direction of pump in the water test network 'versatility' for OpenModelica
- [FIXED] accurate calculation of v in get_internal_results for pipes
- [FIXED] enable loading of nets containing controller

[0.1.2] - 2020-06-05
-------------------------------
- [ADDED] allow pipeflow for empty net (with no results)
- [ADDED] tests for plotting
- [ADDED] new toolbox functions and tests
- [ADDED] get... methods for fluids
- [ADDED] tutorial, documentation and tests for heat networks
- [CHANGED] default column for controllers changed from controller to object
- [CHANGED] deepcopy (now in ADict) and repr of pandapipes net + fluid
- [CHANGED] improved plotting: respect in_service
- [CHANGED] for fluids, comp, molar_mass and der_comp are now read from .txt-files and are no longer hardcoded
- [CHANGED] pandapipes.toolbox renamed to pandapipes.internals_toolbox
- [FIXED] np.isclose comparison instead of 'p_from != p_to' in pipe_component to allow for computational inaccuracy

[0.1.1] - 2020-04-03
-------------------------------
- [ADDED] badges for pypi and versions
- [CHANGED] result table initialization now in most cases only contains one DF creation call #42
- [CHANGED] from pandapower tempdir to pytest tmp_path in test
- [CHANGED] default orientation of sink and source collections to avoid overlapping
- [FIXED] missing images and typos in documentation

[0.1.0] - 2020-03-18
-------------------------------
- first release of pandapipes
