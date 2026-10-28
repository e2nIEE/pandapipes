*****************
Interactive Maps
*****************

Plotting a Network on a Web Map
===============================

:func:`create_map_figure` draws a network on an interactive map using plotly (MapLibre).
The built-in map styles such as ``carto-positron`` or ``open-street-map`` need **no access token**.
Plotly >= 5.24 is required (``pip install "pandapipes[plotting]"``).

The junction geodata must be given as WGS84 longitude (``x``) and latitude (``y``). Convert projected
coordinates first with :func:`pandapipes.plotting.convert_epsg_junction_geodata`. The map tiles are loaded by
the browser from public tile servers, so an internet connection is needed to display the background.

.. code-block:: python

    import pandapipes as pp
    import pandapipes.networks as nw
    from pandapipes.plotting import create_map_figure, convert_epsg_junction_geodata

    net = nw.schutterwald()
    convert_epsg_junction_geodata(net, epsg_in=31467, epsg_out=4326)
    pp.pipeflow(net)
    fig = create_map_figure(net, pipe_result="v_mean_m_per_s")
    fig.show()

See the tutorial ``plotly_map_plotting`` for a complete district heating example showing the
temperatures in a closed supply/return loop.

.. autofunction:: pandapipes.plotting.create_map_figure
