.. _networks:

########
Networks
########


Example and Test-Networks
=========================

In addition, there are several example networks that are described below. The networks were
created in STANET as well as in OpenModelica and converted to pandapipes. Each converted
network was saved in one json file. Calling the corresponding function loads the
results from the json file and saves them as a pandapipesNet. In the case of a
STANET net, you can usually choose between the friction models Nikuradse and
Prandtl-Colebrook. For OpenModelica the results exist for Prandtl-Colebrook
and Swamee-Jain, which approximates the Prandtl-Colebrook formula.
In addition, a separate section lists the thermal networks created with
OpenModelica that were calculated with Prandtl-Colebrook.


.. toctree::
    :maxdepth: 1

    networks/combined/combined_networks
    networks/meshed/meshed_networks
    networks/one_pipe/one_pipe_networks
    networks/strand/strand_networks
    networks/t_cross/t_cross_networks
    networks/two_eg/two_external_grids
    networks/heat_transfer_networks/heat_transfer_networks

Each network is accompanied by a picture, whereby the following description
of the components must be observed:

.. image:: networks_legend.png
	:alt: alternate Text
	:align: center


Schutterwald-Network
====================

pandapipes includes the natural gas distribution network from the supplementary material of
:cite:`Kisse2020`.
In this exemplary network, more than 1500 houses and 10 km of pipes with geo-coordinates are
included. Further description and figures can be found `in the paper <https://doi.org/10.3390/en13164052>`_.
It can be opened by `pandapipes.networks.schutterwald()`

.. autofunction:: pandapipes.networks.schutterwald
