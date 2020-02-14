.. _save_load:

######################
Save and Load Networks
######################

There are several ways to save and load pandapipes networks which all have their advantages and
disadvantages. In general however it is recommended to save networks in JSON files, as it is fast,
human readable, a standard format and thus interpretable for many other software packages, and has
a very dedicated Encoder/Decoder that is inherited from pandapower.


JSON
====

It is possible to save pandapipes networks to JSON files or as strings in JSON format. The saving
function is the same in both cases, the only difference is that the filename parameter is set to
None if a JSON string shall be retrieved. In order to load a network from a JSON string a different
function has to be used than for loading from file.

.. autofunction:: pandapipes.io.file_io.to_json

.. autofunction:: pandapipes.io.file_io.from_json

.. autofunction:: pandapipes.io.file_io.from_json_string


pickle
======

.. autofunction:: pandapipes.io.file_io.to_pickle

.. autofunction:: pandapipes.io.file_io.from_pickle
