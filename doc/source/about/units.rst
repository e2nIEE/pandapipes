.. _conventions:

***************************
Unit system and Conventions
***************************

**Naming conventions**

.. tabularcolumns:: |l|l|
.. csv-table:: 
   :file: nameing1.csv
   :delim: ;
   :widths: 30, 30


**Reference state values**

For calculations of compressible media, the normal conditions following DIN 1343 are used:

.. tabularcolumns:: |l|l|
.. csv-table::
   :file: normal.csv
   :delim: ;
   :widths: 30, 30


**Implemented Constants**

Implemented physical constants can be found in the python module "constants.py":

.. tabularcolumns:: |l|l|l|
.. csv-table::
   :file: constants.csv
   :delim: ;
   :widths: 30, 30, 30


**Flow directions**

Edges between junctions are created using functions, which require a "from_junction" and a "to_junction"
parameter, so the edges in the network are directed. A positive velocity value calculated for an edge is assumed to
be positive if the flow direction is in alignment with the direction of the edge. Otherwise, the velocity value is
negative.


