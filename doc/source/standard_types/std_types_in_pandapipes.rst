****************************
Standard types in pandapipes
****************************

In pandapipes there are several standard types given for different components.
For each component, standard types are defined for, one can find the corresponding CSV-files under
pandapipes.std_types.library.

If no complex model describing a component but only concrete
property characteristics are required, all defined components are combined in one CSV-file.
If there is additional information required to retrieve a model of a certain component, for each
concrete realisation the data is given in a subfolder named after this component.
Within this subfolder, each single CSV-file contains the required additional information.

In the following the currently implemented standard type categories and their implementation are shortly
described.

Property based standard types
=============================

In terms of the property based standard types all required information is stored as dictonary. The keys
of the dictionary correspond to the standard type name. All component realizations are saved
within one component category. At the moment there is

Pipe standard type
==================

A concrete pipe standard type has following properties:

.. tabularcolumns:: |p{0.15\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.40\linewidth}|
.. csv-table::
   :file: pipe_std_type.csv
   :delim: ;
   :widths: 15, 10, 25, 40

All pipe standard types currently realized can be found in pipe.csv in pandapipes.std_types.library


Model based standard types
=============================

In case of the model based standard types a model is derived which is able to describe the behaviour
of a concrete component. As part of the component category within the standard type catalog of a
certain pandapipes net, the different realisations of concrete component models are saved as dictionary
where the keys give the names of the defined


Standard type class
===================
A standard type class is the most abstract and general realisation of a standard type. It just contains
the name and type of an object. Each realisation of a specific standard type class is a subclass of this
class.

.. autoclass:: pandapipes.StdType
    :members:

A standard type class is a child of JSONSerializableClass in order to enable an easy and congruent
saving process of standard type objects.


Pump standard type class
========================
A standard type of pump is an object of the pump standard type class. It is a subclass of the standard type class.

In order to call a pump standard type object one needs to give an arbitrarily chosen name and the parameters of
a regression function which returns the pressure lift to a given volume flowrate. The paramters of the regression function
are pump specific. There are two ways to determine these. One way is, that they are already known and can be directly handed over
when initializing a pump standard type object. Alternatively, a list of pressure values, their corresponding volume flowrate values
and the degree of the regression function are given. In terms of the second option one can call the classmethod
:meth:`pandapipes.PumpStdType.from_path` overhanding the path of a CSV file containing following information.

.. tabularcolumns:: |p{0.15\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.40\linewidth}|
.. csv-table::
   :file: pump_std_type.csv
   :delim: ;
   :widths: 15, 10, 25, 40


.. autoclass:: pandapipes.PumpStdType
    :members:
