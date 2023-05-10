.. _pipeflow_procedure:

**********************
The Pipeflow Procedure
**********************

In the pipeflow calculation, the fluid dynamics of all components with their hydraulic quantities
are calculated, i.e. the pressures for all nodes (junctions, internal pipe nodes, ... ) and the
velocities for all branches (pipes, valves, ... ). In addition it is possible to calculate the
heat transfer through the network and from the network to external components or the environment.


The Newton-Raphson Solver
=========================

In order to do so, all different :ref:`component types <components>` have to deliver their
calculation bases, but some level of abstraction is required as well. The calculations are not
performed with the pandas tables defined in the pandapipes network, but on the basis of an internal
structure called "pandapipes internal tables" or "pit", which can be found in net["_pit"]. This
internal structure consists of two numpy arrays, one for all network nodes and one for all branches.
With the help of this internal structure, a system matrix is built up, i.e. a system of equations
in order to solve for pressure and velocity based on initial guesses and formulations for the
derivatives. That means that a Newton-Raphson solver (c.f. `Newton-Raphson method on Wikipedia
<https://en.wikipedia.org/wiki/Newton%27s_method>`_, or :cite:`Dahmen2008`) is implemented in
pandapipes which works as follows:

* Initial guesses for *p* and *v* are inserted in the node and branch tables for each of the components.
  The initial guesses might either be derived from a flat start (i.e. all values are the same or
  derived from values in the tables, such as the nominal pressure level of a network junction) or
  can be taken from the internal structure if previous calculations have been performed.
* As long as the stopping criteria (usually maximum iterations as well as residual, pressure and
  velocity tolerances that can be set in the :ref:`pipeflow options <pipeflow_options>`) are not
  reached, each component calculates the residual between given values (such as the mass flow
  balance for junctions and the pressure difference for pipes) and the calculated values based on
  the current guesses of *p* and *v*. In addition, the derivatives with respect to *p* and *v* are
  calculated.
* The residual vectors and the derivatives of the components are stacked to a jacobian matrix and
  a load vector from which a linear system of equations is derived internally and solved, thus
  deriving the next Newton step for *p* and *v*.

In order to calculate the values correctly, some boundary conditions have to be given, which are:

* The pressure value for at least one network node, which is then the slack node.
* The mass flow that is extracted at every node by sinks or fed-in by sources.
* Environmental conditions, such as the height of each junction, the ambient pressure, the fluid
  temperature and other properties (c.f. :ref:`properties`).

This approach is very similar for the heat transfer calculation; the main difference is that the
transferred heat is calculated on the basis of the already calculated mass flow and the magnitudes
that are solved for are just temperature values. Here the node temperatures represent the mixed
fluid temperatures and the branch temperatures represent the temperatures at internal nodes at the
rear of the branches. The mixed fluid temperature is calculated based on all incoming flows to the
node. The internal node temperature depends on the previous node's temperature and the heat losses
to the environment (c.f. also the :ref:`description of the pipe component<pipe_component>`).


.. _connectivity_check:

Connectivity Check
==================

An important prerequisite to finding a good solution of the pipeflow is that all nodes and branches
included in the internal structure are connected to different components and finally have a
connection to some external grid that presets a fixed pressure or temperature level. Only if each
considered node has a connection to some slack node, the pressure and thus the temperature can be
calculated. Therefore a connectivity check is included in every pipeflow, unless it is switched off
by the user (see :ref:`initialize_options` for all possible options to be set). It is performed on
the basis of a `scipy sparse matrix <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_ with
the help of the `scipy csgraph functionalities
<https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html>`_ . With the help of the
connectivity check disconnected network areas can be set out of service automatically, reducing the
error-proneness of the calculation process.

.. autofunction:: pandapipes.pf.pipeflow_setup.check_connectivity


.. _internal_matrix:

Internal Structure
==================

As mentioned previously, the calculation is based on the pandapipes internal tables (pit) structure.
The included node and branch arrays contain all the information necessary for :ref:`constructing the
Jacobian matrix <jacobian>`, such as the load vectors and their derivatives. However, it is
important that only the really active parts of the network are considered. In order to simplify some
of the calculations, an internal pit is created which does not contain the nodes and branches that
were set out of service by the user or the :ref:`connectivity check <connectivity_check>`. It can be
found in net["_active_pit"].

.. note:: Branches always connect two nodes, the FROM_NODE and the TO_NODE. In every branch table \
          these two values will probably look different. In the :ref:`pipe table<pipe_component>` \
          these nodes are the indices of the respective junctions. In the pit these nodes refer to \
          the places within the node pit, and in the internal pit, these nodes are adapted once \
          again, as all out of service nodes will be dropped thus changing the other nodes' \
          placement within the table.

The functions used to create the internal pit and extract results back from it are:

.. autofunction:: pandapipes.pf.pipeflow_setup.reduce_pit

.. autofunction:: pandapipes.pf.result_extraction.extract_results_active_pit


.. _jacobian:

Constructing the Jacobian Matrix
================================

Once the internal structure is created, the Newton steps can be performed. That means that the
residual and its derivatives with respect to the estimated variables (*p*, *v*, *T*) are calculated
and written to the branch pit. Then the system matrix is constructed which means that all the
derivatives are written into one large sparse matrix in which the row indices represent the node
indices followed by the branch indices and the column indices represent the indices of the solution
variables (typically they also belong to nodes and branches). It contains all the derivatives. The
load vector (residual vector) is constructed by summarizing all the node related residuals from the
branch pit at all nodes (e.g. incoming and outgoing mass flows) and appending the residuals
calculated for the branches to it. The linearized system of equations for the hydraulic magnitudes
in the end looks like this:

.. math::
   :nowrap:

    \begin{align*}
    \begin{bmatrix}
    \frac{\partial F_0}{\partial p_0} & \frac{\partial F_0}{\partial p_1} & \cdots
    & \frac{\partial F_0}{\partial p_n} & \frac{\partial F_0}{\partial v_0}
    & \frac{\partial F_0}{\partial v_1} & \cdots & \frac{\partial F_0}{\partial v_b} \\

    \frac{\partial F_1}{\partial p_0} & \frac{\partial F_1}{\partial p_1} & \cdots
    & \frac{\partial F_1}{\partial p_n} & \frac{\partial F_1}{\partial v_0}
    & \frac{\partial F_1}{\partial v_1} & \cdots & \frac{\partial F_1}{\partial v_b} \\

    \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \ddots & \vdots \\

    \frac{\partial F_n}{\partial p_0} & \frac{\partial F_n}{\partial p_1} & \cdots
    & \frac{\partial F_n}{\partial p_n} & \frac{\partial F_n}{\partial v_0}
    & \frac{\partial F_n}{\partial v_1} & \cdots & \frac{\partial F_n}{\partial v_b} \\

    \frac{\partial F_{n+1}}{\partial p_0} & \frac{\partial F_{n+1}}{\partial p_1} & \cdots
    & \frac{\partial F_{n+1}}{\partial p_n} & \frac{\partial F_{n+1}}{\partial v_0}
    & \frac{\partial F_{n+1}}{\partial v_1} & \cdots & \frac{\partial F_{n+1}}{\partial v_b} \\

    \frac{\partial F_{n+2}}{\partial p_0} & \frac{\partial F_{n+2}}{\partial p_1} & \cdots
    & \frac{\partial F_{n+2}}{\partial p_n} & \frac{\partial F_{n+2}}{\partial v_0}
    & \frac{\partial F_{n+2}}{\partial v_1} & \cdots & \frac{\partial F_{n+2}}{\partial v_b} \\

    \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \ddots & \vdots \\

    \frac{\partial F_{n+b}}{\partial p_0} & \frac{\partial F_{n+b}}{\partial p_1} & \cdots
    & \frac{\partial F_{n+b}}{\partial p_n} & \frac{\partial F_{n+b}}{\partial v_0}
    & \frac{\partial F_{n+b}}{\partial v_1} & \cdots & \frac{\partial F_{n+b}}{\partial v_b} \\
    \end{bmatrix}

    \bullet

    \begin{bmatrix}
    \Delta p_0 \\[0.6em]
    \Delta p_1 \\[0.6em]
    \vdots \\[0.6em]
    \Delta p_n \\[0.6em]
    \Delta v_0 \\[0.6em]
    \Delta v_1 \\[0.6em]
    \vdots \\[0.6em]
    \Delta v_b \\[0.6em]
    \end{bmatrix}
    =
    \begin{bmatrix}
    F_0 \\[0.6em]
    F_1 \\[0.6em]
    \vdots \\[0.6em]
    F_n \\[0.6em]
    F_{n+1} \\[0.6em]
    F_{n+2} \\[0.6em]
    \vdots \\[0.6em]
    F_{n+b} \\[0.6em]
    \end{bmatrix}
    \end{align*}

In this formulation, *F* stands for the residual or load vector value, *n* is the number of nodes
and *b* the number of branches in the system. So the matrix on the left is the Jacobian matrix, the
vector that it is multiplied with is the vector with the estimates' step and the vector on the right
is the load vector.

.. note:: Normally the meshing in a network is rather low, so the coupling between nodes is rather \
          loose which means that most entries in the Jacobian matrix are in fact 0 and it can be \
          expressed as a sparse matrix. More information can also be found in :cite:`Ferziger2002`.


.. autofunction:: pandapipes.pf.build_system_matrix.build_system_matrix


