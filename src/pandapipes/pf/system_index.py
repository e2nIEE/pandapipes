# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from __future__ import annotations

from enum import Enum

import numpy as np
from dataclasses import dataclass, field
from pandapipes.idx_node import IdxNode

class HydVarEq(str, Enum):
    """Variable and equation types for the hydraulic linear system."""

    NODE          = "NODE"
    BRANCH        = "BRANCH"
    SLACK         = "SLACK"
    PINIT         = "PINIT"
    MDOTINIT      = "MDOTINIT"
    MDOTSLACKINIT = "MDOTSLACKINIT"


class ThermVarEq(str, Enum):
    """Variable and equation types for the thermal linear system."""

    NODE     = "NODE"
    BRANCH   = "BRANCH"
    TINIT    = "TINIT"
    TOUTINIT = "TOUTINIT"


class EqWriteMode(str, Enum):
    """Write mode for ComponentEquations entries.

    UNIQUE:   exclusive row ownership — conflict check on registration, default
              contributions to those rows are stripped in assemble
    ADDITIVE: values accumulated (default)
    MEAN:     mean of all residual contributions to the same row (NaN-filtered);
              Jacobian entries averaged per unique (row, col) pair
    """

    UNIQUE   = "unique"
    ADDITIVE = "additive"
    MEAN     = "mean"


@dataclass
class ComponentEquations:
    """Sparse (COO format) contributions of one component to the global Jacobian and residual vector.

    ADDITIVE (default): contributions accumulate, no conflict check.
    UNIQUE:  the component claims those rows exclusively; conflict check on registration,
             default contributions to those rows are stripped in assemble.
    MEAN:    multiple contributions to the same row are averaged (NaN-filtered).
    """

    rows: np.ndarray           # int32, equation row indices (global)
    cols: np.ndarray           # int32, variable column indices (global)
    data: np.ndarray           # float64, Jacobian values
    residual_rows: np.ndarray  # int32, residual vector positions
    residual_data: np.ndarray  # float64, residual vector values
    mode: EqWriteMode = EqWriteMode.ADDITIVE

    @classmethod
    def empty(cls) -> ComponentEquations:
        return cls(
            np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64),
        )


@dataclass
class ComponentRegistry:
    """Two-bucket registry for component equations.

    default:   equations that accumulate (COO summing)
    overrides: equations written after default; UNIQUE overrides strip default
               contributions from their rows

    UNIQUE entries trigger a row-conflict check against previously registered UNIQUE
    entries in the same bucket (add → default, add_override → overrides).
    """

    default:   list[ComponentEquations] = field(default_factory=list)
    overrides: list[ComponentEquations] = field(default_factory=list)

    def _check_conflict(self, eq: ComponentEquations, bucket: list) -> None:
        unique_equations = [existing for existing in bucket if existing.mode == EqWriteMode.UNIQUE]
        if not unique_equations:
            return
        claimed_rows = np.concatenate([existing.rows for existing in unique_equations])
        conflicting_rows = np.intersect1d(claimed_rows, eq.rows)
        if len(conflicting_rows):
            raise ValueError(
                f"Equation conflict: rows {conflicting_rows.tolist()} are already claimed "
                f"by a UNIQUE entry."
            )

    def add(self, eq: ComponentEquations) -> None:
        if eq.mode == EqWriteMode.UNIQUE:
            self._check_conflict(eq, self.default)
        self.default.append(eq)

    def add_override(self, eq: ComponentEquations) -> None:
        if eq.mode == EqWriteMode.UNIQUE:
            self._check_conflict(eq, self.overrides)
        self.overrides.append(eq)

    def assemble(self, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build COO Jacobian entries and residual vector from all registered components.

        UNIQUE override rows are stripped from the default pool.
        MEAN entries (both buckets combined) are averaged per (row, col) in the Jacobian
        and per row in the residual vector (NaN-filtered); they replace any prior values at
        those positions.

        Returns
        -------
        jac_rows, jac_cols, jac_data : int32 / float64 arrays (COO format)
        residual                     : float64 array of length *size*

        """
        unique_overrides = [eq for eq in self.overrides if eq.mode == EqWriteMode.UNIQUE]
        claimed_rows = (
            np.concatenate([eq.rows for eq in unique_overrides])
            if unique_overrides else np.empty(0, dtype=np.int32)
        )

        default_non_mean  = [eq for eq in self.default   if eq.mode != EqWriteMode.MEAN]
        default_mean      = [eq for eq in self.default   if eq.mode == EqWriteMode.MEAN]
        override_non_mean = [eq for eq in self.overrides if eq.mode != EqWriteMode.MEAN]
        override_mean     = [eq for eq in self.overrides if eq.mode == EqWriteMode.MEAN]

        def _concat(equations, attr, dtype):
            return (np.concatenate([getattr(eq, attr) for eq in equations]).astype(dtype)
                    if equations else np.empty(0, dtype=dtype))

        # --- default bucket (without MEAN), minus the rows claimed by UNIQUE overrides ---
        default_jac_rows = _concat(default_non_mean, 'rows',          np.int32)
        default_jac_cols = _concat(default_non_mean, 'cols',          np.int32)
        default_jac_data = _concat(default_non_mean, 'data',          np.float64)
        default_res_rows = _concat(default_non_mean, 'residual_rows', np.int32)
        default_res_data = _concat(default_non_mean, 'residual_data', np.float64)

        if len(claimed_rows):
            keep_jac = ~np.isin(default_jac_rows, claimed_rows)
            default_jac_rows = default_jac_rows[keep_jac]
            default_jac_cols = default_jac_cols[keep_jac]
            default_jac_data = default_jac_data[keep_jac]
            keep_res = ~np.isin(default_res_rows, claimed_rows)
            default_res_rows = default_res_rows[keep_res]
            default_res_data = default_res_data[keep_res]

        # --- override bucket (without MEAN) ---
        override_jac_rows = _concat(override_non_mean, 'rows',          np.int32)
        override_jac_cols = _concat(override_non_mean, 'cols',          np.int32)
        override_jac_data = _concat(override_non_mean, 'data',          np.float64)
        override_res_rows = _concat(override_non_mean, 'residual_rows', np.int32)
        override_res_data = _concat(override_non_mean, 'residual_data', np.float64)

        # Jacobian: duplicate (row, col) entries are summed later by the COO -> CSR conversion
        jac_rows = np.concatenate([default_jac_rows, override_jac_rows]).astype(np.int32)
        jac_cols = np.concatenate([default_jac_cols, override_jac_cols]).astype(np.int32)
        jac_data = np.concatenate([default_jac_data, override_jac_data]).astype(np.float64)

        # Residual: default contributions are summed, rows touched by an override are replaced
        # by the (summed) override contributions
        residual = np.zeros(size, dtype=np.float64)
        np.add.at(residual, default_res_rows, default_res_data)

        override_residual = np.zeros(size, dtype=np.float64)
        np.add.at(override_residual, override_res_rows, override_res_data)
        if len(override_res_rows):
            is_overridden = np.zeros(size, dtype=bool)
            is_overridden[override_res_rows] = True
            residual[is_overridden] = override_residual[is_overridden]

        # --- MEAN entries of both buckets ---
        mean_equations = default_mean + override_mean
        if mean_equations:
            mean_jac_rows = _concat(mean_equations, 'rows',          np.int32)
            mean_jac_cols = _concat(mean_equations, 'cols',          np.int32)
            mean_jac_data = _concat(mean_equations, 'data',          np.float64)
            mean_res_rows = _concat(mean_equations, 'residual_rows', np.int32)
            mean_res_data = _concat(mean_equations, 'residual_data', np.float64)

            # Jacobian: average per unique (row, col) pair, encoded as a single flat key
            flat_keys = mean_jac_rows.astype(np.int64) * size + mean_jac_cols.astype(np.int64)
            unique_keys, key_inverse, key_counts = np.unique(
                flat_keys, return_inverse=True, return_counts=True
            )
            jac_sums = np.zeros(len(unique_keys), dtype=np.float64)
            np.add.at(jac_sums, key_inverse, mean_jac_data)
            jac_rows = np.concatenate([jac_rows, (unique_keys // size).astype(np.int32)])
            jac_cols = np.concatenate([jac_cols, (unique_keys %  size).astype(np.int32)])
            jac_data = np.concatenate([jac_data, jac_sums / key_counts])

            # Residual: average per row, ignoring NaN contributions
            is_valid = ~np.isnan(mean_res_data)
            if is_valid.any():
                valid_rows, valid_data = mean_res_rows[is_valid], mean_res_data[is_valid]
                unique_rows, row_inverse, row_counts = np.unique(
                    valid_rows, return_inverse=True, return_counts=True
                )
                res_sums = np.zeros(len(unique_rows), dtype=np.float64)
                np.add.at(res_sums, row_inverse, valid_data)
                residual[unique_rows] = res_sums / row_counts

        return jac_rows, jac_cols, jac_data, residual


class PitWriteMode(str, Enum):
    """Write mode for PIT entries.

    UNIQUE:   exclusive write — conflict check on registration, direct assignment in apply
    ADDITIVE: values accumulated with np.add.at (default)
    MEAN:     mean of all values written to the same (row, col) position
    """

    UNIQUE   = "unique"
    ADDITIVE = "additive"
    MEAN     = "mean"


@dataclass
class PitEntries:
    """COO-format data for writing into a PIT (node or branch) array."""

    rows: np.ndarray  # int32, row indices into the PIT
    cols: np.ndarray  # int32, column indices into the PIT
    data: np.ndarray  # values to write
    mode: PitWriteMode = PitWriteMode.UNIQUE


@dataclass
class PitRegistry:
    """Two-bucket registry for PIT initialization.

    default:   base entries written first
    overrides: entries written second, winning over default entries at the same positions

    UNIQUE entries trigger a (row, col) conflict check against all previously registered
    UNIQUE entries in both buckets.
    """

    default:   list[PitEntries] = field(default_factory=list)
    overrides: list[PitEntries] = field(default_factory=list)

    def _check_conflict(self, entries: PitEntries, bucket: list) -> None:
        unique_entries = [existing for existing in bucket if existing.mode == PitWriteMode.UNIQUE]
        if not unique_entries:
            return
        claimed_positions = set(zip(
            np.concatenate([existing.rows for existing in unique_entries]).tolist(),
            np.concatenate([existing.cols for existing in unique_entries]).tolist(),
        ))
        conflicting_positions = claimed_positions & set(
            zip(entries.rows.tolist(), entries.cols.tolist())
        )
        if conflicting_positions:
            raise ValueError(
                f"PIT conflict: (row, col) pairs {conflicting_positions} are already "
                f"claimed by a unique entry."
            )

    def add(self, entries: PitEntries) -> None:
        if entries.mode == PitWriteMode.UNIQUE:
            self._check_conflict(entries, self.default)
        self.default.append(entries)

    def add_override(self, entries: PitEntries) -> None:
        if entries.mode == PitWriteMode.UNIQUE:
            self._check_conflict(entries, self.overrides)
        self.overrides.append(entries)

    def apply(self, pit: np.ndarray) -> None:
        mean_entries = []
        for entries in self.default + self.overrides:
            if entries.mode == PitWriteMode.UNIQUE:
                pit[entries.rows, entries.cols] = entries.data
            elif entries.mode == PitWriteMode.ADDITIVE:
                np.add.at(pit, (entries.rows, entries.cols), entries.data)
            else:
                mean_entries.append(entries)

        if mean_entries:
            mean_rows = np.concatenate([entries.rows for entries in mean_entries])
            mean_cols = np.concatenate([entries.cols for entries in mean_entries])
            mean_data = np.concatenate([entries.data for entries in mean_entries])
            is_valid = ~np.isnan(mean_data)
            mean_rows, mean_cols, mean_data = mean_rows[is_valid], mean_cols[is_valid], mean_data[is_valid]
            if len(mean_rows):
                # average per unique (row, col) pair, encoded as a single flat key
                n_cols = pit.shape[1]
                flat_keys = mean_rows * n_cols + mean_cols
                unique_keys, key_inverse, key_counts = np.unique(
                    flat_keys, return_inverse=True, return_counts=True
                )
                value_sums = np.zeros(len(unique_keys))
                np.add.at(value_sums, key_inverse, mean_data)
                pit[unique_keys // n_cols, unique_keys % n_cols] = value_sums / key_counts


class BaseSystemIndex:
    """Central registry of all variables and equations in the linear system.

    Variables and equations are registered via ``_register()`` using ``HydVarEq``
    (or integer PIT constants for thermal) as keys.  In this square system each
    variable has exactly one equation — the numerical indices are identical.
    """

    def __init__(self) -> None:
        """Initialize an empty variable/equation block registry."""
        self._blocks: dict = {}
        self._size: int = 0

    def _block_key(self, key):
        """Actual dict key ``_blocks`` is stored/looked-up under for variable/equation *key*.

        Overridable so subclasses can namespace keys (e.g. combined_pipeflow's
        ``HydThermSystemIndex``, which needs ``HydVarEq.NODE`` and ``ThermVarEq.NODE`` to resolve
        to different blocks despite being equal as plain strings). All of ``idx``/``_register``/
        ``_register_sparse`` go through this, so overriding it here is enough - no need to
        separately override each of them.
        """
        return key

    def idx(self, var, subset: np.ndarray | None = None) -> np.ndarray:
        """Matrix index for variable/equation *var* (optionally filtered to *subset* positions)."""
        block_indices = self._blocks[self._block_key(var)]
        return block_indices if subset is None else block_indices[subset]

    def size(self) -> int:
        """Total number of rows/columns in the global matrix."""
        return self._size

    def _register(self, key, indices: np.ndarray) -> None:
        """Register a variable or equation block and update _size."""
        block_indices = indices.astype(np.int32)
        self._blocks[self._block_key(key)] = block_indices
        if len(block_indices):
            self._size = max(self._size, int(block_indices[-1]) + 1)

    def _register_sparse(self, key, full_size: int, node_indices: np.ndarray,
                         matrix_indices: np.ndarray) -> None:
        """Register a variable/equation block that only exists for a SUBSET of nodes.

        E.g. MDOTSLACKINIT/SLACK, only defined at P-type nodes - but sized like the FULL node
        array (``full_size``), with -1 at every position outside ``node_indices``. This lets
        ``idx(key, some_node_indices)`` be called with raw node indices directly, exactly like
        PINIT/NODE, instead of requiring callers to translate to a rank-within-subset first.
        ``matrix_indices`` holds the matrix index of each entry in ``node_indices``.
        """
        block_indices = np.full(full_size, -1, dtype=np.int32)
        block_indices[node_indices] = matrix_indices
        self._blocks[self._block_key(key)] = block_indices
        if len(matrix_indices):
            self._size = max(self._size, int(matrix_indices.max()) + 1)


class HydraulicSystemIndex(BaseSystemIndex):
    """Variable / equation registry for the hydraulic solve.

    Layout (columns = rows in square system):
        0 .. n_nodes-1                   PINIT / NODE          pressure / node mass-balance
        n_nodes .. n_nodes+n_branches-1  MDOTINIT / BRANCH     mass flow / branch momentum
        n_nodes+n_branches .. ...        MDOTSLACKINIT / SLACK  slack-mass variables (P-type nodes)

    ``slack_nodes`` is the sorted array of node_pit row indices with NODE_TYPE == P.

    ``MDOTSLACKINIT``/``SLACK`` only exist at P-type nodes, but are sized like the full node
    array (with -1 at every non-slack position) so ``idx(HydVarEq.MDOTSLACKINIT, some_nodes)``
    works with raw node indices directly, same as ``PINIT``/``NODE`` - a caller (e.g. ExtGrid,
    CirculationPump) never needs to translate its own node indices into a rank-within-slack_nodes
    first, it just needs to know which of ITS OWN nodes are P-type slack nodes at all.
    """

    def __init__(self, node_pit: np.ndarray, branch_pit: np.ndarray) -> None:
        """Build the variable/equation index for a hydraulic solve over *node_pit*/*branch_pit*."""
        super().__init__()
        self.slack_nodes = np.where(node_pit[:, IdxNode.NODE_TYPE] == IdxNode.P)[0].astype(np.int32)

        n_nodes = len(node_pit)
        n_branches = len(branch_pit)
        n_slacks = len(self.slack_nodes)
        slack_indices = np.arange(n_slacks, dtype=np.int32) + n_nodes + n_branches

        self._register(HydVarEq.PINIT,    np.arange(n_nodes))
        self._register(HydVarEq.MDOTINIT, np.arange(n_branches) + n_nodes)
        self._register_sparse(HydVarEq.MDOTSLACKINIT, n_nodes, self.slack_nodes, slack_indices)

        self._register(HydVarEq.NODE,   np.arange(n_nodes))
        self._register(HydVarEq.BRANCH, np.arange(n_branches) + n_nodes)
        self._register_sparse(HydVarEq.SLACK, n_nodes, self.slack_nodes, slack_indices)


class HeatSystemIndex(BaseSystemIndex):
    """Variable / equation registry for the thermal solve.

    Layout (columns = rows in square system):
        0 .. n_nodes-1                   TINIT / NODE       node temperature / node energy balance
        n_nodes .. n_nodes+n_branches-1  TOUTINIT / BRANCH  branch outlet temperature / branch energy
    """

    def __init__(self, node_pit: np.ndarray, branch_pit: np.ndarray) -> None:
        """Build the variable/equation index for a thermal solve over *node_pit*/*branch_pit*."""
        super().__init__()
        self.slack_nodes = np.where(node_pit[:, IdxNode.NODE_TYPE_T] == IdxNode.T)[0].astype(np.int32)

        n_nodes = len(node_pit)
        n_branches = len(branch_pit)

        self._register(ThermVarEq.TINIT,    np.arange(n_nodes))
        self._register(ThermVarEq.TOUTINIT, np.arange(n_branches) + n_nodes)

        self._register(ThermVarEq.NODE,   self._blocks[ThermVarEq.TINIT])
        self._register(ThermVarEq.BRANCH, self._blocks[ThermVarEq.TOUTINIT])
