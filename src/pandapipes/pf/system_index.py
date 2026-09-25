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

    UNIQUE:   exclusive row ownership — conflict check on registration, normal
              contributions to those rows are stripped in assemble
    ADDITIVE: values accumulated (default)
    MEAN:     mean of all load contributions to the same row (NaN-filtered);
              Jacobian entries averaged per unique (row, col) pair
    """

    UNIQUE   = "unique"
    ADDITIVE = "additive"
    MEAN     = "mean"


@dataclass
class ComponentEquations:
    """Sparse (COO format) contributions of one component to the global Jacobian and load vector.

    ADDITIVE (default): contributions accumulate, no conflict check.
    UNIQUE:  the component claims those rows exclusively; conflict check on registration,
             normal contributions to those rows are stripped in assemble.
    MEAN:    multiple contributions to the same row are averaged (NaN-filtered).
    """

    rows: np.ndarray      # int32, equation row indices (global)
    cols: np.ndarray      # int32, variable column indices (global)
    data: np.ndarray      # float64, Jacobian values
    load_rows: np.ndarray  # int32, load vector positions
    load_data: np.ndarray  # float64, load vector values
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

    normal:    equations that accumulate (COO summing)
    overrides: equations written after normal; UNIQUE overrides strip normal
               contributions from their rows

    UNIQUE entries trigger a row-conflict check against previously registered UNIQUE
    entries in the same bucket (add → normal, add_override → overrides).
    """

    normal:    list[ComponentEquations] = field(default_factory=list)
    overrides: list[ComponentEquations] = field(default_factory=list)

    def _check_conflict(self, eq: ComponentEquations, bucket: list) -> None:
        existing = [e for e in bucket if e.mode == EqWriteMode.UNIQUE]
        if not existing:
            return
        claimed = np.concatenate([e.rows for e in existing])
        conflict = np.intersect1d(claimed, eq.rows)
        if len(conflict):
            raise ValueError(
                f"Equation conflict: rows {conflict.tolist()} are already claimed "
                f"by a UNIQUE entry."
            )

    def add(self, eq: ComponentEquations) -> None:
        if eq.mode == EqWriteMode.UNIQUE:
            self._check_conflict(eq, self.normal)
        self.normal.append(eq)

    def add_override(self, eq: ComponentEquations) -> None:
        if eq.mode == EqWriteMode.UNIQUE:
            self._check_conflict(eq, self.overrides)
        self.overrides.append(eq)

    def assemble(self, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build COO Jacobian entries and load vector from all registered components.

        UNIQUE override rows are stripped from the normal pool.
        MEAN entries (both buckets combined) are averaged per (row, col) in the Jacobian
        and per row in the load vector (NaN-filtered); they replace any prior values at
        those positions.

        Returns
        -------
        rows, cols, data : int32 / float64 arrays (COO format)
        load_vector      : float64 array of length *size*

        """
        unique_overrides = [ov for ov in self.overrides if ov.mode == EqWriteMode.UNIQUE]
        claimed = (
            np.concatenate([ov.rows for ov in unique_overrides])
            if unique_overrides else np.empty(0, dtype=np.int32)
        )

        n_non_mean = [e for e in self.normal    if e.mode != EqWriteMode.MEAN]
        n_mean     = [e for e in self.normal    if e.mode == EqWriteMode.MEAN]
        ov_non_mean = [e for e in self.overrides if e.mode != EqWriteMode.MEAN]
        ov_mean     = [e for e in self.overrides if e.mode == EqWriteMode.MEAN]

        def _cat(entries, attr, dtype):
            return (np.concatenate([getattr(e, attr) for e in entries]).astype(dtype)
                    if entries else np.empty(0, dtype=dtype))

        n_r  = _cat(n_non_mean, 'rows',      np.int32)
        n_c  = _cat(n_non_mean, 'cols',      np.int32)
        n_d  = _cat(n_non_mean, 'data',      np.float64)
        n_lr = _cat(n_non_mean, 'load_rows', np.int32)
        n_ld = _cat(n_non_mean, 'load_data', np.float64)

        if len(claimed):
            keep = ~np.isin(n_r, claimed)
            n_r, n_c, n_d = n_r[keep], n_c[keep], n_d[keep]
            keep_lr = ~np.isin(n_lr, claimed)
            n_lr, n_ld = n_lr[keep_lr], n_ld[keep_lr]

        ov_r  = _cat(ov_non_mean, 'rows',      np.int32)
        ov_c  = _cat(ov_non_mean, 'cols',      np.int32)
        ov_d  = _cat(ov_non_mean, 'data',      np.float64)
        ov_lr = _cat(ov_non_mean, 'load_rows', np.int32)
        ov_ld = _cat(ov_non_mean, 'load_data', np.float64)

        rows = np.concatenate([n_r, ov_r]).astype(np.int32)
        cols = np.concatenate([n_c, ov_c]).astype(np.int32)
        data = np.concatenate([n_d, ov_d]).astype(np.float64)

        load = np.zeros(size, dtype=np.float64)
        np.add.at(load, n_lr, n_ld)

        load_ov = np.zeros(size, dtype=np.float64)
        np.add.at(load_ov, ov_lr, ov_ld)
        if len(ov_lr):
            ov_set = np.zeros(size, dtype=bool)
            ov_set[ov_lr] = True
            load[ov_set] = load_ov[ov_set]

        all_mean = n_mean + ov_mean
        if all_mean:
            m_r  = _cat(all_mean, 'rows',      np.int32)
            m_c  = _cat(all_mean, 'cols',      np.int32)
            m_d  = _cat(all_mean, 'data',      np.float64)
            m_lr = _cat(all_mean, 'load_rows', np.int32)
            m_ld = _cat(all_mean, 'load_data', np.float64)

            rc = m_r.astype(np.int64) * size + m_c.astype(np.int64)
            u_rc, rc_inv, rc_cnt = np.unique(rc, return_inverse=True, return_counts=True)
            jac_s = np.zeros(len(u_rc), dtype=np.float64)
            np.add.at(jac_s, rc_inv, m_d)
            rows = np.concatenate([rows, (u_rc // size).astype(np.int32)])
            cols = np.concatenate([cols, (u_rc %  size).astype(np.int32)])
            data = np.concatenate([data, jac_s / rc_cnt])

            valid = ~np.isnan(m_ld)
            if valid.any():
                vr, vd = m_lr[valid], m_ld[valid]
                u_r, r_inv, r_cnt = np.unique(vr, return_inverse=True, return_counts=True)
                ld_s = np.zeros(len(u_r), dtype=np.float64)
                np.add.at(ld_s, r_inv, vd)
                load[u_r] = ld_s / r_cnt

        return rows, cols, data, load


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

    normal:    base entries written first
    overrides: entries written second, winning over normal entries at the same positions

    UNIQUE entries trigger a (row, col) conflict check against all previously registered
    UNIQUE entries in both buckets.
    """

    normal:    list[PitEntries] = field(default_factory=list)
    overrides: list[PitEntries] = field(default_factory=list)

    def _check_conflict(self, entries: PitEntries, bucket: list) -> None:
        existing = [e for e in bucket if e.mode == PitWriteMode.UNIQUE]
        if not existing:
            return
        claimed = set(zip(
            np.concatenate([e.rows for e in existing]).tolist(),
            np.concatenate([e.cols for e in existing]).tolist(),
        ))
        conflict = claimed & set(zip(entries.rows.tolist(), entries.cols.tolist()))
        if conflict:
            raise ValueError(
                f"PIT conflict: (row, col) pairs {conflict} are already "
                f"claimed by a unique entry."
            )

    def add(self, entries: PitEntries) -> None:
        if entries.mode == PitWriteMode.UNIQUE:
            self._check_conflict(entries, self.normal)
        self.normal.append(entries)

    def add_override(self, entries: PitEntries) -> None:
        if entries.mode == PitWriteMode.UNIQUE:
            self._check_conflict(entries, self.overrides)
        self.overrides.append(entries)

    def apply(self, pit: np.ndarray) -> None:
        mean_entries = []
        for e in self.normal + self.overrides:
            if e.mode == PitWriteMode.UNIQUE:
                pit[e.rows, e.cols] = e.data
            elif e.mode == PitWriteMode.ADDITIVE:
                np.add.at(pit, (e.rows, e.cols), e.data)
            else:
                mean_entries.append(e)

        if mean_entries:
            all_rows = np.concatenate([e.rows for e in mean_entries])
            all_cols = np.concatenate([e.cols for e in mean_entries])
            all_data = np.concatenate([e.data for e in mean_entries])
            valid = ~np.isnan(all_data)
            all_rows, all_cols, all_data = all_rows[valid], all_cols[valid], all_data[valid]
            if len(all_rows):
                keys = all_rows * pit.shape[1] + all_cols
                unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
                sums = np.zeros(len(unique_keys))
                np.add.at(sums, inverse, all_data)
                pit[unique_keys // pit.shape[1], unique_keys % pit.shape[1]] = sums / counts


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
        arr = self._blocks[self._block_key(var)]
        return arr if subset is None else arr[subset]

    def size(self) -> int:
        """Total number of rows/columns in the global matrix."""
        return self._size

    def _register(self, key, indices: np.ndarray) -> None:
        """Register a variable or equation block and update _size."""
        idx = indices.astype(np.int32)
        self._blocks[self._block_key(key)] = idx
        if len(idx):
            self._size = max(self._size, int(idx[-1]) + 1)

    def _register_sparse(self, key, full_size: int, node_indices: np.ndarray,
                         values: np.ndarray) -> None:
        """Register a variable/equation block that only exists for a SUBSET of nodes.

        E.g. MDOTSLACKINIT/SLACK, only defined at P-type nodes - but sized like the FULL node
        array (``full_size``), with -1 at every position outside ``node_indices``. This lets
        ``idx(key, some_node_indices)`` be called with raw node indices directly, exactly like
        PINIT/NODE, instead of requiring callers to translate to a rank-within-subset first.
        """
        arr = np.full(full_size, -1, dtype=np.int32)
        arr[node_indices] = values
        self._blocks[self._block_key(key)] = arr
        if len(values):
            self._size = max(self._size, int(values.max()) + 1)


class HydraulicSystemIndex(BaseSystemIndex):
    """Variable / equation registry for the hydraulic solve.

    Layout (columns = rows in square system):
        0 .. len_n-1             PINIT / NODE          pressure / node mass-balance
        len_n .. len_n+len_b-1   MDOTINIT / BRANCH     mass flow / branch momentum
        len_n+len_b .. ...       MDOTSLACKINIT / SLACK  slack-mass variables (P-type nodes)

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

        len_n = len(node_pit)
        len_b = len(branch_pit)
        len_s = len(self.slack_nodes)
        slack_vals = np.arange(len_s, dtype=np.int32) + len_n + len_b

        self._register(HydVarEq.PINIT,    np.arange(len_n))
        self._register(HydVarEq.MDOTINIT, np.arange(len_b) + len_n)
        self._register_sparse(HydVarEq.MDOTSLACKINIT, len_n, self.slack_nodes, slack_vals)

        self._register(HydVarEq.NODE,    np.arange(len_n))
        self._register(HydVarEq.BRANCH, np.arange(len_b) + len_n)
        self._register_sparse(HydVarEq.SLACK, len_n, self.slack_nodes, slack_vals)


class HeatSystemIndex(BaseSystemIndex):
    """Variable / equation registry for the thermal solve.

    Layout (columns = rows in square system):
        0 .. len_n-1             TINIT / NODE    node temperature / node energy balance
        len_n .. len_n+len_b-1   TOUTINIT / BRANCH  branch outlet temperature / branch energy
    """

    def __init__(self, node_pit: np.ndarray, branch_pit: np.ndarray) -> None:
        """Build the variable/equation index for a thermal solve over *node_pit*/*branch_pit*."""
        super().__init__()
        self.slack_nodes = np.where(node_pit[:, IdxNode.NODE_TYPE_T] == IdxNode.T)[0].astype(np.int32)

        len_n = len(node_pit)
        len_b = len(branch_pit)

        self._register(ThermVarEq.TINIT,    np.arange(len_n))
        self._register(ThermVarEq.TOUTINIT, np.arange(len_b) + len_n)

        self._register(ThermVarEq.NODE,   self._blocks[ThermVarEq.TINIT])
        self._register(ThermVarEq.BRANCH, self._blocks[ThermVarEq.TOUTINIT])
