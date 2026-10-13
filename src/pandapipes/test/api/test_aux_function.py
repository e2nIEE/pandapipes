# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pytest

from pandapipes.pf.internals_toolbox import select_from_pit, sum_by_group


@pytest.mark.xfail(reason="The function has been changed. Unclear how to interpret the new version.")
def test_select_from_pit():
    """

    :return:
    :rtype:
    """

    input_array = np.array([2, 4, 5])
    table_index_array = np.array([1, 2, 3, 4, 5])
    data = np.array([10, 11, 12, 13, 14])

    ret = select_from_pit(table_index_array, input_array, data)
    expected_result = np.array([11, 13, 14])

    assert np.all(ret == expected_result)


class TestSumByGroup:
    """Tests for `sum_by_group` function

    Some tests follow a naming convention:
        one_group, two_groups -- tells number of groups used in the test
        one_value, two_values -- tells number of values arrays
        one_item, two_items -- tells number of items in values arrays

        So, `test_one_group_two_values_three_items` would mean the test
        uses:
            * group = np.array([1])  or  group = np.array([1, 1, 1])
            * vals = [np.array([1, 2, 3]), np.array([4, 5, 6])],
                meaning len(vals) == 2 and len(vals[0]) == len(vals[1]) == 3

    """

    def test_one_group_one_value_one_item(self):
        """Base test.

        Tests that something does work
        """
        gr = np.array([1])
        val = np.array([1])
        unique_groups, (grouped_sum,) = sum_by_group(gr, val)

        assert np.allclose(unique_groups, [1])
        assert np.allclose(grouped_sum, [1])

    def test_one_group_one_value_two_items(self):
        """Test that summing for one group works."""
        gr = np.array([1, 1])
        val = np.array([1, 2])
        unique_groups, (grouped_sum,) = sum_by_group(gr, val)

        assert np.allclose(unique_groups, [1])
        assert np.allclose(grouped_sum, [3])

    def test_one_group_two_distinct_values_one_item(self):
        """Test that summing for one group still works even with 2 values."""
        gr = np.array([1])
        val = [np.array([1]), np.array([2])]
        unique_groups, (grouped_sum_1, grouped_sum_2) = sum_by_group(gr, *val)

        assert np.allclose(unique_groups, [1])
        assert np.allclose(grouped_sum_1, [1])
        assert np.allclose(grouped_sum_2, [2])

    def test_one_group_two_distinct_values_two_items(self):
        """Test summing for one group with 2 values and with 2 items works."""
        gr = np.array([1, 1])
        val = [np.array([1, 2]), np.array([2, 3])]
        unique_groups, (grouped_sum_1, grouped_sum_2) = sum_by_group(gr, *val)

        assert np.allclose(unique_groups, [1])
        assert np.allclose(grouped_sum_1, [3])
        assert np.allclose(grouped_sum_2, [5])

    def test_two_groups_two_distinct_values_two_items(self):
        """Test that we get sums per group and per value.

        A general test
        """
        gr = np.array([1, 2])
        val = [np.array([1, 3]), np.array([2, 4])]
        unique_groups, (grouped_sum_1, grouped_sum_2) = sum_by_group(gr, *val)

        assert np.allclose(unique_groups, [1, 2])
        assert np.allclose(grouped_sum_1, [1, 3])
        assert np.allclose(grouped_sum_2, [2, 4])

    def test_two_groups_two_distinct_values_nan_item(self):
        """Test that NaNs are preserved and don't get in the way."""
        gr = np.array([1, 2, 2])
        val = [np.array([1, 2, 3]), np.array([2, 4, np.nan])]
        unique_groups, (grouped_sum_1, grouped_sum_2) = sum_by_group(gr, *val)

        assert np.allclose(unique_groups, [1, 2])
        assert np.allclose(grouped_sum_1, [1, 5])
        assert np.allclose(grouped_sum_2, [2, np.nan], equal_nan=True)

    def test_unsorted_groups(self):
        """Test that the groups can be unsorted."""
        gr = np.array([2, 1, 2])
        val = np.array([1, 2, 3])
        unique_groups, (grouped_sum,) = sum_by_group(gr, val)

        assert np.allclose(unique_groups, [1, 2])
        assert np.allclose(grouped_sum, [2, 4])

    def test_nonconsecutive_groups(self):
        """Test that groups don't need to be consecutive."""
        gr = np.array([5, 1, 5])
        val = np.array([1, 2, 3])
        unique_groups, (grouped_sum,) = sum_by_group(gr, val)

        assert np.allclose(unique_groups, [1, 5])
        assert np.allclose(grouped_sum, [2, 4])

    def test_all_nan_items(self):
        """Test we can handle all NaN items."""
        gr = np.array([1, 1, 1])
        val = np.array([np.nan, np.nan, np.nan])
        unique_groups, (grouped_sum,) = sum_by_group(gr, val)

        assert np.allclose(unique_groups, [1])
        assert np.allclose(grouped_sum, [np.nan], equal_nan=True)

    def test_empty_input_one_value(self):
        """Test we handle both empty groups and empty value."""
        gr = np.array([])
        val = np.array([])
        unique_groups, (grouped_sum,) = sum_by_group(gr, val)

        assert np.allclose(unique_groups, [])
        assert np.allclose(grouped_sum, [])

    def test_empty_input_two_values(self):
        """Test we handle both empty groups and return the input number of empty values."""
        gr = np.array([])
        val = [np.array([]), np.array([])]
        unique_groups, (grouped_sum_1, grouped_sum_2) = sum_by_group(gr, *val)

        assert np.allclose(unique_groups, [])
        assert np.allclose(grouped_sum_1, [])
        assert np.allclose(grouped_sum_2, [])

    def test_value_dtype_preserved(self):
        """Test we preserve dtype for a single value."""
        gr = np.array([1])
        val = np.array([1], dtype=np.uint32)

        unique_groups, (grouped_sum,) = sum_by_group(gr, val)

        assert grouped_sum.dtype == np.uint32

    def test_values_dtypes_preserved(self):
        """Test we preserve dtypes (distinct!) for several values."""
        gr = np.array([1])
        val = (np.array([1], dtype=np.float32), np.array([1], dtype=np.uint32))

        unique_groups, (fl32, ui32) = sum_by_group(gr, *val)

        assert fl32.dtype == np.float32
        assert ui32.dtype == np.uint32

    def test_empty_input_dtypes_preserved(self):
        """Test we preserve dtypes (distinct!) even for empty input."""
        gr = np.array([], dtype=int)
        val = (np.array([], dtype=np.float32), np.array([], dtype=np.uint32))

        unique_groups, (fl32, ui32) = sum_by_group(gr, *val)

        assert unique_groups.dtype == int
        assert fl32.dtype == np.float32
        assert ui32.dtype == np.uint32

    def test_negative_groups(self):
        """Test we don't support negative groups."""
        gr = np.array([-1, -2, -3])
        val = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="must have no negative elements"):
            unique_groups, (grouped_sum,) = sum_by_group(gr, val)

    def test_len_mismatch(self):
        """Test we don't allow groups and values of different lengths."""
        gr = np.array([1, 1])
        val = np.array([2])

        with pytest.raises(ValueError, match="don't have the same length"):
            unique_groups, (grouped_sum,) = sum_by_group(gr, val)
