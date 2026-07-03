from hive_abm.fields import RangeCommand1D, RectCommand2D, apply_range_add_1d, apply_rect_add_2d, argmax_first_2d
from hive_abm.windows import max_window_under_threshold, min_fixed_window


def test_apply_range_add_1d_peak_example():
    values = apply_range_add_1d(
        10,
        [RangeCommand1D(0, 4, 3), RangeCommand1D(3, 7, 7), RangeCommand1D(5, 8, 1)],
    )
    assert values == [3, 3, 3, 10, 10, 8, 8, 8, 1, 0]


def test_min_fixed_window_relay_example():
    assert min_fixed_window([3, 3, 3, 10, 10, 8, 8, 8, 1, 0], 3) == (9.0, 0)


def test_safe_corridor_example():
    assert max_window_under_threshold([4, 2, 5, 8, 1, 1, 2, 3, 9, 1], 8)[:3] == (4, 4, 7)


def test_apply_rect_add_2d_overlap_example():
    grid = apply_rect_add_2d(
        5,
        5,
        [RectCommand2D(1, 1, 3, 3, 10), RectCommand2D(2, 2, 4, 4, 5)],
    )
    assert argmax_first_2d(grid) == (15.0, 2, 2)
