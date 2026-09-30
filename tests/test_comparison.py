from dataclasses import replace

import pytest

from vrp_diffusion_quantum.eval.comparison import (
    InstanceResult,
    assert_disjoint_sources,
    cost_gaps,
    metric_value,
    paired_bootstrap,
)


def test_gap_definitions_differ_on_unequal_references() -> None:
    gaps = cost_gaps([2, 9], [1, 9])
    assert gaps == {"mean_instance_gap_percent": 50.0, "ratio_total_gap_percent": 10.0}


def test_graph_f1_and_paired_bootstrap() -> None:
    left = InstanceResult("a", 20, 1, 0, 0, 2, 1, True, 1)
    right = InstanceResult("b", 20, 0, 0, 8, 9, 9, True, 1)
    assert metric_value([left, right], "f1") == pytest.approx(0.2)
    interval = paired_bootstrap([left, right], [right, left], num_resamples=100)
    assert interval.estimate == interval.lower == interval.upper == 0
    single = paired_bootstrap([left], [replace(left, true_positive=0, false_negative=1)])
    assert single.estimate == single.lower == single.upper == 1
    with pytest.raises(ValueError, match="duplicate"):
        paired_bootstrap([left, left])
    with pytest.raises(ValueError, match="same unique"):
        paired_bootstrap([left], [right])


def test_source_overlap_is_rejected() -> None:
    assert_disjoint_sources({"train"}, {"val"})
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_sources({"source"}, {"source"})
