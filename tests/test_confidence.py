import pytest

from vrp_diffusion_quantum.eval.confidence import bootstrap_mean_interval


def test_bootstrap_mean_interval_is_deterministic_and_contains_estimate() -> None:
    first = bootstrap_mean_interval([1.0, 2.0, 3.0, 10.0], num_resamples=2_000, seed=17)
    second = bootstrap_mean_interval([1.0, 2.0, 3.0, 10.0], num_resamples=2_000, seed=17)

    assert first == second
    assert first.estimate == 4.0
    assert first.lower <= first.estimate <= first.upper
    assert first.num_observations == 4


def test_bootstrap_single_observation_is_degenerate() -> None:
    interval = bootstrap_mean_interval([3.5], confidence_level=0.9, seed=2)
    assert interval.lower == interval.estimate == interval.upper == 3.5


@pytest.mark.parametrize(
    ("values", "kwargs", "message"),
    [
        ([], {}, "non-empty"),
        ([1.0, float("nan")], {}, "finite"),
        ([1.0], {"confidence_level": 1.0}, "confidence_level"),
        ([1.0], {"num_resamples": 0}, "num_resamples"),
        ([1.0], {"resample_batch_size": 0}, "resample_batch_size"),
    ],
)
def test_bootstrap_mean_interval_rejects_invalid_inputs(
    values: list[float], kwargs: dict[str, float | int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        bootstrap_mean_interval(values, **kwargs)
