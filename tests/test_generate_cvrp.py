from pathlib import Path
from typing import get_args

import numpy as np
import pytest
import yaml

from vrp_diffusion_quantum.data.generate_cvrp import (
    DEFAULT_CAPACITY_BY_SIZE,
    DEMAND_HIGH,
    DEMAND_LOW,
    DEMAND_RANGES,
    CustomerMode,
    CVRPInstance,
    DemandMode,
    DepotMode,
    generate_dataset,
    generate_instance,
    load_dataset,
    main,
    resolve_capacity,
    sample_capacity_from_weights,
    save_dataset,
    suggested_capacity,
)

# --- mode-definition guards (prevent overlapping / duplicate properties) -------------------


def test_mode_names_are_unique_within_each_category() -> None:
    for mode_type in (DepotMode, CustomerMode, DemandMode):
        names = get_args(mode_type)
        assert len(names) == len(set(names)), f"duplicate name in {mode_type}: {names}"


def test_fixed_demand_ranges_are_distinct() -> None:
    ranges = list(DEMAND_RANGES.values())
    assert len(ranges) == len(set(ranges)), f"two demand modes share a range: {DEMAND_RANGES}"


def test_fixed_demand_ranges_are_declared_demand_modes() -> None:
    # Every named fixed range must be a real DemandMode (keeps the two lists in sync).
    assert set(DEMAND_RANGES).issubset(set(get_args(DemandMode)))


def test_demand_ranges_are_well_formed() -> None:
    for name, (low, high) in DEMAND_RANGES.items():
        assert 0 < low <= high, f"{name} has an invalid range {(low, high)}"


# --- shape tests ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_customers", [20, 50, 100])
def test_generate_instance_has_expected_shapes(n_customers: int) -> None:
    rng = np.random.default_rng(0)
    instance = generate_instance(n_customers, rng)

    assert instance.n_customers == n_customers
    assert instance.coords.shape == (n_customers + 1, 2)
    assert instance.demands.shape == (n_customers + 1,)
    assert instance.customer_coords.shape == (n_customers, 2)
    assert instance.customer_demands.shape == (n_customers,)


@pytest.mark.parametrize("n_customers", [20, 50, 100])
def test_generate_dataset_has_expected_shapes(n_customers: int) -> None:
    dataset = generate_dataset(n_customers, num_instances=8, seed=1)

    assert len(dataset) == 8
    assert dataset.coords.shape == (8, n_customers + 1, 2)
    assert dataset.demands.shape == (8, n_customers + 1)
    assert dataset.capacity.shape == (8,)
    assert dataset[0].n_customers == n_customers


def test_coordinates_are_normalized() -> None:
    dataset = generate_dataset(50, num_instances=16, seed=7)
    assert dataset.coords.min() >= 0.0
    assert dataset.coords.max() <= 1.0


# --- capacity / feasibility tests ----------------------------------------------------------


@pytest.mark.parametrize("n_customers", [20, 50, 100])
def test_capacity_and_demand_invariants(n_customers: int) -> None:
    dataset = generate_dataset(n_customers, num_instances=32, seed=3)
    expected_capacity = DEFAULT_CAPACITY_BY_SIZE[n_customers]

    assert np.all(dataset.capacity == expected_capacity)
    # Depot demand is always zero.
    assert np.all(dataset.demands[:, 0] == 0)
    customer_demands = dataset.demands[:, 1:]
    assert customer_demands.min() >= DEMAND_LOW
    assert customer_demands.max() <= DEMAND_HIGH
    assert customer_demands.max() <= expected_capacity


def test_normalized_demands_lie_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    instance = generate_instance(20, rng)
    normalized = instance.normalized_demands()

    assert normalized[0] == 0.0
    assert np.all(normalized[1:] > 0.0)
    assert np.all(normalized[1:] <= 1.0)


def test_resolve_capacity_rules() -> None:
    assert resolve_capacity(20, None) == 30
    assert resolve_capacity(75, None) == 45  # interpolated between 50→40 and 100→50
    assert resolve_capacity(999, None) == round(50 + 0.2 * (999 - 100))
    assert resolve_capacity(999, 25) == 25
    with pytest.raises(ValueError):
        resolve_capacity(20, 0)


def test_suggested_capacity_uses_size_floor_and_high_demand() -> None:
    # high=10 → n20:30, n50:40, n100:50
    assert suggested_capacity(20, 10) == 30
    assert suggested_capacity(50, 10) == 40
    assert suggested_capacity(100, 10) == 50
    assert suggested_capacity(75, 10) == 45
    # high demand wins when larger than the size floor
    assert suggested_capacity(20, 40) == 40
    assert suggested_capacity(100, 40) == 50


def test_validate_rejects_oversized_demand() -> None:
    coords = np.array([[0.5, 0.5], [0.1, 0.1]])
    demands = np.array([0, 100])
    instance = CVRPInstance(coords=coords, demands=demands, capacity=30.0)
    with pytest.raises(ValueError):
        instance.validate()


# --- reproducibility tests -----------------------------------------------------------------


def test_same_seed_is_byte_identical() -> None:
    first = generate_dataset(50, num_instances=10, seed=123)
    second = generate_dataset(50, num_instances=10, seed=123)

    np.testing.assert_array_equal(first.coords, second.coords)
    np.testing.assert_array_equal(first.demands, second.demands)


def test_different_seed_produces_different_instances() -> None:
    first = generate_dataset(50, num_instances=10, seed=123)
    other = generate_dataset(50, num_instances=10, seed=124)

    assert not np.array_equal(first.coords, other.coords)


def test_clustered_mode_is_reproducible() -> None:
    first = generate_dataset(50, num_instances=6, seed=5, customer_mode="clustered")
    second = generate_dataset(50, num_instances=6, seed=5, customer_mode="clustered")
    np.testing.assert_array_equal(first.coords, second.coords)


def test_spatial_stress_configs_are_disjoint_and_comparable() -> None:
    root = Path(__file__).resolve().parents[1]
    configs = [
        yaml.safe_load((root / "configs/data/cvrp_spatial_r.yaml").read_text()),
        yaml.safe_load((root / "configs/data/cvrp_spatial_c.yaml").read_text()),
        yaml.safe_load((root / "configs/data/cvrp_spatial_rc.yaml").read_text()),
    ]

    assert [cfg["customer_mode"] for cfg in configs] == [
        "random",
        "clustered",
        "random_clustered",
    ]
    assert len({cfg["seed"] for cfg in configs}) == 3
    assert len({cfg["output_dir"] for cfg in configs}) == 3
    for config in configs:
        assert config["sizes"] == [20, 50, 100]
        assert config["num_instances"] == 1000
        assert config["depot_mode"] == "random"
        assert config["demand_mode"] == "uniform"
        assert config["capacity"] is None


# --- depot distribution --------------------------------------------------------------------


def test_center_depot_mode_places_depot_at_center() -> None:
    rng = np.random.default_rng(0)
    instance = generate_instance(20, rng, depot_mode="center")
    assert np.allclose(instance.depot_coords, [0.5, 0.5])


def test_corner_depot_mode_places_depot_at_origin() -> None:
    rng = np.random.default_rng(0)
    instance = generate_instance(20, rng, depot_mode="corner")
    assert np.allclose(instance.depot_coords, [0.0, 0.0])


# --- customer distribution -----------------------------------------------------------------


@pytest.mark.parametrize("customer_mode", ["random", "clustered", "random_clustered"])
def test_customer_modes_produce_valid_normalized_coords(customer_mode: str) -> None:
    rng = np.random.default_rng(2)
    instance = generate_instance(100, rng, customer_mode=customer_mode)

    assert instance.customer_coords.shape == (100, 2)
    assert instance.coords.min() >= 0.0
    assert instance.coords.max() <= 1.0


# --- demand distribution -------------------------------------------------------------------


@pytest.mark.parametrize("demand_mode", list(DEMAND_RANGES))
def test_fixed_range_demand_modes_respect_bounds(demand_mode: str) -> None:
    low, high = DEMAND_RANGES[demand_mode]
    dataset = generate_dataset(
        50,
        num_instances=20,
        seed=11,
        demand_mode=demand_mode,  # type: ignore[arg-type]
        route_size="medium",
    )
    customer_demands = dataset.demands[:, 1:]
    assert customer_demands.min() >= low
    assert customer_demands.max() <= high


def test_unitary_demands_are_all_one() -> None:
    rng = np.random.default_rng(0)
    instance = generate_instance(50, rng, demand_mode="unitary", route_size="short")
    assert np.all(instance.customer_demands == 1)


def test_quadrant_demands_span_small_and_large() -> None:
    rng = np.random.default_rng(0)
    instance = generate_instance(200, rng, demand_mode="quadrant", route_size="long")
    demands = instance.customer_demands
    assert demands.min() >= 1
    assert demands.max() <= 100
    # The regime mixes a small (<=50) and a large (>=51) band.
    assert demands.min() <= 50
    assert demands.max() >= 51


def test_many_small_few_large_is_mostly_small() -> None:
    rng = np.random.default_rng(0)
    instance = generate_instance(300, rng, demand_mode="many_small_few_large", route_size="long")
    demands = instance.customer_demands
    fraction_small = np.mean(demands <= 10)
    assert fraction_small >= 0.5


def test_large_demand_mode_without_capacity_is_infeasible() -> None:
    rng = np.random.default_rng(0)
    # large_demand_wide_spread high floor is 100 > default capacity 30 -> must raise.
    with pytest.raises(ValueError):
        for _ in range(50):
            generate_instance(20, rng, demand_mode="large_demand_wide_spread")


def test_custom_demand_bounds_and_capacity_floor() -> None:
    rng = np.random.default_rng(1)
    instance = generate_instance(
        40,
        rng,
        demand_mode="custom",
        demand_low=10,
        demand_high=25,
        capacity=25,
    )
    assert instance.customer_demands.min() >= 10
    assert instance.customer_demands.max() <= 25
    assert instance.capacity == 25.0


def test_capacity_below_high_demand_raises() -> None:
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="cannot be lower than high demand"):
        generate_instance(
            20,
            rng,
            demand_mode="custom",
            demand_low=1,
            demand_high=40,
            capacity=30,
        )


def test_random_capacity_stays_at_or_above_high_demand() -> None:
    dataset = generate_dataset(
        30,
        num_instances=16,
        seed=9,
        demand_mode="custom",
        demand_low=5,
        demand_high=20,
        random_capacity=True,
        capacity_max=60,
    )
    assert dataset.capacity.min() >= 20
    assert dataset.capacity.max() <= 60


def test_random_capacity_respects_weight_curve() -> None:
    rng = np.random.default_rng(0)
    # Heavy weight only near 50 → samples should cluster there.
    samples = [
        sample_capacity_from_weights(rng, 20, 60, [(20, 0.01), (50, 1.0), (60, 0.01)])
        for _ in range(300)
    ]
    assert min(samples) >= 20
    assert max(samples) <= 60
    assert sum(1 for s in samples if 45 <= s <= 55) > 120

    dataset = generate_dataset(
        30,
        num_instances=40,
        seed=11,
        demand_mode="custom",
        demand_low=5,
        demand_high=20,
        random_capacity=True,
        capacity_max=60,
        capacity_weights=[(20, 0.01), (55, 1.0), (60, 0.01)],
    )
    assert dataset.capacity.min() >= 20
    assert dataset.capacity.max() <= 60
    assert float(np.mean(dataset.capacity)) > 45.0


def test_random_demand_bounds_in_unit_interval() -> None:
    dataset = generate_dataset(
        25,
        num_instances=12,
        seed=3,
        demand_mode="custom",
        random_demand_bounds=True,
        random_capacity=True,
        capacity_max=200,
    )
    cust = dataset.demands[:, 1:]
    assert cust.min() >= 1
    assert cust.max() <= 100
    assert np.all(dataset.capacity >= cust.max(axis=1))


# --- route size / capacity derivation ------------------------------------------------------


def test_route_size_derives_capacity_and_scales_with_r() -> None:
    small_r = generate_dataset(50, num_instances=8, seed=4, route_size=5.0)
    large_r = generate_dataset(50, num_instances=8, seed=4, route_size=20.0)

    # Larger target route size -> larger derived capacity on average.
    assert large_r.capacity.mean() > small_r.capacity.mean()
    # Capacity must still fit every single demand.
    assert np.all(small_r.demands[:, 1:].max(axis=1) <= small_r.capacity)


# --- persistence tests ---------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    dataset = generate_dataset(
        20,
        num_instances=5,
        seed=9,
        depot_mode="center",
        customer_mode="clustered",
        demand_mode="small_demand_wide_spread",
        route_size="medium",
    )
    target = save_dataset(dataset, tmp_path / "cvrp20_test")

    assert target.name == "cvrp20_test_nodes.csv"
    assert target.is_file()
    assert (tmp_path / "cvrp20_test_instances.csv").is_file()
    loaded = load_dataset(target)

    assert loaded.n_customers == dataset.n_customers
    assert loaded.seed == dataset.seed
    assert loaded.depot_mode == "center"
    assert loaded.customer_mode == "clustered"
    assert loaded.demand_mode == "small_demand_wide_spread"
    assert loaded.route_size == "medium"
    np.testing.assert_array_equal(loaded.coords, dataset.coords)
    # Demands and capacity are stored raw, so they round-trip exactly.
    np.testing.assert_allclose(loaded.demands, dataset.demands)
    np.testing.assert_allclose(loaded.capacity, dataset.capacity)


def test_cli_writes_dataset_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "cvrp"
    main(
        [
            "--sizes",
            "20",
            "50",
            "--num",
            "4",
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
        ]
    )

    for size in (20, 50):
        assert (output_dir / f"cvrp{size}_nodes.csv").is_file()
        assert (output_dir / f"cvrp{size}_instances.csv").is_file()
        dataset = load_dataset(output_dir / f"cvrp{size}")
        assert len(dataset) == 4
        assert dataset.n_customers == size

    # Each size uses an independent seed derived from the base seed.
    seed_20 = load_dataset(output_dir / "cvrp20").seed
    seed_50 = load_dataset(output_dir / "cvrp50").seed
    assert seed_20 != seed_50


def test_cli_reads_config_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        "config_name: test\n"
        "seed: 7\n"
        "sizes: [20]\n"
        "num_instances: 3\n"
        "depot_mode: center\n"
        "demand_mode: unitary\n"
        "route_size: short\n"
        f"output_dir: {output_dir.as_posix()}\n"
    )

    main(["--config", str(config_path)])

    dataset = load_dataset(output_dir / "cvrp20")
    assert len(dataset) == 3
    assert dataset.n_customers == 20
    assert dataset.depot_mode == "center"
    assert dataset.demand_mode == "unitary"
    # Depot (node 0) sits at the center for every instance.
    assert np.allclose(dataset.coords[:, 0, :], 0.5)
    # Demands are stored raw, so capacity is the real vehicle capacity.
    assert np.all(dataset.capacity > 1.0)
    # Unitary demands are equal (raw integers) within an instance.
    customer_demands = dataset.demands[:, 1:]
    assert np.all(customer_demands >= 1)
    assert np.allclose(customer_demands, np.round(customer_demands))
    assert np.allclose(customer_demands, customer_demands[:, [0]])


def test_cli_flag_overrides_config(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        "seed: 7\n"
        "sizes: [20]\n"
        "num_instances: 3\n"
        "demand_mode: unitary\n"
        "route_size: short\n"
        f"output_dir: {output_dir.as_posix()}\n"
    )

    main(["--config", str(config_path), "--demand-mode", "uniform"])

    dataset = load_dataset(output_dir / "cvrp20")
    assert dataset.demand_mode == "uniform"
    # Demands are stored raw (integers) with the real vehicle capacity.
    assert np.all(dataset.capacity > 1.0)
    customer_demands = dataset.demands[:, 1:]
    assert customer_demands.min() >= 1.0
    assert np.allclose(customer_demands, np.round(customer_demands))
    assert customer_demands.max() > customer_demands.min()
