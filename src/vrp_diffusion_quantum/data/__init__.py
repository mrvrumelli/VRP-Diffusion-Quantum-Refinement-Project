"""Data generation, labeling, and dataset loading utilities.

Import concrete symbols from the submodules, e.g.::

    from vrp_diffusion_quantum.data.generate_cvrp import load_dataset
    from vrp_diffusion_quantum.data.solve_cvrp import solve_dataset

This package ``__init__`` intentionally avoids eager imports so
``python -m vrp_diffusion_quantum.data.generate_cvrp`` can run cleanly.
"""
