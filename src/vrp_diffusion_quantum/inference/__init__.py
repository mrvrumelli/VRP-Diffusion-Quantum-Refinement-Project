"""Inference routines for matrix prediction and route solving."""

from vrp_diffusion_quantum.inference.predict_matrix import (
    MatrixPredictionResult,
    evaluate_full_chain_sampling,
    example_to_model_inputs,
    load_denoiser_checkpoint,
    predict_matrix_one_shot,
    sample_constraint_matrix,
    select_examples_by_size,
    symmetrize_zero_diagonal,
)

__all__ = [
    "MatrixPredictionResult",
    "evaluate_full_chain_sampling",
    "example_to_model_inputs",
    "load_denoiser_checkpoint",
    "predict_matrix_one_shot",
    "sample_constraint_matrix",
    "select_examples_by_size",
    "symmetrize_zero_diagonal",
]
