"""Shared parameter validation utilities for Titans modules.

Shared validation utilities following the DRY principle.
- Extract common functionality into reusable components
- Create abstractions for repeated logic

Note: This module is for validating function/constructor parameters,
not for ML training/validation dataset splits.
"""

import jax


def validate_positive(value: int | float, name: str) -> None:
    """Validate that a value is positive.

    Args:
        value: The value to validate
        name: Name of the parameter for error message

    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        msg = f"{name} must be positive, got {value}"
        raise ValueError(msg)


def validate_range(
    value: float,
    name: str,
    min_val: float,
    max_val: float,
) -> None:
    """Validate that a value is within a range [min_val, max_val].

    Args:
        value: The value to validate
        name: Name of the parameter for error message
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Raises:
        ValueError: If value is outside the range
    """
    if not min_val <= value <= max_val:
        msg = f"{name} must be in [{min_val},{max_val}], got {value}"
        raise ValueError(msg)


def validate_3d_input(x: jax.Array, expected_d_model: int | None = None) -> None:
    """Validate input tensor is 3D with optional d_model check.

    Args:
        x: Input tensor to validate (expected shape: batch, seq_len, d_model)
        expected_d_model: Expected last dimension size (optional)

    Raises:
        ValueError: If input is not 3D or d_model doesn't match
    """
    if x.ndim != 3:
        msg = f"Expected 3D input (batch, seq_len, d_model), got {x.shape}"
        raise ValueError(msg)
    if expected_d_model is not None and x.shape[-1] != expected_d_model:
        msg = f"Expected d_model={expected_d_model}, got {x.shape[-1]}"
        raise ValueError(msg)
