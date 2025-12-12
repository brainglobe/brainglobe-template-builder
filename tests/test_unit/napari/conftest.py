import numpy as np
import pytest


@pytest.fixture()
def stack() -> np.ndarray:
    """Create 50x50x50 stack with a small off-centre object (value 0.5)."""
    stack = np.zeros((50, 50, 50))
    stack[10:30, 10:30, 10:30] = 0.5
    return stack


@pytest.fixture()
def points() -> np.ndarray:
    """Two example 3D points."""

    return np.array(
        [
            [2, 3, 4],
            [5, 6, 7],
        ]
    )
