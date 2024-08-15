import dask.array as da
import numpy as np
import pytest
from skimage import transform

from brainglobe_template_builder.preproc.transform_utils import (
    downsample_anisotropic_image_stack,
)


@pytest.fixture()
def stack():
    """Create a dask array representing an image stack"""
    rng = np.random.default_rng()
    data = rng.random(size=(10, 100, 100))
    return da.from_array(data, chunks=(1, 100, 100))


@pytest.fixture()
def not_slicewise_stack(stack):
    """Create a dask array representing an image stack"""
    return stack.rechunk({0: 2, 1: 100, 2: 100})


def test_downsample_anisotropic_image_stack(stack):
    """Test that downsampling with dask gives same as without."""
    in_plane = 20
    axial = 2

    downsampled_stack = downsample_anisotropic_image_stack(
        stack, in_plane, axial
    )

    assert downsampled_stack.shape == (5, 5, 5)

    expected = transform.downscale_local_mean(
        stack.compute(), (1, in_plane, in_plane)
    )
    expected = transform.downscale_local_mean(expected, (axial, 1, 1))
    assert np.all(
        downsampled_stack == expected
    ), "dask downsampling does not match expected skimage result"


def test_downsample_anisotropic_image_stack_raises(not_slicewise_stack):
    with pytest.raises(AssertionError, match="not chunked by plane!"):
        downsample_anisotropic_image_stack(
            not_slicewise_stack, in_plane_factor=20, axial_factor=2
        )
