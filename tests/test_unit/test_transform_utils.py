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
    data = np.random.rand(10, 100, 100)  # Random image stack
    return da.from_array(data, chunks=(1, 100, 100))


@pytest.fixture()
def not_slicewise_stack():
    """Create a dask array representing an image stack"""
    data = np.random.rand(10, 100, 100)  # Random image stack
    return da.from_array(data, chunks=(2, 100, 100))


def test_downsample_anisotropic_image_stack(stack):
    """Test that downsampling with dask gives same as without."""
    xy_downsampling = 20
    z_downsampling = 2

    downsampled_stack = downsample_anisotropic_image_stack(
        stack, xy_downsampling, z_downsampling
    )

    assert downsampled_stack.shape == (5, 5, 5)

    expected = transform.downscale_local_mean(
        stack.compute(), (1, xy_downsampling, xy_downsampling)
    )
    expected = transform.downscale_local_mean(expected, (z_downsampling, 1, 1))
    assert np.all(
        downsampled_stack == expected
    ), "dask downsampling does not match expected skimage result"


def test_downsample_anisotropic_image_stack_raises(not_slicewise_stack):
    with pytest.raises(AssertionError) as e:
        downsample_anisotropic_image_stack(
            not_slicewise_stack, xy_downsampling=20, z_downsampling=2
        )
    assert e.match("not chunked slice-wise!")
