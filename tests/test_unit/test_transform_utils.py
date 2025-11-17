import dask.array as da
import numpy as np
import pytest
from skimage import transform

from brainglobe_template_builder.preproc.transform_utils import (
    downsample_anisotropic_image_stack,
    downsample_anisotropic_stack_to_isotropic,
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


@pytest.mark.parametrize(
    "input_vox_sizes,output_vox_size,expected_shape,expected_warning",
    [
        pytest.param(
            [25, 25, 25], 50, (5, 50, 50), None, id="downsampling [2x, 2x, 2x]"
        ),
        pytest.param(
            [5, 10, 50],
            50,
            (1, 20, 100),
            None,
            id="downsampling [10x, 5x, 1x]",
        ),
        pytest.param(
            [50, 25, 25],
            75,
            (7, 33, 33),
            "shape is (7, 33, 33), with voxel size [71.429, 75.758, 75.758].",
            id="downsampling [1.5x, 3x, 3x]",
        ),
        pytest.param(
            [50, 25, 25],
            85,
            (6, 29, 29),
            "shape is (6, 29, 29), with voxel size [83.333, 86.207, 86.207].",
            id="downsampling [1.7x, 3.4x, 3.4x]",
        ),
    ],
)
def test_downsample_anisotropic_stack_to_isotropic(
    caplog,
    stack,
    input_vox_sizes,
    output_vox_size,
    expected_shape,
    expected_warning,
):
    downsampled_stack = downsample_anisotropic_stack_to_isotropic(
        stack, input_vox_sizes, output_vox_size
    )

    assert downsampled_stack.shape == expected_shape
    if expected_warning is not None:
        assert expected_warning in caplog.text

    # Check processing directly with skimage is same as dask based result
    downsampling_factors = [
        vox_size / output_vox_size for vox_size in input_vox_sizes
    ]
    expected = transform.rescale(
        stack.compute(), (1, downsampling_factors[1], downsampling_factors[2])
    )
    expected = transform.rescale(expected, (downsampling_factors[0], 1, 1))
    assert np.all(
        downsampled_stack == expected
    ), "dask downsampling does not match expected skimage result"


def test_downsample_anisotropic_image_stack_raises(not_slicewise_stack):
    with pytest.raises(AssertionError, match="not chunked by plane!"):
        downsample_anisotropic_image_stack(
            not_slicewise_stack, in_plane_factor=20, axial_factor=2
        )
