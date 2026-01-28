import dask.array as da
import numpy as np
import pytest
from skimage import transform

from brainglobe_template_builder.utils.transform_utils import (
    _downsample_anisotropic_stack_by_factors,
    downsample_anisotropic_stack_to_isotropic,
)


@pytest.fixture
def stack():
    """Create a dask array representing an image stack"""
    rng = np.random.default_rng()
    data = rng.random(size=(10, 100, 100))
    return da.from_array(data, chunks=(1, 100, 100))


@pytest.fixture
def mask():
    """Create a dask array representing a mask - only values 0 and 1."""
    data = np.zeros(shape=(10, 100, 100), dtype="float64")
    data[3:7, 25:75, 25:75] = 1
    return da.from_array(data, chunks=(1, 100, 100))


@pytest.fixture
def not_slicewise_stack(stack):
    """Create a dask array representing an image stack,
    chunked by multiple slices."""
    return stack.rechunk({0: 2, 1: 100, 2: 100})


@pytest.fixture
def partial_slicewise_stack(stack):
    """Create a dask array representing an image stack,
    chunked by part of an individual slice."""
    return stack.rechunk({0: 1, 1: 17, 2: 100})


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
        stack.compute(),
        (1, downsampling_factors[1], downsampling_factors[2]),
        order=1,
    )
    expected = transform.rescale(
        expected, (downsampling_factors[0], 1, 1), order=1
    )
    assert np.all(
        downsampled_stack == expected
    ), "dask downsampling does not match expected skimage result"


def test_downsample_mask_to_isotropic(mask):
    downsampled_mask = downsample_anisotropic_stack_to_isotropic(
        mask, [25, 25, 25], 50, mask=True
    )

    assert downsampled_mask.shape == (5, 50, 50)
    # Downsampled mask should still only contain values 0 and 1
    # i.e. no intermediate interpolated values.
    assert np.array_equal(np.unique(downsampled_mask), np.array([0, 1]))

    # Check processing directly with skimage is same as dask based result
    downsampling_factors = [0.5, 0.5, 0.5]
    expected = transform.rescale(
        mask.compute(),
        (1, downsampling_factors[1], downsampling_factors[2]),
        order=0,
        anti_aliasing=False,
    )
    expected = transform.rescale(
        expected, (downsampling_factors[0], 1, 1), order=0, anti_aliasing=False
    )
    assert np.all(
        downsampled_mask == expected
    ), "dask downsampling does not match expected skimage result"


def test_upsampling_raises_error(stack):
    with pytest.raises(ValueError, match="Upsampling would be required."):
        downsample_anisotropic_stack_to_isotropic(stack, [50, 50, 50], 25)


def test_non_planar_chunking_raises_error(not_slicewise_stack):
    with pytest.raises(
        ValueError,
        match="not chunked by entire plane! Chunks on axis 0 are \\(2,",
    ):
        downsample_anisotropic_stack_to_isotropic(
            not_slicewise_stack, [25, 25, 25], 50
        )


def test_chunks_covering_part_of_plane_raises_error(partial_slicewise_stack):
    with pytest.raises(
        ValueError,
        match="not chunked by entire plane! Chunks on axis 1 are \\(17,",
    ):
        downsample_anisotropic_stack_to_isotropic(
            partial_slicewise_stack, [25, 25, 25], 50
        )


@pytest.mark.parametrize(
    ["image_stack"],
    [
        pytest.param("image_uint16", id="preserve uint16 dtype"),
        pytest.param("image", id="preserve float64 dtype"),
    ],
)
def test_downsampling_preserves_dtype(test_stacks, image_stack):
    """Test that downsampling an image stack preserves its original dtype."""
    original_dtype = test_stacks[image_stack].dtype
    dask_stack = da.from_array(test_stacks[image_stack], chunks=(1, 50, 50))
    downsampled_stack = _downsample_anisotropic_stack_by_factors(
        dask_stack, [0.5, 0.5, 0.5], mask=False
    )
    assert downsampled_stack.dtype == original_dtype
