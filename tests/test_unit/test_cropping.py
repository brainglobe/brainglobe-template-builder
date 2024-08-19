import numpy as np
import pytest

from brainglobe_template_builder.preproc.cropping import crop_to_mask


def test_crop_to_mask_invalid_stack_and_mask():
    stack = np.zeros((10, 10, 10))
    mask = np.zeros((20, 20, 20))
    with pytest.raises(AssertionError) as e:
        _ = crop_to_mask(stack, mask)
    assert str(e.value) == "Stack and mask must have the same shape."


def test_crop_to_mask_invalid_mask():
    stack = np.ones((10, 10, 10))
    mask = np.zeros((10, 10, 10))
    with pytest.raises(AssertionError) as e:
        _ = crop_to_mask(stack, mask)
    assert (
        str(e.value)
        == "The mask is invalid because it does not contain foreground."
    )


def test_simple_crop_to_mask():
    stack = np.ones((10, 10, 10))
    mask = np.zeros((10, 10, 10))
    mask[3:7, 3:7, 3:7] = 1
    cropped_stack, cropped_mask = crop_to_mask(stack, mask)
    assert cropped_stack.shape == (4, 4, 4)
    assert cropped_mask.shape == (4, 4, 4)
    assert np.all(cropped_stack == stack[3:7, 3:7, 3:7])
    assert np.all(cropped_mask == mask[3:7, 3:7, 3:7])


@pytest.mark.parametrize("padding", [1, 5, 10])
def test_padding(padding):
    stack = np.ones((10, 10, 10))
    mask = np.ones((10, 10, 10))
    cropped_stack, cropped_mask = crop_to_mask(stack, mask, padding=padding)
    assert cropped_stack.shape == tuple([s + 2 * padding for s in stack.shape])
    assert cropped_mask.shape == tuple([s + 2 * padding for s in stack.shape])
    assert np.all(
        cropped_stack[padding:-padding, padding:-padding, padding:-padding]
        == stack
    )
    assert np.all(
        cropped_mask[padding:-padding, padding:-padding, padding:-padding]
        == mask
    )
    assert np.all(cropped_mask[0:padding, :, :] == 0)
    assert np.all(cropped_mask[-padding:, :, :] == 0)
    assert np.all(cropped_mask[:, 0:padding, :] == 0)
    assert np.all(cropped_mask[:, -padding:, :] == 0)
    assert np.all(cropped_mask[:, :, 0:padding] == 0)
    assert np.all(cropped_mask[:, :, -padding:] == 0)


def test_crop_to_full_mask_does_nothing():
    stack = np.ones((10, 10, 10))
    mask = np.ones((10, 10, 10))
    cropped_stack, cropped_mask = crop_to_mask(stack, mask)
    assert cropped_stack.shape == (10, 10, 10)
    assert cropped_mask.shape == (10, 10, 10)
    assert np.all(cropped_stack == stack)
    assert np.all(cropped_mask == mask)


def test_crop_to_mask_with_padding():
    stack = np.ones((10, 10, 10))
    mask = np.zeros((10, 10, 10))
    mask[3:7, 3:7, 3:7] = 1
    padding = 2
    cropped_stack, cropped_mask = crop_to_mask(stack, mask, padding=padding)
    assert cropped_stack.shape == (8, 8, 8)
    assert cropped_mask.shape == (8, 8, 8)
    assert np.all(
        cropped_stack[padding:-padding, padding:-padding, padding:-padding]
        == stack[3:7, 3:7, 3:7]
    )
    assert np.all(
        cropped_mask[padding:-padding, padding:-padding, padding:-padding]
        == mask[3:7, 3:7, 3:7]
    )
