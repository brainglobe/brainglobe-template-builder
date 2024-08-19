import numpy as np


def crop_to_mask(
    stack: np.ndarray, mask: np.ndarray, padding: np.uint8 = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crops stack and mask to the mask extent, and pad with zeros.

    Args:
        Stack (np.ndarray): Stack
        Mask (np.ndarray): Mask
        padding (np.uint8):
            number of pixels to pad with on all sides. Default is 0.

    Returns:
        tuple[np.ndarray, np.ndarray]: the cropped, padded stack and mask.
    """
    assert (
        stack.shape == mask.shape
    ), "Stack and mask must have the same shape."
    assert not np.all(
        mask == 0
    ), "The mask is invalid because it does not contain foreground."
    # Find the bounding box of the mask
    mask_indices = np.where(mask)
    min_z = np.min(mask_indices[0])
    max_z = np.max(mask_indices[0])
    min_y = np.min(mask_indices[1])
    max_y = np.max(mask_indices[1])
    min_x = np.min(mask_indices[2])
    max_x = np.max(mask_indices[2])

    # Crop the stack and mask to the bounding box
    stack = stack[min_z : max_z + 1, min_y : max_y + 1, min_x : max_x + 1]
    mask = mask[min_z : max_z + 1, min_y : max_y + 1, min_x : max_x + 1]
    if padding:
        stack = np.pad(
            stack,
            ((padding, padding), (padding, padding), (padding, padding)),
            mode="constant",
        )
        mask = np.pad(
            mask,
            ((padding, padding), (padding, padding), (padding, padding)),
            mode="constant",
        )
    return stack, mask
