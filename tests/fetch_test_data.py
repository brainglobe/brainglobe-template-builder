from pathlib import Path

import pooch

POOCH_REGISTRY = pooch.create(
    path=Path(__file__).parents[1] / ".pooch",
    base_url="https://gin.g-node.org/BrainGlobe/test-data/raw/master/brainglobe-template-builder",
    registry={
        "Rat1_sub-01_T2w.nii.gz": "sha256:cd460fdbb2a2dc555ba70510b9b27d93962bc5ee59b3b610c480f27cd6a5d29c",  # noqa: E501
        "Rat1_sub-01_T2w_corrected.nii.gz": "sha256:e11b21abb4bb1e24ef6e0445156af0c6747cd3684a1cb85335b97daaeaef7a8f",  # noqa: E501
    },
    retry_if_failed=5,
)


def load_rat_image() -> Path:
    """
    Return path to test rat data.
    """
    return POOCH_REGISTRY.fetch("Rat1_sub-01_T2w.nii.gz")


def load_corrected_rat_image() -> Path:
    """
    Return path to corrected test rat data.
    Brightness correction via ants.n4_bias_field_correction.
    """
    return POOCH_REGISTRY.fetch("Rat1_sub-01_T2w_corrected.nii.gz")
