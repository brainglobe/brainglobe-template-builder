"""Integration test for standardise -> preprocess pipeline."""

from pathlib import Path

from brainglobe_template_builder.preprocess import preprocess
from brainglobe_template_builder.standardise import standardise
from brainglobe_template_builder.utils.preproc_config import (
    MaskConfig,
    PreprocConfig,
)


def test_standardise_preprocess(source_csv_no_masks):
    """Test standardise followed by preprocess.

    Runs standardise and preprocess and checks that expected txt output files:
    - are created for both brain and mask and are readable
    - contain file paths that point to existing files
    """

    output_dir = source_csv_no_masks.parents[1]
    standardise(source_csv_no_masks, output_dir, output_vox_size=50)
    standardised_csv = output_dir / "standardised" / "standardised_images.csv"
    config = PreprocConfig(output_dir=output_dir, mask=MaskConfig())
    preprocess(standardised_csv, config)

    preprocessed_dir = output_dir / "preprocessed"
    for file_type in ["brain", "mask"]:
        paths_file = preprocessed_dir / f"all_processed_{file_type}_paths.txt"
        assert paths_file.exists()
        with open(paths_file) as f:
            for line in f:
                assert Path(line.strip()).exists()
