from pathlib import Path

import pandas as pd
import pytest

from brainglobe_template_builder.validate import (
    validate_file_extension,
)


@pytest.fixture
def valid_df():
    """Create a DataFrame with minimal valid input data."""
    data = {
        "species": "Zebra finch",
        "sex": ["F", "M", "F", "M", "F"],
        "subject_id": [f"ZF{i+1}" for i in range(5)],
        "resolution_z": 50,
        "resolution_y": 50,
        "resolution_x": 50,
        "channel": "green",
        "origin": ["PSL", "PSL", "LAS", "SAL", "LPI"],
        "source_filepath": [f"/path/to/atlas-{i+1}" for i in range(5)],
    }
    return pd.DataFrame(data)


@pytest.fixture
def df_missing_origin(valid_df):
    return valid_df.drop(columns=["origin"])


@pytest.mark.parametrize(
    ["filepath", "extension", "is_valid"],
    [
        pytest.param("data.csv", ".csv", True, id="expected extension (str)"),
        pytest.param(
            Path("data.csv"), ".csv", True, id="expected extension (Path)"
        ),
        pytest.param("data.txt", ".csv", False, id="unexpected extension"),
    ],
)
def test_validate_file_extension(filepath, extension, is_valid):
    """Test validate_file_extension.

    This test verifies that the validate_file_extension function correctly:
    - Accepts files when they have the expected extension
    - Raises a ValueError when files do not have the expected extension
    - Includes an appropriate error message when validation fails
    """
    if is_valid:
        validate_file_extension(filepath, extension)
    else:
        with pytest.raises(ValueError) as excinfo:
            validate_file_extension(filepath, extension)
        assert f"File must have {extension} extension" in str(excinfo.value)
