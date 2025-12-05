from pathlib import Path

import pandas as pd
import pytest

from brainglobe_template_builder.validate import (
    validate_column_names_format,
    validate_column_names_unique,
    validate_file_extension,
    validate_input_csv,
    validate_required_columns,
)


@pytest.fixture
def valid_df():
    """Create a DataFrame with minimal valid input data.

    The format of this DataFrame matches input CSV file is
    """
    data = {
        "species": "Zebra finch",
        "sex": ["F", "M", "F", "M", "F"],
        "subject_id": [f"ZF{i + 1}" for i in range(5)],
        "resolution_0": 50,
        "resolution_1": 50,
        "resolution_2": 50,
        "channel": "green",
        "origin": ["PSL", "PSL", "LAS", "SAL", "LPI"],
        "filepath": [f"/path/to/atlas-{i + 1}" for i in range(5)],
    }
    return pd.DataFrame(data)


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


@pytest.mark.parametrize(
    ["columns", "required", "error_message"],
    [
        pytest.param(
            ["col1", "col2"], ["col1"], None, id="valid (extra col present)"
        ),
        pytest.param(
            ["col1"],
            ["col1", "col2"],
            "Column with name 'col2' required but missing from input CSV.",
            id="invalid (col2 missing)",
        ),
        pytest.param(
            [],
            ["col1", "col2"],
            "Columns with names 'col1', 'col2' required but missing from"
            " input CSV.",
            id="invalid (multiple cols missing)",
        ),
        pytest.param(
            ["col1", "col2"],
            ["col1", "col2"],
            None,
            id="valid (all required present)",
        ),
    ],
)
def test_validate_required_columns(columns, required, error_message):
    """Tests whether all required columns are present."""
    if error_message:
        with pytest.raises(ValueError, match=error_message):
            validate_required_columns(columns, required)
    else:
        validate_required_columns(columns, required)


@pytest.mark.parametrize(
    ["column_names", "error_message"],
    [
        pytest.param(["col1", "col2"], None, id="valid"),
        pytest.param(
            ["col 1", "col2"],
            "Column name 'col 1' contains whitespace.",
            id="invalid (space in name)",
        ),
        pytest.param(
            ["col1", "col\t2"],
            "Column name 'col\t2' contains whitespace.",
            id="invalid (tab in name)",
        ),
        pytest.param(
            ["col_1", "col-2", "col.3", "!$£"],
            None,
            id="valid (special chars)",
        ),
    ],
)
def test_validate_column_names_format(column_names, error_message):
    """Tests format of column names"""
    if error_message:
        with pytest.raises(ValueError, match=error_message):
            validate_column_names_format(column_names)
    else:
        validate_column_names_format(column_names)


@pytest.mark.parametrize(
    ["column_names", "error_message"],
    [
        pytest.param(["col1", "col2", "col3"], None, id="valid (all unique)"),
        pytest.param(
            ["col1", "col2", "col1"],
            "Column names of input CSV are not unique.",
            id="invalid (duplicate col1)",
        ),
        pytest.param(
            ["Col1", "col1", "COL1"],
            None,
            id="valid (case-sensitive unique)",
        ),
    ],
)
def test_validate_column_names_unique(column_names, error_message):
    """Test whether duplicate column names are recognized correctly."""
    if error_message:
        with pytest.raises(ValueError, match=error_message):
            validate_column_names_unique(column_names)
    else:
        validate_column_names_unique(column_names)


@pytest.mark.parametrize(
    ["valid"],
    [
        pytest.param(True, id="valid"),
        pytest.param(False, id="invalid"),
    ],
)
def test_validate_input_csv(mocker, valid_df, valid):
    """Test validate_input_csv with patched DataFrame.

    Invalid case: missing required 'subject_id' column.
    """
    df = valid_df if valid else valid_df.drop(columns=["subject_id"])
    mocker.patch("pandas.read_csv", return_value=df)

    if valid:
        validate_input_csv("input.csv")
    else:
        with pytest.raises(ValueError):
            validate_input_csv("input.csv")
