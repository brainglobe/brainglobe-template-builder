from pathlib import Path
from typing import Sequence

import pandas as pd


def validate_file_extension(
    file_path: str | Path, expected_extension: str
) -> None:
    """Check if file has the expected extension.

    Parameters
    ----------
    file_path : str or Path
        Path to the file to validate
    expected_extension : str
        Expected file extension (with or without leading dot)

    Raises
    ------
    ValueError
        If the file extension does not match the expected extension
    """
    file_path = Path(file_path)
    if not expected_extension.startswith("."):
        expected_extension = "." + expected_extension
    if file_path.suffix.lower() != expected_extension.lower():
        raise ValueError(
            f"File must have {expected_extension} extension, "
            f"got: {file_path.suffix}"
        )


def validate_required_columns(
    column_names: Sequence[str], required_columns: Sequence[str]
) -> None:
    """Check if all required columns are present.

    Parameters
    ----------
    column_names : Sequence[str]
        List of column names present in the data
    required_columns : Sequence[str]
        List of required column names
    """
    for required_column in required_columns:
        if required_column not in column_names:
            missing = [c for c in required_columns if c not in column_names]
            column_str = "Column" if len(missing) == 1 else "Columns"
            name_str = "name" if len(missing) == 1 else "names"
            missing_list = ", ".join(f"'{col}'" for col in missing)
            raise ValueError(
                f"{column_str} with {name_str} {missing_list} required "
                f"but missing from source CSV."
            )


def validate_column_names_format(column_names: Sequence[str]) -> None:
    """Check if column names don't contain whitespace.

    Parameters
    ----------
    column_names : Sequence[str]
        List of column names to validate
    """
    for column_name in column_names:
        if any(c.isspace() for c in column_name):
            raise ValueError(
                f"Column name '{column_name}' contains whitespace."
            )


def validate_column_names_unique(column_names: Sequence[str]) -> None:
    """Check if column names are unique.

    Parameters
    ----------
    column_names : Sequence[str]
        List of column names to validate
    """
    if len(set(column_names)) != len(column_names):
        raise ValueError("Column names of source CSV are not unique.")


def validate_input_csv(input_csv_path: str | Path) -> None:
    """
    Validate input CSV.

    Checks:
    - whether file has csv extension
    - columns
        - all required columns are present
        - column names do not contain spaces
        - column names are unique

    Parameters
    ----------
    input_csv_path : str or Path
        Path to the input CSV file
    """

    required_columns = [
        "subject_id",
        "resolution_z",
        "resolution_x",
        "resolution_y",
        "origin",
        "source_filepath",
    ]

    # Validate file extension
    validate_file_extension(input_csv_path, ".csv")

    # Read CSV to get column names
    df = pd.read_csv(input_csv_path, nrows=0)
    column_names = df.columns.tolist()

    # Validate columns
    validate_required_columns(column_names, required_columns)
    validate_column_names_format(column_names)
    validate_column_names_unique(column_names)
