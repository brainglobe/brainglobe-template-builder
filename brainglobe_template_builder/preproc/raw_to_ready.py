from pathlib import Path

import pandas as pd

from brainglobe_template_builder.preproc.preproc_config import PreprocConfig


def raw_to_ready(input_csv: Path, config_file: Path) -> None:

    input_df = pd.read_csv(input_csv)
    # TODO - Validate input csv

    config_json = config_file.read_text()
    config = PreprocConfig.model_validate_json(config_json)

    # TODO - processing
    print(input_df, config)
