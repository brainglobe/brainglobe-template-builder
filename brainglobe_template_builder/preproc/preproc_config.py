from enum import Enum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class ThresholdMethod(str, Enum):
    TRIANGLE = "triangle"
    OTSU = "otsu"
    ISODATA = "isodata"


class MaskConfig(BaseModel):
    gaussian_sigma: Annotated[float, Field(ge=0)] = 3
    threshold_method: ThresholdMethod = ThresholdMethod.TRIANGLE
    closing_size: Annotated[int, Field(ge=0)] = 5
    erode_size: Annotated[int, Field(ge=0)] = 0


class PreprocConfig(BaseModel):
    derivatives_dir: Path
    resolution_z: Annotated[float, Field(gt=0)]
    resolution_y: Annotated[float, Field(gt=0)]
    resolution_x: Annotated[float, Field(gt=0)]
    mask: MaskConfig
    pad_pixels: Annotated[int, Field(ge=0)] = 5

    model_config = ConfigDict(extra="forbid")
