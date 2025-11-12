from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ThresholdMethod(str, Enum):
    TRIANGLE = "triangle"
    OTSU = "otsu"
    ISODATA = "isodata"


class MaskConfig(BaseModel):
    gaussian_sigma: float = 3
    threshold_method: ThresholdMethod = ThresholdMethod.TRIANGLE
    closing_size: int = 5
    erode_size: int = 0


class PreprocConfig(BaseModel):
    derivatives_dir: Path
    resolution_z: float
    resolution_y: float
    resolution_x: float
    mask: MaskConfig
    pad_pixels: int = 5

    model_config = ConfigDict(extra="forbid")
