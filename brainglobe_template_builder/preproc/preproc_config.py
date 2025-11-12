from pathlib import Path

from pydantic import BaseModel, ConfigDict


class MaskConfig(BaseModel):
    gaussian_sigma: float = 3
    threshold_method: str = "triangle"
    closing_size: int = 5
    erode_size: int = 0


class PreprocConfig(BaseModel):
    derivatives_dir: Path
    resolution_mm: list[float]
    mask: MaskConfig
    pad_pixels: int = 5

    model_config = ConfigDict(extra="forbid")
