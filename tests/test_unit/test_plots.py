import numpy as np
import pytest

from brainglobe_template_builder.plots import plot_grid, plot_orthographic


@pytest.mark.parametrize(
    "plotting_func",
    [
        plot_grid,
        plot_orthographic,
    ],
)
def test_img_and_overlay_different_sizes(plotting_func):
    img = np.ones(shape=(3, 3, 3))
    overlay = np.ones(shape=(3, 3, 4))

    with pytest.raises(ValueError, match="Overlay dimensions must match img"):
        plotting_func(img, overlay=overlay)
