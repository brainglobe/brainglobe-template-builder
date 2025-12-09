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


@pytest.mark.parametrize(
    "plotting_func",
    [
        plot_grid,
        plot_orthographic,
    ],
)
@pytest.mark.parametrize(
    "save_path,expected_paths",
    [
        pytest.param(
            "test_image.png",
            ["test_image.png", "test_image.pdf"],
            id="save path provided",
        ),
        pytest.param(None, [], id="no save path provided"),
    ],
)
def test_save_path(tmp_path, plotting_func, save_path, expected_paths):
    """Test plots are saved to the correct location
    (if a save path is provided)."""

    if save_path is not None:
        save_path = tmp_path / save_path
        expected_paths = [tmp_path / path for path in expected_paths]

    rng = np.random.default_rng()
    image = rng.random(size=(10, 100, 100))

    plotting_func(image, save_path=save_path)
    created_paths = list(tmp_path.glob("*"))

    assert sorted(created_paths) == sorted(expected_paths)
