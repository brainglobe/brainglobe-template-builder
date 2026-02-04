from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
from skimage import measure

from brainglobe_template_builder.utils.transform_utils import (
    apply_transform,
    get_rotation_from_vectors,
)

NAPARI_AXIS_ORDER = "zyx"


class MidplaneEstimator:
    """Class to estimates points along the midplane of a 3D object, based on a
    binary mask of the object."""

    def __init__(
        self,
        mask: np.ndarray,
        symmetry_axis: Literal["x", "y", "z"] = "x",
    ):
        """Initialise the estimator.

        Parameters
        ----------
        mask : np.ndarray
            A 3D binary mask of the object.
        symmetry_axis : str
            Axis of symmetry, one of 'x', 'y', and 'z'.
            Defaults to 'x'. For brains, this would be the left-right axis.
            Keep in mind that the axis order is zyx in napari.
        """
        self.mask = mask
        self.symmetry_axis = symmetry_axis

        self._validate_inputs()
        self.symmetry_axis_idx = NAPARI_AXIS_ORDER.index(symmetry_axis)

    def _validate_inputs(self):
        """Validate the inputs to the aligner."""

        if self.mask.ndim != 3:
            raise ValueError("Mask must be 3D")
        if self.symmetry_axis not in ["x", "y", "z"]:
            raise ValueError("Symmetry axis must be one of 'x', 'y', or 'z'")
        try:
            self.mask = self.mask.astype(bool)
        except ValueError:
            raise ValueError("Mask must be binary")

    def _get_mask_properties(self):
        """Get properties of the mask, specifically the centroid and the
        dimensions of the mask's bounding box (in pixels)."""
        props = measure.regionprops(measure.label(self.mask))[0]
        self.centroid = np.array(props.centroid)
        self.bbox = np.array(props.bbox).reshape(2, 3).T

    def get_points(self):
        """Estimate 9 points along the midplane of the mask.

        The midplane is initialised as the plane perpendicular to the symmetry
        axis that contains the centroid of the mask. The midplane is then
        intersected with 3 slices along each of the other two axes, resulting
        in 9 points.
        """

        self._get_mask_properties()
        # Find middle of the symmetry axis
        mid_symmetry_axis = self.centroid[self.symmetry_axis_idx]
        # Find slices at 1/4, 2/4, and 3/4 of the mask extents along the
        # other two (non-symmetry) axes
        a, b = [i for i in range(3) if i != self.symmetry_axis_idx]
        mask_extents = self.bbox[:, 1] - self.bbox[:, 0]  # (3,) per axis
        other_planes_a = [
            self.bbox[a, 0] + mask_extents[a] / 4 * i for i in [1, 2, 3]
        ]
        other_planes_b = [
            self.bbox[b, 0] + mask_extents[b] / 4 * i for i in [1, 2, 3]
        ]

        # Create 9 points by combining all pairs of coordinates from axes a and
        # b, keeping the symmetry axis at its midpoint.
        points = []
        for coor_axis_a, coor_axis_b in product(
            other_planes_a, other_planes_b
        ):
            point = np.zeros(3)
            point[self.symmetry_axis_idx] = mid_symmetry_axis
            point[a] = coor_axis_a
            point[b] = coor_axis_b
            points.append(point)

        self.points = np.array(points)
        return self.points


class MidplaneAligner:
    """Class for aligning a given plane (as defined by a set of 3D points)
    to the midplane of an image along a given symmetry axis."""

    def __init__(
        self,
        image: np.ndarray,
        points: np.ndarray,
        symmetry_axis: Literal["x", "y", "z"] = "x",
    ):
        """Initialise the aligner.

        Parameters
        ----------
        image : np.ndarray
            A 3D image to align.
        points : np.ndarray
            An array of shape (n_points, 3) containing 3D point coordinates.
            At least 3 points are required, and they must not be colinear.
        symmetry_axis : str
            Axis of symmetry, one of 'x', 'y', and 'z'.
            Defaults to 'x'. For brains, this would be the left-right axis.
            Keep in mind that the axis order is zyx in napari.
        """
        self.image = image
        self.points = points
        self.symmetry_axis = symmetry_axis

        self._validate_inputs()
        self.symmetry_axis_idx = NAPARI_AXIS_ORDER.index(symmetry_axis)

    def _validate_inputs(self):
        """Validate the inputs to the aligner."""
        if self.image.ndim != 3:
            raise ValueError("Image must be 3D")
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("Points must be an array of shape (n_points, 3)")
        if self.points.shape[0] < 3:
            raise ValueError("At least 3 points are required")
        if np.linalg.matrix_rank(self.points) < 3:
            raise ValueError("Points must not be colinear")
        if self.symmetry_axis not in ["x", "y", "z"]:
            raise ValueError("Symmetry axis must be one of 'x', 'y', or 'z'")

    def _fit_plane_to_points(self):
        """Fit a plane to the points.

        The plane is fitted using SVD, and the normal vector is computed as the
        last row of the V matrix. The normal vector is then inverted if it
        points in the opposite direction of the symmetry axis unit vector.
        """
        # Use SVD to get the normal vector to the plane
        centroid = np.mean(self.points, axis=0)
        _, _, vh = np.linalg.svd(self.points - centroid)
        normal_vector = vh[-1]

        # Unit vector along the symmetry axis
        symmetry_axis_vector = np.zeros(3)
        symmetry_axis_vector[self.symmetry_axis_idx] = 1

        # invert the normal vector if it points in the opposite direction of
        # symmetry axis unit vector
        if np.dot(normal_vector, symmetry_axis_vector) < 0:
            normal_vector = -normal_vector

        self.centroid = centroid
        self.normal_vector = normal_vector
        self.symmetry_axis_vector = symmetry_axis_vector

    def _compute_transform(self):
        """Find the transformation matrix that aligns the plane defined by the
        points to the midplane of the image along the symmetry axis.
        """
        self._fit_plane_to_points()

        # Compute the necessary transforms
        # 1. translate to origin (so that centroid is at origin)
        translation_to_origin = np.eye(4)
        translation_to_origin[:3, 3] = -self.centroid
        # 2. rotate to align fitted plane with symmetry axis
        rotation = np.eye(4)
        rotation[:3, :3] = get_rotation_from_vectors(
            self.normal_vector, self.symmetry_axis_vector
        )
        # 3. translate to mid-axis (so that centroid is at middle of axis)
        translation_to_mid_axis = np.eye(4)
        offset = (
            self.image.shape[self.symmetry_axis_idx] / 2
            - self.centroid[self.symmetry_axis_idx]
        ) * self.symmetry_axis_vector
        translation_to_mid_axis[:3, 3] = self.centroid + offset
        # Combine the transforms
        self.transform = (
            translation_to_mid_axis @ rotation @ translation_to_origin
        )

    def transform_image(self, image: np.ndarray | None = None):
        """Transform the image using the transformation matrix.

        Parameters
        ----------
        image : np.ndarray
            The image to transform. If None, the image passed to the
            constructor is used.
        """
        if not hasattr(self, "transform"):
            self._compute_transform()
        if image is None:
            image = self.image
        self.transformed_image = apply_transform(image, self.transform)
        return self.transformed_image

    def save_transform(self, dest_path: Path):
        """Save the midplane alignment transform to a text file."""
        if not hasattr(self, "transform"):
            raise ValueError("Please align the image to the midplane first")
        np.savetxt(dest_path, self.transform)

    def label_halves(self, image: np.ndarray) -> np.ndarray:
        """Label each half of the image along the symmetry axis with different
        integer values, to help diagnose issues with the splitting process.

        Parameters
        ----------
        image : np.ndarray
            The image to label. This should be aligned along the symmetry axis.

        Returns
        -------
        np.ndarray
            An array of the same shape as the input image, with each half
            labelled with a different integer value (2 and 3).
            The output data type is uint8.
        """
        axi = self.symmetry_axis_idx
        axis_len = image.shape[axi]
        half_len = axis_len // 2
        labelled_halves = np.zeros_like(image, dtype=np.uint8)

        # Create slicing objects for each half
        slicer_half1 = [slice(None)] * image.ndim
        slicer_half1[axi] = slice(0, half_len)
        slicer_half2 = [slice(None)] * image.ndim
        slicer_half2[axi] = slice(half_len, axis_len)

        # Apply different integer values to each half
        labelled_halves[tuple(slicer_half1)] = 2
        labelled_halves[tuple(slicer_half2)] = 3

        return labelled_halves
