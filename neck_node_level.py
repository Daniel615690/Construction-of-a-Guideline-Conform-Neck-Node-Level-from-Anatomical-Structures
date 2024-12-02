import typing

from abc import ABC, abstractmethod
import os
from pydicom import dcmread
import numpy as np
from sklearn.decomposition import PCA
from shapely import Polygon
from shapely.ops import split
from shapely.geometry.polygon import orient

from interpolation import (_find_index, _slice_wrap_around, _close_contour, interpolate_contour_in_z, clip_corners,
                           interpolate_contour_in_xy, extract_structure_endpoints)
from remove_self_intersections import clip_contour, plot_contour
from utils import _orient_positive, get_orthogonal_in_xy, closest_to_point, closest_between_points
from dicom import add_contour, extract_rt_ct_paths, get_slice_thickness


class AnatomicalStructure:
    """
    Base class for general anatomical structures, independent of orientation.

    This class models the borders of an anatomical structure using polygonal chains for each z-value.
    It provides methods to create an anatomical structure from a DICOM RT-struct file and methods to extract
    specific points or features from the DICOM RT-struct.

    The DICOM coordinate system convention is followed in this class:
    - The x-axis extends from the patient's right (-x) to left (+x).
    - The y-axis extends from the patient's anterior (front, +y) to posterior (back, -y).
    - The z-axis extends from inferior (towards feet, -z) to superior (towards head, +z).

    The class can serve as a base for more specialized anatomical structure classes,
    such as those representing structures on the left or right side of the patient.
    """

    def __init__(self, points, contour_sequences=None):
        """
        Initialize an AnatomicalStructure object.

        This class represents an anatomical structure whose border is defined by one or more polygonal chains
        (contours) in the xy-plane for each unique z-coordinate. The input data can describe the structure
        as a single contiguous polygonal chain or as multiple disjoint chains within a single xy-plane.

        When only `points` are provided, the class assumes that all points at the same z-coordinate form a
        single polygonal chain, ordered consecutively. If multiple disjoint chains exist at a given z-coordinate,
        the `contour_sequences` parameter should be used to explicitly define each chain.

        :param points: A numpy array of shape (M, 3) representing the 3D coordinates of the points that define
            the anatomical structure's border. Points belonging to the same z-coordinate should be ordered
            to form a valid polygonal chain in the xy-plane.
        :type points: np.ndarray[float]
        :param contour_sequences: Optional. A dictionary mapping each unique z-coordinate (float) to a list of
            numpy arrays. Each numpy array represents a single contour, defined as a sequence of 3D points
            (shape (N, 3)). This parameter allows for describing multiple disjoint contours in the same xy-plane.
            Contours are directed, with consecutive points connected to form open or closed loops.
        :type contour_sequences: typing.Optional[typing.Dict[float, typing.List[np.ndarray[float]]]]
        """
        self._points = np.empty((0, 3))
        self._contour_sequences = {}

        if contour_sequences is None:
            # orient points and create contour_sequences
            points_oriented = []
            for z in np.unique(points[:, 2]):
                points_in_slice = points[points[:, 2] == z]
                points_in_slice_oriented = _orient_positive(points_in_slice)

                points_oriented.extend(points_in_slice_oriented)
                self._contour_sequences[z] = [points_in_slice_oriented]
            self._points = np.array(points_oriented)
        else:
            # orient points in contour_sequences
            points_oriented = []
            for z, contour_sequence in contour_sequences.items():
                if z not in self._contour_sequences:
                    self._contour_sequences[z] = []
                for contour in contour_sequence:
                    contour_oriented = _orient_positive(contour)
                    points_oriented.extend(contour_oriented)
                    self._contour_sequences[z].append(contour_oriented)
            self._points = np.array(points_oriented)

    @classmethod
    def _extract_points_contour_sequence_from_dicom(cls, path, label):
        """
        Extract 3D points and contour sequences for an anatomical structure identified by `label`
        from a DICOM RT-struct file.

        This method reads the DICOM RT-struct file at the specified `path`, searches for the anatomical
        structure with the given `label`, and extracts its contours. Each contour is represented as a
        sequence of 3D points in the format (x, y, z). If the structure consists of multiple contours
        at the same z-coordinate, these are grouped together in the returned contour sequence.

        :param path: The file path to the DICOM RT-struct file.
        :type path: str
        :param label: The label of the anatomical structure in the DICOM RT-struct file.
        :type label: str
        :return: A tuple containing:
            1.) A numpy array of shape (M, 3) with all points defining the structure's contours.
            2.) A dictionary mapping each unique z-coordinate (float) to a list of numpy arrays,
              where each numpy array represents a single contour (shape (N, 3)) in that plane.
        :rtype: (np.ndarray[float], typing.Dict[float, typing.List[np.ndarray[float]]])
        """
        points = []
        contour_sequences = {}

        dataset = dcmread(path)
        for structureSetROISequence, ROIContourSequence in zip(
                dataset.StructureSetROISequence, dataset.ROIContourSequence):
            roi_name = structureSetROISequence.ROIName
            if roi_name == label:
                for contour in ROIContourSequence.ContourSequence:
                    contour_points = np.array(contour.ContourData).reshape(-1, 3)
                    z = contour_points[0, 2]
                    if z not in contour_sequences:
                        contour_sequences[z] = []
                    contour_sequences[z].append(contour_points)
                    points.extend(contour_points)
        points = np.array(points)

        return points, contour_sequences

    @classmethod
    def from_dicom(cls, path, label):
        """
        Create an AnatomicalStructure instance from a DICOM RT-struct file.

        This method reads a DICOM RT-struct file at the specified `path` and extracts the anatomical
        structure identified by the provided `label`.

        :param path: The file path to the DICOM RT-struct file.
        :type path: str
        :param label: The label of the anatomical structure to extract from the DICOM RT-struct file.
        :type label: str
        :return: An instance of the `AnatomicalStructure` class representing the specified structure.
        :rtype: AnatomicalStructure
        """
        points, contour_sequences = cls._extract_points_contour_sequence_from_dicom(path, label)
        return AnatomicalStructure(points, contour_sequences)

    @property
    def points(self):
        """
        Retrieve the 3D points defining the border of the anatomical structure.

        :return: A numpy array of shape (M, 3) containing the 3D coordinates of the border points.
        :rtype: np.ndarray[float]
        """
        return self._points

    def get_points(self, z):
        """
        Retrieve the points defining the border of the anatomical structure at a specific z-coordinate.

        This method filters and returns the 3D points that lie on the plane corresponding to the
        specified z-coordinate.

        :param z: The z-coordinate of the plane for which to retrieve the border points.
        :type z: float
        :return: A numpy array of shape (N, 3) containing the 3D coordinates of the border points at
            the specified z-coordinate.
        :rtype: np.ndarray[float]
        """
        return self.points[self.points[:, 2] == z]

    def has_points(self, z):
        """
        Check if the anatomical structure has points at the specified z-coordinate.

        This method determines whether there are any points defining the border of the
        anatomical structure on the plane corresponding to the given z-coordinate.

        :param z: The z-coordinate to check.
        :type z: float
        :return: True if the anatomical structure has points at the specified z-coordinate, otherwise False.
        :rtype: bool
        """
        return len(self.get_points(z)) != 0

    def _get_indices_and_contour(self, point):
        """
        Retrieve the indices and contour containing a specified point.

        This method identifies the contour in the anatomical structure that contains the given
        `point` and returns:
          - The index of the point within the identified contour.
          - The index of the contour within the contour sequence at the point's z-coordinate.
          - The contour itself.

        If the point is not found in any contour, the method returns `(None, None, None)`.

        :param point: The 3D point to search for in the contours.
        :type point: np.ndarray[float]
        :return: A tuple containing:
                 - The index of the point within the contour (int).
                 - The index of the contour within the contour sequence (int).
                 - The contour containing the point (np.ndarray[float]).
        :rtype: tuple[int | None, int | None, np.ndarray[float] | None]
        """
        z = point[2]
        for contour_index, contour in enumerate(self.get_contour_sequence(z)):
            index = _find_index(point, contour)
            if index is not None:
                return index, contour_index, contour
        return None, None, None

    def get_points_between(self, start_point, end_point, orientation=1):
        """
        Retrieve the points on the border of the anatomical structure between `start_point` (inclusive) and `end_point`
        (exclusive).

        This method returns the points from the contour corresponding to the z-coordinate of `start_point`, between the
        specified `start_point` (inclusive) and `end_point` (exclusive). The orientation determines the direction of the
        returned points: if `orientation >= 0`, the points are returned in a mathematically positive orientation; if
        `orientation < 0`, the points are returned in a mathematically negative orientation.

        :param start_point: The starting point (inclusive).
        :type start_point: np.ndarray[float]
        :param end_point: The ending point (exclusive).
        :type end_point: np.ndarray[float]
        :param orientation: The direction of the points. If `>= 0`, the points are returned in positive mathematical
            orientation, otherwise in negative orientation.
        :type orientation: int
        :return: The points between `start_point` (inclusive) and `end_point` (exclusive) in the specified orientation.
        :rtype: np.ndarray[float]
        """
        z = start_point[2]
        if z != end_point[2]:
            return np.empty((0, 3))

        start_index, start_contour_index, contour = self._get_indices_and_contour(start_point)
        end_index, end_contour_index, contour = self._get_indices_and_contour(end_point)

        if (start_index is None or start_contour_index is None or end_index is None or end_contour_index is None or
                start_contour_index != end_contour_index):
            # point could not be found or start_point and end_point belong to different contours
            return np.empty((0, 3))

        # Note: the y-axis is inverted.
        if orientation >= 0:
            points_between = _slice_wrap_around(contour, end_index, start_index + 1)[::-1]
        else:
            points_between = _slice_wrap_around(contour, start_index, end_index + 1)

        return points_between

    def get_contour_sequence(self, z):
        """
        Retrieve the sequence of polygonal chains that describe the border of the anatomical structure for a specified
        z-coordinate.

        This method returns the contours (polygonal chains) that describe the border of the anatomical structure at the
        specified z-coordinate. Each contour in the sequence is represented as a list of points forming a polygonal
        chain in the xy-plane at the given z-level.

        :param z: The z-coordinate for which the contour sequence is requested.
        :type z: float
        :return: A list of contours (polygonal chains) describing the structure at the specified z-coordinate, or None
            if no contours exist at that z-level.
        :rtype: list[np.ndarray[float]] | None
        """
        if not z in self._contour_sequences:
            return None
        return self._contour_sequences[z]

    def get_anterior_point(self, z):
        """
        Retrieve the anterior point of the anatomical structure for a specified z-coordinate in the global coordinate
        system.

        This method identifies the anterior point of the anatomical structure using the conventional DICOM coordinate
        system, where the anterior direction corresponds to the smallest y-coordinate value.

        :param z: The z-coordinate at which to find the anterior point.
        :type z: float
        :return: The anterior point in the global coordinate system at the specified z-coordinate.
        :rtype: np.ndarray[float]
        """
        return self.get_furthest_point_in_xy_direction([0, -1, z])

    def get_posterior_point(self, z):
        """
        Retrieve the posterior point of the anatomical structure for a specified z-coordinate in the global coordinate
        system.

        This method identifies the posterior point of the anatomical structure using the conventional DICOM coordinate
        system, where the posterior direction corresponds to the largest y-coordinate value.

        :param z: The z-coordinate at which to find the posterior point.
        :type z: float
        :return: The posterior point in the global coordinate system at the specified z-coordinate.
        :rtype: np.ndarray[float]
        """
        return self.get_furthest_point_in_xy_direction([0, 1, z])

    def get_rightmost_point(self, z):
        """
        Retrieve the rightmost point of the anatomical structure for a specified z-coordinate in the global coordinate
        system.

        :param z: The z-coordinate at which to find the rightmost point.
        :type z: float
        :return: The rightmost point in the global coordinate system at the specified z-coordinate.
        :rtype: np.ndarray[float]
        """
        return self.get_furthest_point_in_xy_direction([1, 0, z])

    def get_leftmost_point(self, z):
        """
        Retrieve the leftmost point of the anatomical structure for a specified z-coordinate in the global coordinate
        system.

        :param z: The z-coordinate at which to find the leftmost point.
        :type z: float
        :return: The leftmost point in the global coordinate system at the specified z-coordinate.
        :rtype: np.ndarray[float]
        """
        return self.get_furthest_point_in_xy_direction([-1, 0, z])

    def get_furthest_point_in_xy_direction(self, direction):
        """
        Retrieve the furthest point of the anatomical structure in a specified direction within the xy-plane.

        This method identifies the point in the anatomical structure that lies furthest in the specified direction in
        the xy-plane. The z-coordinate of the plane is determined by the z-value of the given direction vector. If
        multiple points meet this criterion, one of them will be returned (selection is not specified).

        :param direction: A 3D vector specifying the direction in the xy-plane to determine the furthest point.
        :type direction: np.ndarray[float]
        :return: The furthest point in the specified direction at the given z-coordinate.
        :rtype: np.ndarray[float]
        """
        z_value = direction[2]
        points = self.get_points(z_value)
        projections = np.dot(points, direction)
        return points[np.argmax(projections)]

    def get_furthest_point_in_xy_directions(self, direction_1, direction_2):
        """
        Retrieve the furthest point of the anatomical structure in two specified directions within the xy-plane.

        This method identifies the point in the anatomical structure that is furthest in the primary `direction_1` and,
        among those, furthest in the secondary `direction_2`. The z-coordinate of the plane is determined by the z-value
        of the given direction vectors. If the z-values of `direction_1` and `direction_2` differ, an empty array is
        returned. If multiple points meet the criteria, one of them will be returned (selection is not specified).

        :param direction_1: A 3D vector specifying the primary direction in the xy-plane to determine the furthest
            point.
        :type direction_1: np.ndarray[float]
        :param direction_2: A 3D vector specifying the secondary direction in the xy-plane for further refinement.
        :type direction_2: np.ndarray[float]
        :return: The furthest point based on the specified directions at the given z-coordinate, or an empty array if
            the z-values of the directions differ.
        :rtype: np.ndarray[float]
        """
        if direction_1[2] != direction_2[2]:
            return np.empty((0, 3))

        z_value = direction_1[2]
        points = self.get_points(z_value)
        projections_direction_1 = np.dot(points, direction_1)

        max_value_direction_1 = np.max(projections_direction_1)
        extreme_points_direction_1 = points[projections_direction_1 == max_value_direction_1]

        projections_direction_2 = np.dot(extreme_points_direction_1, direction_2)
        return extreme_points_direction_1[np.argmax(projections_direction_2)]

    def get_closest_pair_between_structures(self, anatomical_structure, z):
        """
        Retrieve the closest pair of points between this anatomical structure and another in the xy-plane.

        This method computes the pair of points—one from this anatomical structure and one from the specified anatomical
        structure—that are closest to each other in the xy-plane at a given z-coordinate.

        :param anatomical_structure: The other anatomical structure to compare against.
        :type anatomical_structure: AnatomicalStructure
        :param z: The z-coordinate of the xy-plane where the closest points are determined.
        :type z: float
        :return: A tuple containing the closest point on this anatomical structure and the closest point on the
            specified anatomical structure. The order of the points in the tuple corresponds to this structure and the
            other structure respectively.
        :rtype: Tuple[np.ndarray[float], np.ndarray[float]]
        """
        min_point_self, min_point_other, distance = closest_between_points(
            self.get_points(z), anatomical_structure.get_points(z))
        return min_point_self, min_point_other

    def get_closest_point_to_structure(self, anatomical_structure, z):
        """
        Retrieve the closest point in the xy-plane on this anatomical structure to another anatomical structure.

        :param anatomical_structure: The other anatomical structure to calculate the closest point to.
        :type anatomical_structure: AnatomicalStructure
        :param z: The z-coordinate of the xy-plane at which the closest point is determined.
        :type z: float
        :return: The point on this anatomical structure that is closest to the specified anatomical structure in the
                 xy-plane at the given z-coordinate.
        :rtype: np.ndarray[float]
        """
        min_point_self, min_point_other = self.get_closest_pair_between_structures(anatomical_structure, z)
        return min_point_self

    def get_closest_point_to_point(self, point, z):
        """
         Retrieve the closest point in the xy-plane on this anatomical structure to a specified point.

        :param point: The target point in the xy-plane for which the closest point on the anatomical structure is sought.
        :type point: np.ndarray[float]
        :param z: The z-coordinate of the xy-plane at which the closest point is determined.
        :type z: float
        :return: The point on this anatomical structure that is closest to the specified point in the xy-plane at
                 the given z-coordinate.
        :rtype: np.ndarray[float]
        """
        closest_point, distance = closest_to_point(self.get_points(z), point)
        return closest_point

    def get_first_intersection_with_line(self, start, direction):
        """
        Retrieve the intersection of this anatomical structure with a line.

        This function determines the edge on the border of the anatomical structure that intersects with the line
        that starts in `start` and has the specified direction in the xy-plane. It then returns the point of the edge
        that is closest to `start`.
        The xy-plane is determined by the z-value of `direction`.

        :param start: The starting point of the line.
        :type start: np.ndarray[float]
        :param direction: The direction of the line.
        :type direction: np.ndarray[float]
        :return: The intersection point of the line with this anatomical structure.
        :rtype: np.ndarray[float]
        """
        z = start[2]
        if z != direction[2] or not self.has_points(z):
            return None

        z = start[2]
        for contour in self.get_contour_sequence(z):
            structure_to_start = start - contour
            projection = np.dot(structure_to_start, get_orthogonal_in_xy(direction))

            if np.any(projection <= 0) and np.any(projection >= 0):
                # structure intersects with line
                intersecting_points = contour[(projection * np.roll(projection, 1) <= 0)
                                              | (projection * np.roll(projection, -1) <= 0)]
                intersecting_point_closest_to_start, _ = closest_to_point(intersecting_points, start)
                return intersecting_point_closest_to_start

        return np.empty((0, 3))

    def get_caudal_point(self):
        """
        Retrieve the caudal point of this anatomical structure.

        This function returns the caudal point of this anatomical structure, where the caudal point is interpreted as
        the point with minimal z-value.

        :return: The caudal point of this anatomical structure.
        :rtype: np.ndarray[float]
        """
        return self.points[np.argmin(self.points[:, 2])]

    def get_cranial_point(self):
        """
        Retrieve the cranial point of this anatomical structure.

        This function returns the cranial point of this anatomical structure, where the cranial point is interpreted as
        the point with maximal z-value.

        :return: The cranial point of this anatomical structure.
        :rtype: np.ndarray[float]
        """
        return self.points[np.argmax(self.points[:, 2])]

    def get_principal_component(self, z):
        """
        Retrieve the first principal component of this anatomical structure in the xy-plane.

        :param z: The z-coordinate of the xy-plane.
        :type z: np.ndarray[float]
        """
        pca = PCA(n_components=1)
        pca.fit(self.get_points(z))
        first_principal_component = pca.components_[0]
        # z-value is subject to numerical imprecision
        first_principal_component[2] = z
        return first_principal_component

    def get_tips(self, z):
        """
        Retrieve the tips of this anatomical structure in the xy-plane.

        The tips of this anatomical structure in the xy-plane are defined as the points furthest
        along the direction of the first principal component (positive and negative directions).
        If multiple points qualify as the lateral tip, one is arbitrarily chosen.

        The xy-plane is determined by the specified `z`-coordinate.

        :param z: The z-coordinate of the xy-plane.
        :type z: np.ndarray[float]
        :return: The tips of this anatomical structure in the xy-plane.
        :rtype: np.ndarray[float]
        """
        points = self.get_points(z)

        pca = PCA(n_components=1)
        pca.fit(points)
        first_principal_component = pca.components_[0]
        projections = np.dot(points - pca.mean_, first_principal_component)
        tip_1, tip_2 = points[np.argmin(projections)], points[np.argmax(projections)]

        return tip_1, tip_2

    def get_anterior_tip(self, z):
        """
        Retrieve the anterior tip of this anatomical structure in the xy-plane.

        The tips of this anatomical structure in the xy-plane are defined as the points furthest
        along the direction of the first principal component (positive and negative directions).
        This method returns the anterior tip, which is the tip with the smaller y-coordinate.
        If multiple points qualify as the anterior tip, one is arbitrarily chosen.

        The xy-plane is determined by the specified `z`-coordinate.

        :param z: The z-coordinate defining the xy-plane.
        :type z: np.ndarray[float]
        :return: The anterior tip of the anatomical structure in the xy-plane.
        :rtype: np.ndarray[float]
        """
        tip_1, tip_2 = self.get_tips(z)
        return tip_1 if tip_1[1] < tip_2[1] else tip_2

    def get_posterior_tip(self, z):
        """
        Retrieve the posterior tip of this anatomical structure in the xy-plane.

        The tips of this anatomical structure in the xy-plane are defined as the points furthest
        along the direction of the first principal component (positive and negative directions).
        This method returns the posterior tip, which is the tip with the larger y-coordinate.
        If multiple points qualify as the posterior tip, one is arbitrarily chosen.

        The xy-plane is determined by the specified `z`-coordinate.

        :param z: The z-coordinate defining the xy-plane.
        :type z: np.ndarray[float]
        :return: The posterior tip of the anatomical structure in the xy-plane.
        :rtype: np.ndarray[float]
        """
        tip_1, tip_2 = self.get_tips(z)
        return tip_1 if tip_1[1] > tip_2[1] else tip_2

    def get_left_tip(self, z):
        """
        Retrieve the left tip of this anatomical structure in the xy-plane.

        The tips of this anatomical structure in the xy-plane are defined as the points furthest along the direction of
        the first principal component (positive and negative directions).
        This function returns the tip with smaller x-value.
        If multiple points qualify as the lateral tip, one is arbitrarily chosen.
        The xy-plane is determined by the z-value of `direction`.

        :param z: The z-coordinate of the xy-plane.
        :type z: np.ndarray[float]
        :return: The left tip of this anatomical structure in the xy-plane.
        :rtype: np.ndarray[float]
        """
        tip_1, tip_2 = self.get_tips(z)
        return tip_1 if tip_1[0] < tip_2[0] else tip_2

    def get_right_tip(self, z):
        """
        Retrieve the right tip of this anatomical structure in the xy-plane.

        The tips of this anatomical structure in the xy-plane are defined as the points furthest along the direction of
        the first principal component (positive and negative directions).
        This function returns the tip with larger x-value.
        If multiple points qualify as the lateral tip, one is arbitrarily chosen.
        The xy-plane is determined by the z-value of `direction`.

        :param z: The z-coordinate of the xy-plane.
        :type z: np.ndarray[float]
        :return: The right tip of this anatomical structure in the xy-plane.
        :rtype: np.ndarray[float]
        """
        tip_1, tip_2 = self.get_tips(z)
        return tip_1 if tip_1[0] > tip_2[0] else tip_2

    def get_mean_point(self, z):
        """
        Retrieve the mean point of this anatomical structure in the xy-plane.

        :param z: The z-coordinate of the xy-plane.
        :type z: np.ndarray[float]
        :return: The mean point of this anatomical structure in the xy-plane.
        :rtype: np.ndarray[float]
        """
        return np.mean(self.get_points(z), axis=0)


class LeftAnatomicalStructure(AnatomicalStructure):
    """
    Represents anatomical structures located on the left side of the patient.

    This class extends the general functionality of `AnatomicalStructure` by adding methods
    specific to structures that are located on one side of the body.
    The methods in this class are meaningful only for structures positioned on the left side
    of the patient.

    In CT scans, the left side of the patient is represented on the right side of the scan.
    This class, therefore, models anatomical structures on the patient's left side, which
    correspond to the right side of the CT scan.

    The DICOM coordinate system convention is followed in this class:
    - The x-axis extends from the patient's right (-x) to left (+x).
    - The y-axis extends from the patient's anterior (front, +y) to posterior (back, -y).
    - The z-axis extends from inferior (towards feet, -z) to superior (towards head, +z).
    """
    @classmethod
    def from_dicom(cls, path, label):
        """
        Create a `LeftAnatomicalStructure` instance from a DICOM RT-struct file.

        This method initializes a new `LeftAnatomicalStructure` object based on the
        specified DICOM RT-struct file and the provided label.

        :param path: The file path to the DICOM RT-struct file.
        :type path: str
        :param label: The label identifying the anatomical structure within the DICOM RT-struct file.
        :type label: str
        :return: A `LeftAnatomicalStructure` instance.
        :rtype: LeftAnatomicalStructure
        """
        points, contour_sequence = cls._extract_points_contour_sequence_from_dicom(path, label)
        return LeftAnatomicalStructure(points, contour_sequence)

    def get_lateral_tip(self, z):
        """
        Retrieve the lateral tip of this anatomical structure in the xy-plane.

        The tips of this anatomical structure in the xy-plane are defined as the points furthest
        along the direction of the first principal component (positive and negative directions).
        This method returns the lateral tip, which is the tip with larger x-value.
        If multiple points qualify as the lateral tip, one is arbitrarily chosen.

        The xy-plane is determined by the specified `z`-coordinate.

        :param z: The z-coordinate defining the xy-plane.
        :type z: np.ndarray[float]
        :return: The lateral tip of the anatomical structure in the xy-plane.
        :rtype: np.ndarray[float]
        """
        return self.get_right_tip(z)

    def get_medial_tip(self, z):
        """
        Retrieve the medial tip of this anatomical structure in the xy-plane.

        The tips of this anatomical structure in the xy-plane are defined as the points furthest
        along the positive and negative directions of the first principal component.
        This method returns the medial tip, which is the tip with smaller x-value.
        If multiple points qualify as the medial tip, one is arbitrarily chosen.

        The xy-plane is determined by the specified `z`-coordinate.

        :param z: The z-coordinate defining the xy-plane.
        :type z: np.ndarray[float]
        :return: The medial tip of the anatomical structure in the xy-plane.
        :rtype: np.ndarray[float]
        """
        return self.get_left_tip(z)


class RightAnatomicalStructure(AnatomicalStructure):
    """
    Represents anatomical structures located on the right side of the patient.

    This class extends the general functionality of `AnatomicalStructure` by adding methods
    specific to structures that are located on one side of the body.
    The methods in this class are meaningful only for structures positioned on the right side
    of the patient.

    In CT scans, the right side of the patient is represented on the left side of the scan.
    This class, therefore, models anatomical structures on the patient's right side, which
    correspond to the left side of the CT scan.

    The DICOM coordinate system convention is followed in this class:
    - The x-axis extends from the patient's right (-x) to left (+x).
    - The y-axis extends from the patient's anterior (front, +y) to posterior (back, -y).
    - The z-axis extends from inferior (towards feet, -z) to superior (towards head, +z).
    """
    @classmethod
    def from_dicom(cls, path, label):
        """
        Create a `RightAnatomicalStructure` instance from a DICOM RT-struct file.

        This method initializes a new `RightAnatomicalStructure` object based on the
        specified DICOM RT-struct file and the provided label.

        :param path: The file path to the DICOM RT-struct file.
        :type path: str
        :param label: The label identifying the anatomical structure within the DICOM RT-struct file.
        :type label: str
        :return: A `RightAnatomicalStructure` instance.
        :rtype: RightAnatomicalStructure
        """
        points, contour_sequence = cls._extract_points_contour_sequence_from_dicom(path, label)
        return RightAnatomicalStructure(points, contour_sequence)

    def get_lateral_tip(self, z):
        """
        Retrieve the lateral tip of this anatomical structure in the xy-plane.

        The tips of this anatomical structure in the xy-plane are defined as the points furthest
        along the direction of the first principal component (positive and negative directions).
        This method returns the lateral tip, which is the tip with smaller x-value.
        If multiple points qualify as the lateral tip, one is arbitrarily chosen.

        The xy-plane is determined by the specified `z`-coordinate.

        :param z: The z-coordinate defining the xy-plane.
        :type z: np.ndarray[float]
        :return: The lateral tip of the anatomical structure in the xy-plane.
        :rtype: np.ndarray[float]
        """
        return self.get_left_tip(z)

    def get_medial_tip(self, z):
        """
        Retrieve the medial tip of this anatomical structure in the xy-plane.

        The tips of this anatomical structure in the xy-plane are defined as the points furthest
        along the positive and negative directions of the first principal component.
        This method returns the medial tip, which is the tip with larger x-value.
        If multiple points qualify as the medial tip, one is arbitrarily chosen.

        The xy-plane is determined by the specified `z`-coordinate.

        :param z: The z-coordinate defining the xy-plane.
        :type z: np.ndarray[float]
        :return: The medial tip of the anatomical structure in the xy-plane.
        :rtype: np.ndarray[float]
        """
        return self.get_right_tip(z)


class NeckNodeLevel(ABC):
    """
    Base class for a neck node level.

    This class provides methods to generale an initial neck node level, process plot and the neck node level.
    """

    def __init__(self, rt_path, oar, caudal_boundary, cranial_boundary, slice_thickness, relevant_structures):
        """
        Initialize an `NeckNodeLevel` object.

        For each axial slice, the neck node level is defined by its boundary , which is a closed polygonal chain.
        Each Neck Node Level is associated with an CT-scan in DICOM format.

        :param rt_path: The path to the RT-struct of the DICOM file.
        :type rt_path: str
        :param oar: The labels in the DICOM file of the organs at risk. The neck node level should not cut these organs.
        :type oar: list[str]
        :param caudal_boundary: The z-value of the caudal boundary (exclusively) of this neck node level in mm.
        :type caudal_boundary: float
        :param cranial_boundary: The z-value of the cranial boundary (exclusively) of this neck node level in mm.
        :type cranial_boundary: float
        :param slice_thickness: The distance between each axial slice.
        :type slice_thickness: float
        :param relevant_structures: The labels in the DICOM file of the organs that must exist in every slice
            between `caudal_boundary` and `cranial_boundary` (both exclusively) to generate the neck node level.
        :type relevant_structures: list[str]
        :return: The initialized `NeckNodeLevel` object.
        :rtype: NeckNodeLevel
        """
        self._rt_path = rt_path
        self._oar = oar
        self._caudal_boundary = caudal_boundary
        self._cranial_boundary = cranial_boundary
        self._slice_thickness = slice_thickness
        self._relevant_structures = relevant_structures
        self._contour = None

    def _check_relevant_structures_exist(self):
        """
        Check if all structures in `_relevant_structures` exist on every slice between `_caudal_boundary`
        and `_cranial_boundary` (both exclusively).

        :return: `True` iff all structures in `_relevant_structures` exist on every slice between `_caudal_boundary`
            and `_cranial_boundary` (both exclusively), otherwise `False`.
        :rtype: bool
        """
        for z in np.arange(self._caudal_boundary + self._slice_thickness, self._cranial_boundary,
                           self._slice_thickness):
            for structure in self._relevant_structures:
                if not np.any(structure.points[:, 2] == z):
                    return False
        return True

    @property
    def contour(self):
        """
        Retrieve the boundary of this neck node level.

        :return: The boundary of the neck node level.
        :rtype: np.ndarray[float]
        """
        if self._contour is None:
            self._initialize_contour()
        return self._contour

    def remove_self_intersections(self):
        """
        Remove intersections of the boundary of this neck node level with itself.

        :return: This `NeckNodeLevel` instance.
        :rtype: NeckNodeLevel
        """
        if self.contour.size == 0:
            return self

        self._contour = clip_contour(self.contour)
        return self

    def interpolate_in_z(self, point_distance=2):
        """
        Smooth the neck node level in the z-direction by averaging boundaries across slices.

        This method reduces variation in the boundary contours between axial slices by replacing the
        boundary of every second slice with an averaged boundary derived from its neighboring slices.
        New points are placed along the boundary at the specified distance.

        :param point_distance: The distance (in mm) between interpolated points along the boundary.
        :type point_distance: float
        :return: The current `NeckNodeLevel` instance with smoothed boundaries.
        :rtype: NeckNodeLevel
        """
        if self.contour.size == 0:
            return self

        self._contour = interpolate_contour_in_z(self.contour, self._rt_path, point_distance)
        return self

    def interpolate_in_xy(self, corners=None, radius=2, num_interpolation_points=10):
        """
        Smooth the neck node level boundary in each xy-plane.

        This method refines the boundary of the neck node level for each 2D axial slice by smoothing specific
        regions around specified points (`corners`). For each corner point, a circle with the given radius
        is drawn. Boundary points inside the circle are used as control points to define a curve in Bézier form.
        These control points are replaced with `num_interpolation_points` equally spaced points along the curve
        in Bézier form, creating a smoother boundary.

        :param corners: Points on the boundary to be smoothed. Each point must belong to the boundary.
        :type corners: np.ndarray[float]
        :param radius: The radius (in mm) of the circle used to select points for smoothing.
        :type radius: float
        :param num_interpolation_points: The number of equally spaced points used to interpolate the curve in Bézier form.
        :type num_interpolation_points: int
        :return: The current `NeckNodeLevel` instance with smoothed boundaries.
        :rtype: NeckNodeLevel
        """
        if self.contour.size == 0:
            return self

        self._contour = interpolate_contour_in_xy(self.contour, corners, radius, num_interpolation_points)
        return self

    def extract_structure_endpoints(self):
        """
        Extract endpoints of anatomical structures in the neck node level for each axial slice.

        The neck node level is represented as a closed polygonal chain in each axial slice. This method identifies
        the first and last points on the boundary of the neck node level that lie on each anatomical structure
        within the slice.

        :return: An array containing the endpoints of the anatomical structures.
        :rtype: np.ndarray[float]
        """
        if self.contour.size == 0:
            return np.empty((0, 3))

        return extract_structure_endpoints(self.contour, self._rt_path)

    def clip_corners(self, corners=None, radius=2, angle=np.pi / 2):
        """
        Clip the corners of a 3D contour that have an angle smaller than specified.

        The neck node level is represented as a closed polygonal chain in each axial slice. For each point in `corners`
        this method checks if the angle formed ar that point is smaller than the specified angle. If so, it removes the
        portion of the polygonal chain within a circle of the given radius centered ar the corner, and replaces it with
        the circle's intersection points with the contour.

        :param corners: The points to consider for clipping, as a 2D array of points.
        :type corners: np.ndarray
        :param radius: The radius of the circle used for clipping.
        :type radius: float
        :param angle: The threshold angle in radians. Points in `corners` with angles smaller than this will be clipped.
        :type angle: float
        :return: The current `NeckNodeLevel` instance with smoothed boundaries.
        :rtype: NeckNodeLevel
        """
        if self.contour.size == 0:
            return self

        corners = self.extract_structure_endpoints() if corners is None else corners
        if corners.size == 0:
            self.extract_structure_endpoints()
            return self

        self._contour = clip_corners(self.contour, corners, radius, angle)
        return self

    def remove_intersections(self):
        """
        Remove intersections of the boundary of this neck node level with itself and anatomical structures.

        :return: This `NeckNodeLevel` instance.
        :rtype: NeckNodeLevel
        """
        if self.contour.size == 0:
            return self

        self.remove_self_intersections()
        new_contour = []
        for z in np.unique(self.contour[:, 2]):
            contour_slice = self.contour[self.contour[:, 2] == z][:, :2]
            contour_poly = Polygon(contour_slice)

            for structure in self._oar:
                if structure.has_points(z):
                    for partial_structure in map(AnatomicalStructure, structure.get_contour_sequence(z)):
                        structure_poly = Polygon(partial_structure.get_points(z)[:, :2])
                        if structure.orientation == 1:
                            for geom in split(contour_poly, structure_poly.boundary).geoms:
                                if not geom.difference(structure_poly, 0.1).area < 1e-10:
                                    contour_poly = geom
                        else:
                            contour_poly = contour_poly.union(structure_poly)
            contour_poly = orient(contour_poly, 1)
            new_contour_slice = np.empty((len(contour_poly.exterior.coords), 3))
            new_contour_slice[:, :2] = np.array(contour_poly.exterior.coords)
            new_contour_slice[:, 2] = z
            new_contour.extend(new_contour_slice)
        self._contour = np.array(new_contour)
        return self

    def plot(self, points=None, plot_anatomical_structures=True, anatomical_structures=None, fixed_view=False,
             marker_anatomical_structures='o', marker_contour='x', marker_points='v', slices=None, full_screen=True):
        """
        Plot this neck node level.

        This method plots this neck node level for each 2D axial slice.

        :param points: Points that shall be plotted in addition to this neck node level.
        :type points: np.ndarray[float]
        :param plot_anatomical_structures: Whether to plot anatomical structures.
        :type plot_anatomical_structures: bool
        :param anatomical_structures: The labels in the DICOM file of the anatomical structures that are to be plotted.
            If not set, the thyroid gland, trachea, left common carotid artery, left internal jugular vein, left
            anterior scalene muscle, left middle scalene muscle, left sternocleidomastoid muscle and left sternothyroid
            muscle are plotted.
        :type anatomical_structures: dict[str, str]
        :param fixed_view: Whether the window of the plot shall be fixed between slices.
        :type fixed_view: bool
        :param marker_anatomical_structures: The marker used to plot the anatomical structures.
        :type marker_anatomical_structures: str
        :param marker_contour: The marker used to plot this neck node level.
        :type marker_contour: str
        :param marker_points: The marker used to plot `points`.
        :type marker_points: str
        :param slices: The z-value of the slices that are to be plotted. If not set, all slices of this neck node level
            will be plotted.
        :type slices: np.ndarray[float]
        :param full_screen: Whether to show the plot in fullstreen.
        :type full_screen: bool
        :return: This `NeckNodeLevel` instance.
        :rtype: NeckNodeLevel
        """
        plot_contour(self.contour,
                     path=self._rt_path,
                     cranial_boundary=self._cranial_boundary,
                     caudal_boundary=self._caudal_boundary,
                     slice_thickness=self._slice_thickness,
                     points=points,
                     plot_anatomical_structures=plot_anatomical_structures,
                     anatomical_structures=anatomical_structures,
                     fixed_view=fixed_view,
                     marker_anatomical_structures=marker_anatomical_structures,
                     marker_contour=marker_contour,
                     marker_points=marker_points,
                     slices=slices,
                     full_screen=full_screen)
        return self

    def save(self, path, contour_name):
        """
        Add this `NeckNodeLevel` instance to a DICOM file at `path`.

        :param path: The path to a folder containing the RT-struct and all CT-images belonging to the CT scan.
        :type path: str
        :param contour_name: The label used to add this neck node level to the DICOM file.
        :type contour_name: str
        :return: This `NeckNodeLevel` instance.
        :rtype: NeckNodeLevel
        """
        add_contour(path, self.contour, contour_name)
        return self

    @abstractmethod
    def _initialize_contour(self):
        """
        Initialize the `_contour` attribute.
        """
        pass


class NeckNodeLevel4aLeft(NeckNodeLevel):
    """
    A class representing the left neck node level IVa, according to the Guidelines of Grégoire et al. (2013).

    The neck node level IVa can be divided into two parts: the one located on the left and the one located on the right
    side of the patient. This class represents the first part. The neck node level defined with respect to bordering
    anatomical structures as described by Grégoire et al.
    """
    def __init__(self, path):
        """
        Initialize an `NeckNodeLevel4aLeft` object.

        For each axial slice, the neck node level is defined by its boundary , which is a closed polygonal chain.
        Each Neck Node Level is associated with an CT-scan in DICOM format.

        :param path: The path to a folder containing the CT-scan in DICOM format that is associated with this neck node
            level.
        :type path: str
        """
        self._rt_path, self._ct_path = extract_rt_ct_paths(path)
        self._cricoid = AnatomicalStructure.from_dicom(self._rt_path, 'KNORPEL_CRICOID')
        self._sternum = AnatomicalStructure.from_dicom(self._rt_path, 'STERNUM_MANUBRIUM')
        self._sternocleido = LeftAnatomicalStructure.from_dicom(self._rt_path, 'M_STERNOCLEIDOMASTOID_LINKS')
        self._scalenus_med = LeftAnatomicalStructure.from_dicom(self._rt_path, 'M_SCALENUS_MEDIUS_LINKS')
        self._scalenus_ant = LeftAnatomicalStructure.from_dicom(self._rt_path, 'M_SCALENUS_ANTERIOR_LINKS')
        self._carotid = LeftAnatomicalStructure.from_dicom(self._rt_path, 'ARTERY_COMMONCAROTID_LINKS')
        self._gland_thyroid = AnatomicalStructure.from_dicom(self._rt_path, 'GLAND_THYROID')
        self._sterno_thyroid = LeftAnatomicalStructure.from_dicom(self._rt_path, 'M_STERNO_THYROID_LINKS')
        self._trachea = AnatomicalStructure.from_dicom(self._rt_path, 'TRACHEA')

        self.oar = [self._cricoid, self._sternum, self._sternocleido, self._scalenus_med,
                    self._scalenus_ant, self._carotid, self._gland_thyroid, self._trachea]
        # structures that must exist in every slice of the contour
        self._relevant_structures = [self._sternocleido, self._scalenus_ant, self._carotid]

        self._cricoid.orientation = 1
        self._sternum.orientation = 1
        self._sternocleido.orientation = 1
        self._scalenus_med.orientation = 1
        self._scalenus_ant.orientation = 1
        self._carotid.orientation = -1
        self._gland_thyroid.orientation = 1
        self._sterno_thyroid.orientation = 1
        self._trachea.orientation = 1

        self._slice_thickness = get_slice_thickness(self._ct_path)
        self._cranial_boundary = self._cricoid.get_caudal_point()[2]
        self._caudal_boundary = (self._sternum.get_cranial_point()[2]
                                 + np.floor(20 / self._slice_thickness) * self._slice_thickness)
        super().__init__(self._rt_path, self.oar, self._caudal_boundary, self._cranial_boundary,
                         self._slice_thickness, self._relevant_structures)


    def _initialize_contour(self):
        """
        Initialize the `_contour` attribute.
        """
        if not self._check_relevant_structures_exist():
            self._contour = np.empty((0, 3))
        else:
            contour = []
            for z in np.arange(self._caudal_boundary + self._slice_thickness, self._cranial_boundary,
                               self._slice_thickness):
                contour_slice = []
                contour_slice.extend(self._get_sternocleido_points(z))
                contour_slice.extend(self._get_scalenus_med_points(z))
                contour_slice.extend(self._get_scalenus_ant_points(z))
                contour_slice.extend(self._get_carotid_points(z))
                contour_slice.extend(self._get_gland_thyroid_points(z))
                # the following line can be uncommented to exclude the sterno thyroid muscle in the contour
                # contour_slice.extend(self._get_sterno_thyroid_points(z))
                contour_slice = _close_contour(contour_slice)
                contour.extend(contour_slice)
            self._contour = np.array(contour)

    def _get_sternocleido_points(self, z):
        """
        Return the points of the neck node level in the specified slice that lie on the sternocleidomastoideus
        muscle.

        :param z: The z-value of the slice.
        :type z: float
        :return: The points of the neck node level that lie on the sternocleidomastoideus muscle.
        :rtype: np.ndarray
        """
        if not self._sternocleido.has_points(z):
            return np.empty((0, 3))

        points = []
        for partial_sternocleido in map(LeftAnatomicalStructure, self._sternocleido.get_contour_sequence(z)):
            start_point = partial_sternocleido.get_medial_tip(z)
            end_point = partial_sternocleido.get_lateral_tip(z)
            points.extend(partial_sternocleido.get_points_between(start_point, end_point))
        return np.array(points)

    def _get_scalenus_med_points(self, z):
        """
        Return the points of the neck node level in the specified slice that lie on the scalene medius muscle.

        :param z: The z-value of the slice.
        :type z: float
        :return: The points of the neck node level that lie on the scalene medius muscle.
        :rtype: np.ndarray
        """
        if not self._scalenus_med.has_points(z):
            return np.empty((0, 3))

        lateral_sternocleido = LeftAnatomicalStructure(self._sternocleido.get_contour_sequence(z)[-1])
        start_point = self._scalenus_med.get_first_intersection_with_line(
            lateral_sternocleido.get_lateral_tip(z),
            get_orthogonal_in_xy(lateral_sternocleido.get_principal_component(z)))

        if len(start_point) == 0:
            # Orthogonal to sternocleidomastoideus does not intersect with scalenus medius
            return np.empty((0, 3))

        end_point = self._scalenus_med.get_closest_point_to_point(self._scalenus_ant.get_lateral_tip(z), z)
        if start_point[0] < end_point[0]:
            start_point, end_point = end_point, start_point
            return start_point[np.newaxis, :]
        return self._scalenus_med.get_points_between(start_point, end_point)

    def _get_scalenus_ant_points(self, z):
        """
        Return the points of the neck node level in the specified slice that lie on the scalene anterior muscle.

        :param z: The z-value of the slice.
        :type z: float
        :return: The points of the neck node level that lie on the scalene anterior muscle.
        :rtype: np.ndarray
        """
        if not self._scalenus_ant.has_points(z):
            return np.empty((0, 3))

        start_point = self._scalenus_ant.get_lateral_tip(z)
        end_point = self._scalenus_ant.get_medial_tip(z)
        return self._scalenus_ant.get_points_between(start_point, end_point)

    def _get_carotid_points(self, z):
        """

        Return the points of the neck node level in the specified slice that lie on the common carotid artery.

        :param z: The z-value of the slice.
        :type z: float
        :return: The points of the neck node level that lie on the common carotid artery.
        :rtype: np.ndarray
        """
        if not self._carotid.has_points(z):
            return np.empty((0, 3))

        start_point = self._carotid.get_closest_point_to_point(self._scalenus_ant.get_medial_tip(z), z)
        if (self._gland_thyroid.has_points(z) and self._trachea.has_points(z)
                and self._gland_thyroid.get_rightmost_point(z)[0] > self._trachea.get_rightmost_point(z)[0]):
            end_point = self._carotid.get_closest_point_to_structure(self._gland_thyroid, z)
        else:
            end_point = self._carotid.get_closest_point_to_point(self._sternocleido.get_anterior_tip(z), z)

        return self._carotid.get_points_between(start_point, end_point, -1)

    def _get_gland_thyroid_points(self, z):
        """
        Return the points of the neck node level in the specified slice that lie on the thyroid gland.

        :param z: The z-value of the slice.
        :type z: float
        :return: The points of the neck node level that lie on the thyroid gland.
        :rtype: np.ndarray
        """
        if not self._gland_thyroid.has_points(z):
            return np.empty((0, 3))

        end_point = self._gland_thyroid.get_rightmost_point(z)

        if self._trachea.has_points(z) and end_point[0] < self._trachea.get_rightmost_point(z)[0]:
            # Thyroid gland on right side of trachea does not exist.
            return np.empty((0, 3))

        start_point = self._gland_thyroid.get_closest_point_to_structure(self._carotid, z)
        return self._gland_thyroid.get_points_between(start_point, end_point)

    def _get_sterno_thyroid_points(self, z):
        """
        Return the points of the neck node level in the specified slice that lie on the sternothyroid muscle.

        :param z: The z-value of the slice.
        :type z: float
        :return: The points of the neck node level that lie on the sternothyroid muscle.
        :rtype: np.ndarray
        """
        if not self._sterno_thyroid.has_points(z):
            return np.empty((0, 3))

        start_point = self._sterno_thyroid.get_lateral_tip(z)

        medial_sternocleido = LeftAnatomicalStructure(self._sternocleido.get_contour_sequence(z)[0])
        end_point = self._sterno_thyroid.get_first_intersection_with_line(
            medial_sternocleido.get_medial_tip(z),
            get_orthogonal_in_xy(medial_sternocleido.get_principal_component(z)))
        if end_point.size == 0:
            # orthogonal to sternocleido does not intersect with sterno thyroid
            return np.empty((0, 3))

        return self._sterno_thyroid.get_points_between(start_point, end_point)


def _add_neck_node_level_4a_left_to_dicom(input_path, output_path, name='Level_IVa_left'):
    neck_node_level = NeckNodeLevel4aLeft(input_path)
    neck_node_level.remove_self_intersections()
    neck_node_level.interpolate_in_z()
    neck_node_level.clip_corners(neck_node_level.extract_structure_endpoints())
    neck_node_level.interpolate_in_xy()
    neck_node_level.remove_intersections()
    # The following line can be uncommented to plot the neck node level before saving it.
    neck_node_level.plot()
    neck_node_level.save(output_path, name)


def _find_paths(root):
    paths = []
    for dir_path, _, filenames in os.walk(root):
        rt_path, ct_path = None, None
        for filename in filenames:
            if filename.startswith('RS'):
                rt_path = os.path.join(dir_path, filename)
            elif filename.startswith('CT'):
                ct_path = os.path.join(dir_path, filename)
        if rt_path is not None and ct_path is not None:
            paths.append((dir_path, rt_path, ct_path))
    return paths


def add_neck_node_level_4a(root, name='Level_IVa_left'):
    """ root: path to a folder with folders. Each folder containing one rt struct and all ct images. """
    for dir_path, rt_path, ct_path in _find_paths(root):
        _add_neck_node_level_4a_left_to_dicom(rt_path, ct_path, dir_path, name)


if __name__ == '__main__':
    # example usage
    _rt_path = './HN_P001_pre/RS1.2.752.243.1.1.20230309171637404.1400.85608.dcm'
    _ct_path = './HN_P001_pre/CT1.3.6.1.4.1.14519.5.2.1.2193.7172.982935462018887556067134421255.dcm'
    _output_path = './HN_P001_pre'
    _add_neck_node_level_4a_left_to_dicom(_rt_path, _ct_path, _output_path)
