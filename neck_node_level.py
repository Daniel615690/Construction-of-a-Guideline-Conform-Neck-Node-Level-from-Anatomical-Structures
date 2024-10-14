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
from add_contour_to_dicom import add_contour


class AnatomicalStructure:
    def __init__(self, points, contour_sequences=None):
        self._points = np.empty((0, 3))
        self._contour_sequences = {}

        if contour_sequences is None:
            points_oriented = []
            for z in np.unique(points[:, 2]):
                points_in_slice = points[points[:, 2] == z]
                points_in_slice_oriented = _orient_positive(points_in_slice)

                points_oriented.extend(points_in_slice_oriented)
                self._contour_sequences[z] = [points_in_slice_oriented]
            self._points = np.array(points_oriented)
        else:
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
        points, contour_sequences = cls._extract_points_contour_sequence_from_dicom(path, label)
        return AnatomicalStructure(points, contour_sequences)

    @property
    def points(self):
        return self._points

    def get_points(self, z):
        return self.points[self.points[:, 2] == z]

    def has_points(self, z):
        return len(self.get_points(z)) != 0

    def _get_indices_and_contour(self, point):
        z = point[2]
        for contour_index, contour in enumerate(self.get_contour_sequence(z)):
            index = _find_index(point, contour)
            if index is not None:
                return index, contour_index, contour
        return None, None, None

    def get_points_between(self, start_point, end_point, orientation=1):
        z = start_point[2]
        if z != end_point[2]:
            return np.empty((0, 3))

        start_index, start_contour_index, contour = self._get_indices_and_contour(start_point)
        end_index, end_contour_index, contour = self._get_indices_and_contour(end_point)

        if (start_index is None or start_contour_index is None or end_index is None or end_contour_index is None or
                start_contour_index != end_contour_index):
            return np.empty((0, 3))

        # note that y-axis is inverted
        if orientation >= 0:
            points_between = _slice_wrap_around(contour, end_index, start_index + 1)[::-1]
        else:
            points_between = _slice_wrap_around(contour, start_index, end_index + 1)

        return points_between

    def get_contour_sequence(self, z):
        if not z in self._contour_sequences:
            return None
        return self._contour_sequences[z]

    def get_anterior_point(self, z):
        # Returns the point with the SMALLEST y value, as y-axis is from front to back of patient.
        return self.get_furthest_point_in_xy_direction([0, -1, z])

    def get_posterior_point(self, z):
        # Returns the point with the BIGGEST y value, as y-axis is from front to back of patient.
        return self.get_furthest_point_in_xy_direction([0, 1, z])

    def get_rightmost_point(self, z):
        # In the coordinate system if the DICOM file, x-axis goes to left side of patient
        return self.get_furthest_point_in_xy_direction([1, 0, z])

    def get_leftmost_point(self, z):
        # In the coordinate system if the DICOM file, x-axis goes to left side of patient
        return self.get_furthest_point_in_xy_direction([-1, 0, z])

    def get_furthest_point_in_xy_direction(self, direction):
        z_value = direction[2]
        points = self.get_points(z_value)
        projections = np.dot(points, direction)
        return points[np.argmax(projections)]

    def get_furthest_point_in_xy_directions(self, direction_1, direction_2):
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
        min_point_self, min_point_other, distance = closest_between_points(
            self.get_points(z), anatomical_structure.get_points(z))
        return min_point_self, min_point_other

    def get_closest_point_to_structure(self, anatomical_structure, z):
        min_point_self, min_point_other = self.get_closest_pair_between_structures(anatomical_structure, z)
        return min_point_self

    def get_closest_point_to_point(self, point, z):
        closest_point, distance = closest_to_point(self.get_points(z), point)
        return closest_point

    def get_first_intersection_with_line(self, start, direction):
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
        # Returns the point with the smallest z-value, as z-axis goes from foot to head of patient.
        return self.points[np.argmin(self.points[:, 2])]

    def get_cranial_point(self):
        # Returns the point with the biggest z-value, as z-axis goes from foot to head of patient.
        return self.points[np.argmax(self.points[:, 2])]

    def get_principal_component(self, z):
        pca = PCA(n_components=1)
        pca.fit(self.get_points(z))
        first_principal_component = pca.components_[0]
        first_principal_component[2] = z
        return first_principal_component

    def get_tips(self, z):
        points = self.get_points(z)

        pca = PCA(n_components=1)
        pca.fit(points)
        first_principal_component = pca.components_[0]
        projections = np.dot(points - pca.mean_, first_principal_component)
        tip_1, tip_2 = points[np.argmin(projections)], points[np.argmax(projections)]

        return tip_1, tip_2

    def get_anterior_tip(self, z):
        # Returns tip sith SMALLER y value, as y-axis is from front to back of patient.
        tip_1, tip_2 = self.get_tips(z)
        return tip_1 if tip_1[1] < tip_2[1] else tip_2

    def get_posterior_tip(self, z):
        # Returns tip sith BIGGER y value, as y-axis is from front to back of patient.
        tip_1, tip_2 = self.get_tips(z)
        return tip_1 if tip_1[1] > tip_2[1] else tip_2

    def get_left_tip(self, z):
        # Returns tip with smaller x value, x-axis goes from right side to left side of patient
        tip_1, tip_2 = self.get_tips(z)
        return tip_1 if tip_1[0] < tip_2[0] else tip_2

    def get_right_tip(self, z):
        # Returns tip with bigger x value, x-axis goes from right side to left side of patient
        tip_1, tip_2 = self.get_tips(z)
        return tip_1 if tip_1[0] > tip_2[0] else tip_2

    def get_mean_point(self, z):
        return np.mean(self.get_points(z), axis=0)


class LeftAnatomicalStructure(AnatomicalStructure):
    @classmethod
    def from_dicom(cls, path, label):
        points, contour_sequence = cls._extract_points_contour_sequence_from_dicom(path, label)
        return LeftAnatomicalStructure(points, contour_sequence)

    # the actual left side of the patient, so the right side in the CT image
    def get_lateral_tip(self, z):
        # Returns tip with BIGGER x-value, as x-axis goes from right to left side of patient
        return self.get_right_tip(z)

    def get_medial_tip(self, z):
        # Returns tip with SMALLER x-value, as x-axis goes from right to left side of patient
        return self.get_left_tip(z)


class RightAnatomicalStructure(AnatomicalStructure):
    @classmethod
    def from_dicom(cls, path, label):
        points, contour_sequence = cls._extract_points_contour_sequence_from_dicom(path, label)
        return RightAnatomicalStructure(points, contour_sequence)

    # the actual right side of the patient, so the left side in the CT image
    def get_lateral_tip(self, z):
        # Returns tip with SMALLER x-value, as x-axis goes from right to left side of patient
        return self.get_left_tip(z)

    def get_medial_tip(self, z):
        # Returns tip with BIGGER x-value, as x-axis goes from right to left side of patient
        return self.get_right_tip(z)


class NeckNodeLevel(ABC):
    def __init__(self, path, oar, caudal_boundary, cranial_boundary, slice_thickness, relevant_structures):
        self._path = path
        self._oar = oar
        self._caudal_boundary = caudal_boundary
        self._cranial_boundary = cranial_boundary
        self._slice_thickness = slice_thickness
        self._relevant_structures = relevant_structures
        self._contour = None

    def _check_relevant_structures_exist(self):
        for z in np.arange(self._caudal_boundary + self._slice_thickness, self._cranial_boundary,
                           self._slice_thickness):
            for structure in self._relevant_structures:
                if not np.any(structure.points[:, 2] == z):
                    return False
        return True

    @property
    def contour(self):
        if self._contour is None:
            self._initialize_contour()
        return self._contour

    def remove_self_intersections(self):
        if self.contour.size == 0:
            return self

        self._contour = clip_contour(self.contour)
        return self

    def interpolate_in_z(self, point_distance=2):
        if self.contour.size == 0:
            return self

        self._contour = interpolate_contour_in_z(self.contour, self._path, point_distance)
        return self

    def interpolate_in_xy(self, corners=None, radius=2, num_interpolation_points=10):
        if self.contour.size == 0:
            return self

        self._contour = interpolate_contour_in_xy(self.contour, corners, radius, num_interpolation_points)
        return self

    def extract_structure_endpoints(self):
        if self.contour.size == 0:
            return np.empty((0, 3))

        return extract_structure_endpoints(self.contour, self._path)

    def clip_corners(self, corners=None, radius=2, angle=np.pi / 2):
        if self.contour.size == 0:
            return self

        corners = self.extract_structure_endpoints() if corners is None else corners
        if corners.size == 0:
            self.extract_structure_endpoints()
            return self

        self._contour = clip_corners(self.contour, corners, radius, angle)
        return self

    def remove_intersections(self):
        """ Removes self-intersections and intersections with the anatomical structures self.spared_structures. """
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
        plot_contour(self.contour,
                     path=self._path,
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
        add_contour(path, self.contour, contour_name)
        return self

    @abstractmethod
    def _initialize_contour(self):
        pass


class NeckNodeLevel4aLeft(NeckNodeLevel):
    def __init__(self, rt_path, ct_path):
        self._path = rt_path
        self._ct_path = ct_path
        self._cricoid = AnatomicalStructure.from_dicom(rt_path, 'KNORPEL_CRICOID')
        self._sternum = AnatomicalStructure.from_dicom(rt_path, 'STERNUM_MANUBRIUM')
        self._sternocleido = LeftAnatomicalStructure.from_dicom(rt_path, 'M_STERNOCLEIDOMASTOID_LINKS')
        self._scalenus_med = LeftAnatomicalStructure.from_dicom(rt_path, 'M_SCALENUS_MEDIUS_LINKS')
        self._scalenus_ant = LeftAnatomicalStructure.from_dicom(rt_path, 'M_SCALENUS_ANTERIOR_LINKS')
        self._carotid = LeftAnatomicalStructure.from_dicom(rt_path, 'ARTERY_COMMONCAROTID_LINKS')
        self._gland_thyroid = AnatomicalStructure.from_dicom(rt_path, 'GLAND_THYROID')
        self._sterno_thyroid = LeftAnatomicalStructure.from_dicom(rt_path, 'M_STERNO_THYROID_LINKS')
        self._trachea = AnatomicalStructure.from_dicom(rt_path, 'TRACHEA')

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

        self._slice_thickness = self._get_slice_thickness(ct_path)
        self._cranial_boundary = self._cricoid.get_caudal_point()[2]
        self._caudal_boundary = (self._sternum.get_cranial_point()[2]
                                 + np.floor(20 / self._slice_thickness) * self._slice_thickness)
        super().__init__(self._path, self.oar, self._caudal_boundary, self._cranial_boundary,
                         self._slice_thickness, self._relevant_structures)

    def _get_slice_thickness(self, ct_path):
        ds = dcmread(ct_path)
        return float(ds.SliceThickness)

    def _initialize_contour(self):
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
        if not self._sternocleido.has_points(z):
            return np.empty((0, 3))

        points = []
        for partial_sternocleido in map(LeftAnatomicalStructure, self._sternocleido.get_contour_sequence(z)):
            start_point = partial_sternocleido.get_medial_tip(z)
            end_point = partial_sternocleido.get_lateral_tip(z)
            points.extend(partial_sternocleido.get_points_between(start_point, end_point))
        return np.array(points)

    def _get_scalenus_med_points(self, z):
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
        if not self._scalenus_ant.has_points(z):
            return np.empty((0, 3))

        start_point = self._scalenus_ant.get_lateral_tip(z)
        end_point = self._scalenus_ant.get_medial_tip(z)
        return self._scalenus_ant.get_points_between(start_point, end_point)

    def _get_carotid_points(self, z):
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
        if not self._gland_thyroid.has_points(z):
            return np.empty((0, 3))

        end_point = self._gland_thyroid.get_rightmost_point(z)

        if self._trachea.has_points(z) and end_point[0] < self._trachea.get_rightmost_point(z)[0]:
            # Thyroid gland on right side of trachea does not exist.
            return np.empty((0, 3))

        start_point = self._gland_thyroid.get_closest_point_to_structure(self._carotid, z)
        return self._gland_thyroid.get_points_between(start_point, end_point)

    def _get_sterno_thyroid_points(self, z):
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

    @property
    def path(self):
        return self._path


def _add_neck_node_level_4a_left_to_dicom(rt_path, ct_path, output_path, name='Level_IVa_left'):
    """ input_path: path to rt struct, output_path: path to folder containing rt struct and ct images """
    neck_node_level = NeckNodeLevel4aLeft(rt_path, ct_path)
    neck_node_level.remove_self_intersections()
    neck_node_level.interpolate_in_z()
    neck_node_level.clip_corners(neck_node_level.extract_structure_endpoints())
    neck_node_level.interpolate_in_xy()
    neck_node_level.remove_intersections()
    # The following line can be uncommented to plot the neck node level before saving it.
    # neck_node_level.plot()
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
