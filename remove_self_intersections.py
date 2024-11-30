import numpy as np
from numpy.linalg import lstsq
from pydicom import dcmread
import matplotlib.pyplot as plt
from interpolation import _close_contour


def _find_intersection(p_0, p_1, p_2, p_3):
    """
    Finds intersection between (p_0, p_1) and (p_2, p_3) that are not one of p_0, p_1, p_2 or p_3.

    :type p_0: np.ndarray
    :type p_1: np.ndarray
    :type p_2: np.ndarray
    :type p_3: np.ndarray
    :return: The intersection between (p_0, p_1) and (p_2, p_3) that are not one of p_0, p_1, p_2 or p_3.
    :rtype: np.ndarray
    """
    # solve the equation p_0 + s(p_1 - p_0) = p_2 + t(p_3 - p_2) for s, t
    a = np.array([p_1 - p_0, p_2 - p_3]).T
    b = p_2 - p_0
    (s, t), residual, _, _ = lstsq(a, b, None)
    if np.allclose(residual, 0) and 0 < s < 1 and 0 < t < 1 \
            and not np.isclose(s, 0) and not np.isclose(s, 1) \
            and not np.isclose(t, 0) and not np.isclose(t, 1):
        # lines do intersect
        return p_0 + s * (p_1 - p_0)


def _find_intersecting_edge(polygon, edge_start, edge_end, start_index):
    """
    Finds an edge intersecting with ('edge_start', 'edge_end') in 'polygon', starting at index 'start_index'.

    Points that are part of 'polygon' are not considered to be intersections.

    :param polygon: 3D points that form a polygon.
    type polygon: list[np.ndarray]
    :param edge_start: The 3D start point of the edge that an intersection is searched for.
    :type edge_start: np.ndarray
    :param edge_end: The 3D end point of the edge that an intersection is searched for
    :type edge_end: np.ndarray
    :param int start_index: The index in 'polygon' at which the search starts.
    :return: The intersection and the start index of the edge intersecting with ('edge_start', 'edge_end').
    :rtype: (np.ndarray, int)
    """
    for i in range(start_index, len(polygon) - 1):
        intersection = _find_intersection(edge_start, edge_end, polygon[i], polygon[i + 1])
        if intersection is not None:
            return intersection, i
    return None, None


def _resolve_intersections(polygon):
    """
    Divides a polygon with self-intersections into a list of polygons without self-intersections.

    The polygon is divided at intersection points. Points that are part of 'polygon' are not considered to be
    intersections.

    :param polygon: A list of 3D points that form the polygon.
    :type polygon: list[np.ndarray]
    :return: The polygons without self-intersection.
    :rtype: list[list[np.ndarray]]
    """
    polygons_without_intersections = []
    i = 0
    while i < len(polygon) - 1:  # range does not work as polygon changes
        intersection, j = _find_intersecting_edge(polygon, polygon[i], polygon[i + 1], i + 1)
        if intersection is not None:
            polygon_1 = polygon[:i + 1]
            polygon_1.append(intersection)
            polygon_1.extend(polygon[j + 1:len(polygon)])
            polygon_2 = [intersection]
            polygon_2.extend(polygon[i + 1:j + 1])
            polygon_2.append(intersection)
            polygons_without_intersections.extend(_resolve_intersections(polygon_2))
            polygon = polygon_1
        i += 1
    polygons_without_intersections.append(polygon)
    return polygons_without_intersections


def _clip_polygon(polygon):
    """
    Removes self-intersections in the polygon.

    The polygon is divided at intersection points. The polygon with the highest number of points remains. Points that
    are part of 'polygon' are not considered intersections.

    :param polygon: The 3D points that form the polygon.
    :type polygon: list[np.ndarray]
    :return: The 3D points that form the polygon without intersections.
    :rtype: list[np.ndarray]
    """
    polygons_without_intersections = _resolve_intersections(polygon)
    largest_polygon_index = np.argmax([len(polygon) for polygon in polygons_without_intersections])
    return polygons_without_intersections[largest_polygon_index]


def clip_contour(contour):
    """
    Removes self-intersections in the contour.

    For each slice, the contour is divided at intersections. The contour with the highest number of points is retained.
    Contour points are not considered intersections.

    :param contour: The 3D points that form the contour.
    :type contour: np.ndarray
    :return: The 3D points that form the contour without intersections.
    :rtype: np.ndarray
    """
    clipped_contour = []
    for z in np.unique(contour[:, 2]):
        polygon = [point for point in contour[contour[:, 2] == z]]
        clipped_polygon = _clip_polygon(polygon)
        clipped_contour.extend(clipped_polygon)
    return np.array(clipped_contour)


def get_all_labels(path):
    """
    Return the labels of all structures defined in the DICOM file at the specified path.

    :param path: The path to the RT-struct of the DICOM file.
    :type path: str
    :return: The labels of all structures defined in the DICOM file.
    :rtype: set[str]
    """
    ds = dcmread(path)

    labels = set()
    for structureSetROISequence, ROIContourSequence in zip(ds.StructureSetROISequence, ds.ROIContourSequence):
        labels.add(structureSetROISequence.ROIName)
    return labels


def plot_contour(*contours, path, caudal_boundary, cranial_boundary, slice_thickness, points=None,
                 plot_anatomical_structures=True, anatomical_structures=None, fixed_view=False,
                 marker_anatomical_structures='o', marker_contour='x', marker_points='v', slices=None,
                 full_screen=True, contours_labels=None, grid=True, order=-1, print_boundaries=False):
    """
    Plot the contours of a neck node level.

    :param contours: The contours as an array of polygonal chains.
    :type contours: list[np.ndarray]
    :param path: The path to the RT-struct of the DICOM file associated with the contours.
    :type path: str
    :param caudal_boundary: The z-value of the caudal boundary (exclusively) of this neck node level in mm.
    :type caudal_boundary: np.ndarray
    :param cranial_boundary: The z-value of the cranial boundary (exclusively) of this neck node level in mm.
    :type cranial_boundary: np.ndarray
    :param slice_thickness: The distance between each axial slice.
    :type slice_thickness: float
    :param points: Points that shall be plotted in addition to this neck node level.
    :type points: np.ndarray
    :param plot_anatomical_structures: Whether to plot anatomical structures.
    :type plot_anatomical_structures: bool
    :param anatomical_structures: The labels in the DICOM file of the anatomical structures that are to be plotted.
        If not set, the thyroid gland, trachea, left common carotid artery, left internal jugular vein, left
        anterior scalene muscle, left middle scalene muscle, left sternocleidomastoid muscle and left sternothyroid
        muscle are plotted.
    :type anatomical_structures: list[str]
    :param fixed_view: Whether the window of the plot shall be fixed between slices.
    :type fixed_view: bool
    :param marker_anatomical_structures: The marker used to plot the anatomical structures.
    :type marker_anatomical_structures: str
    :param marker_contour: marker_contour: The marker used to plot this neck node level.
    :type marker_contour: str
    :param marker_points: The marker used to plot `points`.
    :type marker_points: str
    :param slices: The z-value of the slices that are to be plotted. If not set, all slices of this neck node level
        will be plotted.
    :type slices: np.ndarray[float]
    :param full_screen: Whether to show the plot in fullstreen.
    :type full_screen: bool
    :param contours_labels: The name of each contour in the plot.
    :type contours_labels: list[str]
    :param grid: Whether to show the plot grid.
    :type grid: bool
    :param order: The order in which the slices are plotted. -1 for plotting from `cranial_boundary` to
        `caudal_boundary`. 1 for plotting from `caudal_boundary` to `cranial_boundary`.
    :type order: int
    :param print_boundaries: Whether to plot the cranial and caudal boundaries.
    :type print_boundaries: bool
    """
    if anatomical_structures is None:
        anatomical_structures = {'GLAND_THYROID': 'Thyroid Gland',
                                 'TRACHEA': 'Trachea',
                                 'ARTERY_COMMONCAROTID_LINKS': 'Left Common Carotid Arterie',
                                 'IJV_LINKS': 'Left Internal Jugular Vein',
                                 'M_SCALENUS_ANTERIOR_LINKS': 'Left Anterior Scalene Muscle',
                                 'M_SCALENUS_MEDIUS_LINKS': 'Left Middle Scalene Muscle',
                                 'M_STERNOCLEIDOMASTOID_LINKS': 'Left Sternocleidomastoid Muscle',
                                 'M_STERNO_THYROID_LINKS': 'Left Sternothyroid Muscle'}
    ds = dcmread(path)

    structures = {}
    for structureSetROISequence, ROIContourSequence in zip(ds.StructureSetROISequence, ds.ROIContourSequence):
        ROIName = structureSetROISequence.ROIName
        coordinates = []
        for contour in ROIContourSequence.ContourSequence:
            coordinates.extend(contour.ContourData)
        structures[ROIName] = np.array(coordinates).reshape(-1, 3)

    first_slice = caudal_boundary if print_boundaries else caudal_boundary + slice_thickness
    last_slice = cranial_boundary if print_boundaries else cranial_boundary - slice_thickness

    contours = [contour for contour in contours if len(contour) > 0]
    plot_points = np.vstack((np.vstack([structures[name] for name in anatomical_structures]),
                             np.vstack(contours).reshape(-1, 3)))
    plot_points = plot_points[np.isin(plot_points[:, 2],
                                      np.arange(first_slice, last_slice + slice_thickness, slice_thickness))]
    x_min = np.min(plot_points[:, 0])
    x_max = np.max(plot_points[:, 0])
    y_min = np.min(plot_points[:, 1])
    y_max = np.max(plot_points[:, 1])

    for z in np.arange(first_slice, last_slice + slice_thickness, slice_thickness)[::order]:
        if slices is not None and z not in slices:
            continue

        if contours_labels is None:
            for contour in contours:
                contour_slice = _close_contour(contour[contour[:, 2] == z])
                plt.plot(contour_slice[:, 0], contour_slice[:, 1], marker=marker_contour)
        else:
            for contour, label in zip(contours, contours_labels):
                contour_slice = _close_contour(contour[contour[:, 2] == z])
                plt.plot(contour_slice[:, 0], contour_slice[:, 1], marker=marker_contour, label=label)

        if points is not None:
            plt.scatter(points[points[:, 2] == z][:, 0], points[points[:, 2] == z][:, 1], marker=marker_points)

        for name in anatomical_structures.keys():
            structure = structures[name]
            if plot_anatomical_structures:
                plt.scatter(structure[structure[:, 2] == z][:, :2][:, 0],
                            structure[structure[:, 2] == z][:, :2][:, 1],
                            label=anatomical_structures[name], marker=marker_anatomical_structures)

        if not fixed_view:
            plt.xlim(x_min - 2, x_max + 2)
            plt.ylim(y_min - 2, y_max + 2)
        plt.gca().invert_yaxis()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'z = {z}')
        if plot_anatomical_structures:
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.gca().set_aspect('equal')
        if full_screen:
            plt.get_current_fig_manager().full_screen_toggle()
        if grid:
            plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()


def create_circles(centres, radius, num_points):
    return np.array([[centre[0] + radius * np.cos(phi), centre[1] + radius * np.sin(phi), centre[2]]
                     for centre in centres
                     for phi in np.linspace(0, 2 * np.pi, num_points)])


def save_plot(contour_3d, path):
    names = ['GLAND_THYROID',
             'TRACHEA',
             'ARTERY_COMMONCAROTID_LINKS',
             'IJV_LINKS',
             'M_SCALENUS_ANTERIOR_LINKS',
             'M_SCALENUS_MEDIUS_LINKS',
             'M_STERNOCLEIDOMASTOID_LINKS',
             'M_STERNO_THYROID_LINKS']
    PATH = r'C:\Users\danie\Documents\Studium\Bachelorarbeit\Dateien_von_Alexandra\Extract_Polygons\data\HN_P001_pre\RS1.2.752.243.1.1.20230309171637404.1400.85608.dcm'
    ds = dcmread(PATH)

    structures = {}
    for structureSetROISequence, ROIContourSequence in zip(ds.StructureSetROISequence, ds.ROIContourSequence):
        ROIName = structureSetROISequence.ROIName
        coordinates = []
        for contour in ROIContourSequence.ContourSequence:
            coordinates.extend(contour.ContourData)
        structures[ROIName] = np.array(coordinates).reshape(-1, 3)

    for z in np.unique(contour_3d[:, 2]):
        for name in names:
            structure = structures[name]
            plt.scatter(structure[structure[:, 2] == z][:, :2][:, 0],
                        structure[structure[:, 2] == z][:, :2][:, 1],
                        label=name)
        plt.plot(contour_3d[contour_3d[:, 2] == z][:, 0], contour_3d[contour_3d[:, 2] == z][:, 1])
        plt.scatter(contour_3d[contour_3d[:, 2] == z][:, 0], contour_3d[contour_3d[:, 2] == z][:, 1], marker='^')
        plt.gca().invert_yaxis()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'z = {z}')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(f'{path}_{z}.png', bbox_inches='tight')
        plt.clf()


def plot_polygons(polygons, z):
    names = ['GLAND_THYROID',
             'TRACHEA',
             'ARTERY_COMMONCAROTID_LINKS',
             'IJV_LINKS',
             'M_SCALENUS_ANTERIOR_LINKS',
             'M_SCALENUS_MEDIUS_LINKS',
             'M_STERNOCLEIDOMASTOID_LINKS',
             'M_STERNO_THYROID_LINKS']
    PATH = r'C:\Users\danie\Documents\Studium\Bachelorarbeit\Dateien_von_Alexandra\Extract_Polygons\data\HN_P001_pre\RS1.2.752.243.1.1.20230309171637404.1400.85608.dcm'
    ds = dcmread(PATH)

    structures = {}
    for structureSetROISequence, ROIContourSequence in zip(ds.StructureSetROISequence, ds.ROIContourSequence):
        ROIName = structureSetROISequence.ROIName
        coordinates = []
        for contour in ROIContourSequence.ContourSequence:
            coordinates.extend(contour.ContourData)
        structures[ROIName] = np.array(coordinates).reshape(-1, 3)

    for name in names:
        structure = structures[name]
        plt.scatter(structure[structure[:, 2] == z][:, :2][:, 0],
                    structure[structure[:, 2] == z][:, :2][:, 1],
                    label=name)
    for polygon in polygons:
        polygon = np.array(polygon)
        plt.plot(polygon[polygon[:, 2] == z][:, 0], polygon[polygon[:, 2] == z][:, 1])
    plt.gca().invert_yaxis()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'z = {z}')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
