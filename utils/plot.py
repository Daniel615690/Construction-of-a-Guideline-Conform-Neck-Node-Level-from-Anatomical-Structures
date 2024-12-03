import numpy as np
from matplotlib import pyplot as plt
from pydicom import dcmread

from interpolation import _close_contour


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
