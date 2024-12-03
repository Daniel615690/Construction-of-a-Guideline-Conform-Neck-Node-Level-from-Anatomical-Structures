from interpolation import _is_positive_oriented
import numpy as np


def _orient_positive(contour):
    """
    Orient a contour in mathematical positive orientation.

    :param contour: A polygonal chain.
    :type contour: np.ndarray
    :return: The oriented contour.
    :rtype: np.ndarray
    """
    oriented_contour = []
    for z in np.unique(contour[:, 2]):
        contour_slice = contour[contour[:, 2] == z]
        if _is_positive_oriented(contour_slice):
            oriented_contour.extend(contour_slice)
        else:
            oriented_contour.extend(contour_slice[::-1])
    oriented_contour = np.array(oriented_contour)

    return oriented_contour


def get_orthogonal_in_xy(vector):
    """
    Compute an orthogonal vector in the xy-plane.

    This function computes an orthogonal vector by swapping the x- and y-components of the input vector.
    If the input vector has more than two dimensions, the additional components (e.g., z) remain unchanged.
    The resulting vector lies in the same xy-plane as the input.

    :param vector: A 2D or higher-dimensional vector.
    :type vector: np.ndarray
    :return: A vector orthogonal to the input in the xy-plane, with unchanged additional dimensions.
    :rtype: np.ndarray
    """
    orthogonal = np.copy(vector)
    orthogonal[0], orthogonal[1] = vector[1], vector[0]
    return orthogonal


def closest_to_point(points, point):
    """
    Find the point closest to a given reference point.

    This function computes the Euclidean distance between the reference point and each point in the provided
    set of points, and returns the closest point along with its distance to the reference point.
    The function works for points in any dimensional space.

    :param points: A 2D NumPy array where each row represents a point in n-dimensional space.
    :type points: np.ndarray
    :param point: A 1D NumPy array representing the reference point to which the closest point is determined.
    :type point: np.ndarray
    :return: The closest point to the reference point and the Euclidean distance between them.
    :rtype: (np.ndarray, float)
    """
    distances = np.linalg.norm(points - point, axis=1)
    min_index = np.argmin(distances)
    closest_point = points[min_index]
    min_distance = distances[min_index]
    return closest_point, min_distance


def closest_between_points(points_1, points_2):
    """
    Find the closest pair of points between two sets of points.

    This function calculates the pair of points, one from each of the two provided sets,
    that are closest to each other based on Euclidean distance. The function returns
    the closest points from both sets, as well as the Euclidean distance between them.

    :param points_1: A 2D NumPy array where each row represents a point in n-dimensional space.
    :type points_1: np.ndarray
    :param points_2: A 2D NumPy array where each row represents a point in n-dimensional space.
    :type points_2: np.ndarray
    :return: The closest pair of points, one from each set, and their Euclidean distance.
    :rtype: (np.ndarray, np.ndarray, float)
    """
    differences = points_1[np.newaxis, :, :] - points_2[:, np.newaxis, :]
    distances = np.linalg.norm(differences, axis=2)
    index_points_2, index_points_1 = np.unravel_index(np.argmin(distances), distances.shape)
    min_points_1 = points_1[index_points_1]
    min_points_2 = points_2[index_points_2]
    return min_points_1, min_points_2, distances[index_points_2, index_points_1]