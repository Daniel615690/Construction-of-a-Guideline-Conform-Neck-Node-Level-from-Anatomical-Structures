from interpolation import _is_positive_oriented
import numpy as np


def _orient_positive(contour):
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
    orthogonal = np.copy(vector)
    orthogonal[0], orthogonal[1] = vector[1], vector[0]
    return orthogonal


def closest_to_point(points, point):
    distances = np.linalg.norm(points - point, axis=1)
    min_index = np.argmin(distances)
    closest_point = points[min_index]
    min_distance = distances[min_index]
    return closest_point, min_distance


def closest_between_points(points_1, points_2):
    differences = points_1[np.newaxis, :, :] - points_2[:, np.newaxis, :]
    distances = np.linalg.norm(differences, axis=2)
    index_points_2, index_points_1 = np.unravel_index(np.argmin(distances), distances.shape)
    min_points_1 = points_1[index_points_1]
    min_points_2 = points_2[index_points_2]
    return min_points_1, min_points_2, distances[index_points_2, index_points_1]


def sort_array(array, order):
    """ Sort a 2D array of shape (-1, 3) based on a 2D array of shape (-1, 3). Elements that do not occur in the order
    array are filtered out. """
    indices = np.argwhere(np.all(order[:, None] == array, axis=2))[:, 0]
    return order[indices]