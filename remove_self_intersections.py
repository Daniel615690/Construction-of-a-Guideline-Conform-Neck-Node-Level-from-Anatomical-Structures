import numpy as np
from numpy.linalg import lstsq


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
