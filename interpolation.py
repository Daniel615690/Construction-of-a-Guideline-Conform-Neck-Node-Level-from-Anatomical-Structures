import numpy as np
import bezier
from functools import reduce
from pydicom import dcmread
from shapely import LineString

# A dictionary storing a label for each point in the contour.
# Points are converted to tuples (as np.ndarray is not hashable) and used as keys with the corresponding label as value.
_point_to_label_dict = {}


def _consecutive_occurrences(array, start):
    """
    Count the number of consecutive occurrences of the element at index `start` in `array`.

    This function counts the number of elements that directly follow and match the element at index `start`, wrapping
    around the end of the `array` if necessary.
    The count includes the element at index `start`.
    :param array: The array to search through.
    :type array: array_like
    :param start: The index of the element to start counting from.
    :type start: int
    :return: The number of consecutive occurrences of the element at index `start` (inclusively).
    :rtype: int
    """
    number_consecutive_occurrences = 1
    for i in range(1, len(array) - 1):
        if np.allclose(array[start], array[(start + i) % len(array)]):
            number_consecutive_occurrences += 1
        else:
            break
    return number_consecutive_occurrences


def _close_contour(contour):
    """
    Ensures that a given 2D contour is closed by connecting the first point to the last.

    If the contour is not closed, a copy is returned with the first point connected to the last.
    :param contour: The contour to close.
    :type contour: array_like
    :return: Closed contour.
    :rtype: array_like
    """
    if len(contour) > 0 and not np.allclose(contour[0], contour[-1]):
        contour = np.insert(contour, len(contour), contour[0], axis=0)
    return contour


def _open_contour(contour):
    """
    Ensures that a given 2D contour is not closed, i.e. the first and last point are different.

    If the contour is closed, a copy is returned with the last point removed.
    :param contour: The contour to open.
    :type contour: array_like
    :return: Open contour.
    :rtype: array_like
    """
    if len(contour) > 0 and np.allclose(contour[0], contour[-1]):
        return contour[:-1]
    return contour


def _intersect_line_segment_circle(line_segment_start, line_segment_end, centre, radius):
    """
    This function calculates the intersection of a line segment with a circle.

    A line segment is part of a line, bounded by two points.
    The number of intersection points can be 0, 1 or 2. If the line segment is a tangent to the circle, the intersection
    is returned twice.

    :param line_segment_start: The start of the line segment.
    :type line_segment_start: np.ndarray
    :param line_segment_end: The end of the line segment.
    :type line_segment_end: np.ndarray
    :param centre: The centre of the circle.
    :type centre: np.ndarray
    :param radius: The radius of the circle. Must be > 0.
    :type radius: float
    :return: An array of the intersection points. If there is no intersection, an empty array is returned.
    :rtype: np.ndarray
    """
    a = np.dot(line_segment_end - line_segment_start, line_segment_end - line_segment_start)
    b = 2 * np.dot(line_segment_end - line_segment_start, line_segment_start - centre)
    c = np.dot(line_segment_start - centre, line_segment_start - centre) - radius**2

    # solve the equation a * t**2 + b * t + c = 0
    t = np.roots([a, b, c])

    # circle intersects with line iff 0 <= t <= 1 and iff t is real
    t = t[(0 <= t) & (t <= 1) & np.isreal(t)]

    return line_segment_start + np.outer(t, line_segment_end - line_segment_start)


def _interpolate_points(points, num_interpolation_points):
    """
    Interpolate points along a Bézier curve.

    This function calculates the Bézier curve defined by the input points and returns `num_interpolation_points`
    points placed the curve. The first point is at the start of the curve, and the last point is at the end of the
    curve.

    :param points: 2D array of control points for the Bézier curve.
    :type points: np.ndarray
    :param num_interpolation_points: Number of points to interpolate along the curve.
    :type num_interpolation_points: int
    :return: 2D array of `num_interpolation_points` points placed along the Bézier curve.
    :rtype: np.ndarray
    """
    curve = bezier.Curve(points.T, len(points) - 1)
    t = np.linspace(0, 1, num_interpolation_points)
    return np.array(curve.evaluate_multi(t).T.tolist())


def _get_contour_segment_within_circle(contour, centre_index, radius):
    """
    Extracts the part of the 2D contour that lies inside a circle centred on a specific contour point.

    This function determines the contour segment that lies inside or on the boundary of a circle centred at a specified
    point on the contour. Among potentially several segments, it selects the one where the centre point of the circle is
    part of the segment. The segment extends in both directions from the centre point and includes the two intersections
    of the contour with the circle. The direction of the segment is that of the contour.

    :param contour: Closed 2D contour represented as a 2D array of points.
    :type contour: np.ndarray
    :param centre_index: Index of the point on the contour to use as the circle's centre.
    :type centre_index: int
    :param radius: Radius of the circle. Must be > 0.
    :type: radius: float
    :return:
        contour:
            The contour segment inside or on the circle that includes the centre point.
        start_index:
            Index in 'contour' of the segment's first point.
        end_index:
            Index in 'contour' of the point after the segment's last point.
    :rtype: (np.ndarray, int, int)
    """
    centre_index = (centre_index + len(contour)) % len(contour)
    segment = [contour[centre_index]]
    segment_start, segment_end = None, None

    # Find the contour segment within the circle, starting from the centre point and extending backwards.
    for i in range(len(contour)):
        start_index, end_index = centre_index - i, centre_index - i - 1
        line_start, line_end = contour[start_index], contour[end_index]
        line_intersections = _intersect_line_segment_circle(line_start, line_end, contour[centre_index], radius)

        if len(line_intersections) > 0:
            # Only one intersection is possible as the line segment starts within the circle.
            line_intersection = line_intersections[0]

            # Add the contour points on the circles boundary to the segment.
            if np.allclose(line_end, line_intersection):
                positive_end_index = (end_index + len(contour)) % len(contour)
                reversed_end_index = len(contour) - 1 - positive_end_index
                number_consecutive_occurrences = _consecutive_occurrences(contour[::-1], reversed_end_index)
                last_consecutive_occurrence = end_index - number_consecutive_occurrences + 1

                segment_start = last_consecutive_occurrence
                segment = number_consecutive_occurrences * [line_end] + segment
            else:
                segment_start = start_index
                segment = [line_intersection] + segment
            break
        segment = [line_end] + segment

    if segment_start is None:
        # No intersection found. => Circle encloses contour.
        return contour, 0, len(contour)

    # Find the contour segment within the circle, starting from the centre point and extending forward.
    for i in range(len(contour)):
        start_index, end_index = (centre_index + i) % len(contour), (centre_index + i + 1) % len(contour)
        line_start, line_end = contour[start_index], contour[end_index]
        line_intersections = _intersect_line_segment_circle(line_start, line_end, contour[centre_index], radius)

        if len(line_intersections) > 0:
            # Only one intersection is possible as the line segment starts within the circle.
            line_intersection = line_intersections[0]

            # Add the contour points on the circles boundary to the segment.
            if np.allclose(line_end, line_intersection):
                number_consecutive_occurrences = _consecutive_occurrences(contour, end_index)
                last_consecutive_occurrence = (end_index + number_consecutive_occurrences - 1) % len(contour)

                segment_end = last_consecutive_occurrence + 1
                segment = segment + number_consecutive_occurrences * [line_end]
            else:
                segment_end = end_index
                segment = segment + [line_intersection]
            break
        segment = segment + [line_end]

    return np.array(segment), segment_start, segment_end


def _find_index(elem, array):
    """
    Find the index of the first occurrence of `elem` in `array`.

    If no occurrence is found, None is returned.
    :param elem: The element to be found.
    :type elem: array_like
    :param array: The array to search through.
    :type array: array_like
    :return: The index of the first occurrence of `elem` in `array`, or None if no occurrence is found.
    :rtype: array_like or None
    """
    for i, candidate in enumerate(array):
        if np.allclose(candidate, elem):
            return i
    return None


def _slice_wrap_around(array, start, end):
    """
    Return a slice of an array, handling wrap-around cases.

    This function returns the part of the `array` from `start` (inclusively) to `end` (exclusively), handling
    wrap-around cases in when `start` > `end`. Negative indices are supported.
    :param array: The input array from which to extract the slice.
    :type array: array_like
    :param start: The start index (inclusive).
    :type start: int
    :param end: The end index (exclusive).
    :type end: int
    :return: A new array containing the elements from `start` to `end`, with wrap-around handling.
    :rtype: np.ndarray
    """
    start = min(max(-len(array), start), len(array))
    end = min(max(-len(array), end), len(array))

    start = start if start >= 0 else start + len(array)
    end = end if end >= 0 else end + len(array)

    if start <= end:
        return array[start:end]
    else:
        return np.concatenate((array[start:], array[:end]))


def _replace_section_wrap_around(array, start, end, replacement):
    """
    Replace part of an array with a new array.

    This function deletes all elements from `start` (inclusively) to `end` (exclusively) of the array and inserts the
    replacement array before index start. The function handles wrap-around cases in case of start > end.
    If `start` == `end`, no element is deleted. Negative indices are supported.
    A new array is returned. The original array is not changed.
    :param array: The array to be replaced.
    :type array: array-like
    :param start: Index of the first element to be replaced, inclusively.
    :type start: int
    :param end: Index of the last element to be replaced, exclusively.
    :type end: int
    :param replacement: The array that is used for replacement. Must have the same dimension as `array`.
    :type replacement: array-like
    :return: A new array where the elements from `start` to `end` are replaced by `replacement`.
    :rtype: array_like
    """
    start = (start + len(array)) % len(array)
    end = (end + len(array)) % len(array)

    if start <= end:
        return np.concatenate((array[:start], replacement, array[end:]))
    else:
        return np.concatenate((array[end:start], replacement))


def _delete_section_wrap_around(array, start, end):
    """
    Delete a section of an array.

    This function deletes all elements from `start` (inclusively) to `end` (exclusively) of the array. The function
    handles wrap-around cases in case of start > end.
    If `start` == `end`, no element is deleted. Negative indices are supported.
    A new array is returned. The original array is not changed.
    :param array: The array.
    :type array: np.ndarray
    :param start: Index of the first element to be deleted, inclusively.
    :type start: int
    :param end: Index of the last element to be deleted, exclusively.
    :type end: int
    :return: A new array where the elements from `start` to `end` are deleted.
    :rtype: np.ndarray
    """
    empty = np.empty((0,) + array.shape[1:])
    return _replace_section_wrap_around(array, start, end, empty)


def _unique_array(array):
    """
    Find the unique elements of an array.

    This functions returns the unique elements of an array. In contrast to :func:`numpy.unique`, the elements are not
    sorted.
    :param array: Input array.
    :type array: np.ndarray
    :return: The unique elements of the input array.
    :rtype: np.ndarray
    """
    _, indices = np.unique(array, axis=0, return_index=True)
    indices.sort()
    return array[indices]


def _find_nearest_distinct_neighbors(array, index):
    """
    Find the two nearest neighbors that are distinct from the element at the specified index.

    This function searches for the two nearest elements in the array that are different from the element at the
    specified index. The array is treated as circular array. If no neighbor is found, (None, None) is returned.
    :param array: The input array to search.
    :type array: array_like
    :param index: The index of the reference element.
    :type index: int
    :return:
        left_neighbor:
            The first distinct element, transversing from index to the left, or None if there is no distinct neighbor.
        right_neighbor:
            The first distinct element, transversing from index to the right, or None if there is no distinct neighbor.
    :rtype: (any, any)
    """
    left_neighbor, right_neighbor = None, None
    for i in range(1, len(array) - 1):
        left_index = ((index - i) + len(array)) % len(array)
        right_index = (index + i) % len(array)
        if left_neighbor is None and not np.allclose(array[left_index], array[index]):
            left_neighbor = array[left_index]
        if right_neighbor is None and not np.allclose(array[index], array[right_index]):
            right_neighbor = array[right_index]
    return left_neighbor, right_neighbor


def _angle(left_point, middle_point, right_point):
    """
    Calculate the angle between three points in radians.

    This function computes the angle formed by the line segments from the middle point to the left point and from the
    middle point to the right point. The angle always lies between 0 and pi (inclusively). If either the left point or
    the right point coincides with the middle point, the angle is undefined, and the function returns None.
    :param left_point: The left point as 1d array.
    :type left_point: array_like
    :param middle_point: The middle point as 1d array.
    :type middle_point: array_like
    :param right_point: The right point as 1d array.
    :type right_point: array_like
    :return: The angle in radians between the three points if defined, or None if the angle is undefined.
    :rtype: float or None
    """
    a, b, c = np.asarray(left_point), np.asarray(middle_point), np.asarray(right_point)

    ba = b - a
    bc = b - c

    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return None
    ba = ba / np.linalg.norm(ba)
    bc = bc / np.linalg.norm(bc)

    # Dot product must be clipped to [-1, 1] in case of rounding errors.
    return np.arccos(np.clip(np.dot(ba, bc), -1, 1))


def _intersection(*lists):
    """
    Calculates the intersection of a multiple lists.

    The order of the resulting list is defined by the first list in lists.
    :param lists: The lists to intersect.
    :type lists: tuple(list[Any])
    :return: The intersection of the lists.
    :rtype: list[Any]
    """
    intersection = reduce(set.intersection, map(set, lists))
    ordered_intersection = [elem for elem in lists[0] if elem in intersection]
    return ordered_intersection


def _is_positive_oriented(contour):
    """
    Determine if a closed 2D contour is positively oriented.

    This function calculates the orientation of a closed 2D contour. It returns True if the contour is mathematically
    positive oriented and False otherwise.
    :param contour: The closed contour represented as a 2d array of points.
    :type contour: np.ndarray
    :return: True if the contour is mathematically positive oriented and False otherwise.
    :rtype: bool
    """
    contour = _close_contour(contour)
    # Calculate the orientation with the shoelace formula.
    x_values = contour[:, 0]
    y_values = contour[:, 1]
    signed_area = np.sum((x_values[:-1] * y_values[1:] - x_values[1:] * y_values[:-1]))
    return signed_area >= 0


def _adjust_orientation(*contours):
    """
    Make sure that all contours have the same orientation.

    This function adjusts the orientation of a sequence of contours if necessary, so that all contours have the same
    orientation.
    :param contours: The contours.
    :type contours: array_like
    :return: None
    """
    goal_orientation = _is_positive_oriented(contours[0])
    for contour in contours[1:]:
        if _is_positive_oriented(contour) != goal_orientation:
            contour[:] = contour[::-1]


def _initialize_labels(dcm_path):
    """
    Initialize the global variable `_point_to_label_dict`.

    This function extracts the label for each point of the contour as defined in the DICOM file at `dcm_path` and stores
    the point-label pair in the dictionary `_point_to_label_dict`. The points are stored as keys, the labels as value.
    The points are converted to tuples, as `np.ndarray` is not hashable.
    :param dcm_path: The path to the rt struct.
    :type dcm_path: str
    :return: None
    """
    global _point_to_label_dict
    dataset = dcmread(dcm_path)
    for structureSetROISequence, ROIContourSequence in zip(dataset.StructureSetROISequence, dataset.ROIContourSequence):
        roi_name = structureSetROISequence.ROIName

        coordinates = []
        for contour in ROIContourSequence.ContourSequence:
            coordinates.extend(contour.ContourData)
        coordinates = np.array(coordinates).reshape(-1, 3)

        for coordinate in coordinates:
            _point_to_label_dict[tuple(coordinate)] = roi_name


def _get_label(point, dcm_path):
    """
    Return the label of a point as defined in the DICOM file at `dcm_path`.

    This functions returns the label of a point as defined in the DICOM file at `dcm_path`, or None if no label can be
    found. `_point_to_label_dict` is initialized, if it is empty.
    :param point: The point of which the label is retrieved.
    :type point: np.ndarray
    :param dcm_path: The path to the rt struct.
    :type dcm_path: str
    :return: The label of the point, or None if no label can be found.
    :rtype: str
    """
    if _point_to_label_dict == {}:
        _initialize_labels(dcm_path)

    point = tuple(point)
    if point in _point_to_label_dict:
        return _point_to_label_dict[point]
    return None


def _extract_labels(contour, dcm_path):
    """
    Extract the unique labels for a sequence of points in a contour.

    This function extracts the labels of the contour points. It maintains the order of labels based on their first
    appearance in `contour`.
    :param contour: The contour for which the labels are retrieved as a 2D array of points.
    :type contour: np.ndarray
    :param dcm_path: The path to the rt struct.
    :type dcm_path: str
    :return: A list of unique labels, ordered according to their first appearance in `contour`.
    :rtype: list[str]
    """
    labels = []
    for point in contour:
        label = _get_label(point, dcm_path)
        if label is not None and label not in labels:
            labels.append(label)
    return labels


def _extract_points(contour, label, dcm_path):
    """
    Extract points from a contour that match a specific label.

    This function filters the points in the given contour, retaining only those that have the specified label. The
    resulting points maintain their original order as they appear in the input contour.
    :param contour: The contour as 2D array of points.
    :type contour: np.ndarray
    :param label: The label to filter points by.
    :type label: str
    :param dcm_path: The path to the rt struct.
    :type dcm_path: str
    :return: 2D array of the filtered points, ordered as in `contour`.
    """
    extracted_points = []
    for point in contour:
        if _get_label(point, dcm_path) == label:
            extracted_points.append(point)
    return np.array(extracted_points)


def _extract_points_between(contour, start_label, end_label, dcm_path):
    """
    Extract points from a 2D contour between specified start and end labels.

    This function returns the points in the contour between the last occurrence of the specified start label and the
    first occurrence of the specified end label (inclusive of both). The label of a point is determined from the DICOM
    file stored at `dcm_path`. The function wraps around the contour if the end label appears before the start label. If
    the contour doesn't contain a point with the start or end label, an empty array is returned.
    :param contour: The contour as a 2D array of points.
    :type contour: np.ndarray
    :param start_label: The label marking the start of the extraction.
    :type start_label: str
    :param end_label: The label marking the end of the extraction.
    :type end_label: str
    :param dcm_path: The path to the rt struct.
    :type dcm_path: str
    :return: A 2D array containing the extracted points between the specified labels.
    :rtype: np.ndarray
    """
    last_start_label_index = None
    for i, point in enumerate(contour):
        label = _get_label(point, dcm_path)
        if label == start_label:
            last_start_label_index = i

    first_end_label_index = None
    for i, point in enumerate(contour[::-1]):
        label = _get_label(point, dcm_path)
        if label == end_label:
            first_end_label_index = len(contour) - 1 - i

    if start_label is None or end_label is None:
        # start_label or end_label not found
        return np.empty((0, 3))

    return _slice_wrap_around(contour, last_start_label_index, first_end_label_index + 1)


def _set_points(chain, point_distance):
    """
    Set points in the specified distance on a polygonal chain.
    
    This function sets points in approximately the specified distance on a polygonal chain. The first points is set at
    the start of the polygonal chain, the last point at the end. First and last point are always set, even when the
    specified distance is larger than the length of the chain.
    :param chain: A sequence of at least two points forming a polygonal chain.
    :type chain: sequence of array_like
    :param point_distance: The positive distance at which the points are set.
    :type point_distance: float
    :return: A 2D array of points set in the specified distance on the polygonal chain.
    :rtype: np.ndarray
    """
    if len(chain) < 2:
        return chain

    line = LineString(chain[:, :2])

    # Always set first and last point on the chain.
    num_points = 2
    if point_distance > 0:
        num_points = max(num_points, round(line.length / point_distance))

    ts = np.linspace(0, line.length, num_points)
    return np.array([[point.x, point.y, chain[0][2]] for point in line.interpolate(ts)])


def _interpolate_chain(chain_a, chain_b, chain_c, point_distance):
    """
    Interpolates two polygonal chains.

    This function creates an interpolated polygonal chain between two input polygonal chains by setting points at
    approximately `point_distance` intervals along each chain. It then maps points from the shorter chain to the longer
    chain and calculates the average position for each pair, resulting in the interpolated chain. The first and last
    points are directly mapped. Other points are mapped to the nearest point on the longer chain. The interpolated chain
    has at least two points, even when the specified point distance is greater than the chain length.
    :param chain_a: The first polygonal chain as sequence of at least two points.
    :type chain_a: sequence of array_like
    :param chain_b: The second polygonal chain as sequence of at least two points.
    :type chain_b: sequence of array_like
    :param point_distance: The positive approximate distance between points on each chain.
    :type point_distance: float
    :return: A 2D array of points, representing the interpolated polygonal chain.
    """
    points_a = _set_points(chain_a, point_distance)
    points_c = _set_points(chain_c, point_distance)

    # Make chain_a the smaller chain.
    if len(points_a) > len(points_c):
        points_a, points_c = points_c, points_a

    interpolation = [chain_b[0]]

    # Map points between first and last point.
    for point_a in points_a[1:-1]:
        index_point_c = np.argmin(np.linalg.norm(points_c - point_a, axis=1))
        point_c = points_c[index_point_c]
        interpolation.append((point_a + point_c) / 2)

    interpolation.append(chain_b[-1])

    return np.array(interpolation)


def _interpolate_slices_in_z(slice_a, slice_b, slice_c, point_distance, dcm_path):
    """
    Interpolates `slice_b` by partially averaging over `slice_a` and `slice_c`.

    This function interpolates `slice_b`, by filtering the points that belong to an anatomical structure which is
    present in `slice_a`, `slice_b` and `slice_c`. The points that lie between two consecutive filtered structures are
    interpolated with the points from `slice_a` and `slice_c`. For this, the points between the two structures are
    extracted from `slice_a` and `slice_c`, including the two points on the structures themselves, resulting in two
    polygonal chains that connect the two structures. The function then creates an interpolated polygonal chain by
    setting points at a distance of approximately `point_distance` along each chain. It then maps points from the
    shorter chain to the longer chain and calculates the average position for each pair, resulting in the interpolated
    chain. The first and last points are directly mapped. Other points are mapped to the nearest point on the longer
    chain. The interpolated chain has at least two points, even when the specified point distance is greater than the
    chain length.

    The function guarantees, that the resulting slice is closed.
    :param slice_a: The first slice that is used for the interpolation of `slice_b` as a 2D array of point.
    :type slice_a: np.ndarray
    :param slice_b: The slice that is being interpolated as a 2D array of point.
    :type slice_b: np.ndarray
    :param slice_c: The second slice that is used for the interpolation of `slice_b` as a 2D array of point.
    :type slice_c: np.ndarray
    :param point_distance: The positive approximate distance between the new points which are set on the slice.
    :type point_distance: float
    :param dcm_path: The path to the rt struct.
    :type dcm_path: str
    :return: The interpolated closed slice.
    """
    _adjust_orientation(slice_a, slice_b, slice_c)

    labels_a, labels_b, labels_c = (_extract_labels(slice_a, dcm_path), _extract_labels(slice_b, dcm_path),
                                    _extract_labels(slice_c, dcm_path))
    label_intersection = _intersection(labels_a, labels_b, labels_c)

    contour_interpolated = []
    slice_a_open, slice_b_open, slice_c_open = _open_contour(slice_a), _open_contour(slice_b), _open_contour(slice_c)
    for start_label, end_label in zip(label_intersection, label_intersection[1:] + label_intersection[:1]):
        # Add points on anatomical structure to interpolated slice.
        points_start = _extract_points(slice_b_open, start_label, dcm_path)
        contour_interpolated.extend(points_start)

        # Interpolate points between two anatomical structures.
        points_between_start_end_a = _extract_points_between(slice_a_open, start_label, end_label, dcm_path)
        points_between_start_end_b = _extract_points_between(slice_b_open, start_label, end_label, dcm_path)
        points_between_start_end_c = _extract_points_between(slice_c_open, start_label, end_label, dcm_path)

        points_between_start_end_interpolated = _interpolate_chain(
            points_between_start_end_a, points_between_start_end_b, points_between_start_end_c, point_distance)
        contour_interpolated.extend(points_between_start_end_interpolated)

    contour_interpolated = np.array(contour_interpolated)
    contour_interpolated = _close_contour(contour_interpolated)

    return contour_interpolated


def interpolate_contour_in_z(contour, dcm_path, point_distance=2):
    """
    Interpolate a 3D contour in z-direction by partially averaging over every second slice.

    This function interpolates a closed 3D contour, represented as a 2D array of points. The contour is divided in
    xy-direction in 2D slices. Beginning from the slice with the second lowest z-value, every second slice is
    interpolated partially. This is done by first filtering the points that belong to an anatomical structure which is
    present in the slice above and the slice below as well. The points that lie between two consecutive filtered
    structures are extracted from the slice above and the slice below, including the two points on the structures
    themselves, resulting in two polygonal chains that connect the two structures. The function then creates an
    interpolated polygonal chain by setting points at a distance of approximately `point_distance` along each chain.
    It then maps points from the shorter chain to the longer chain and calculates the average position for each pair,
    resulting in the interpolated chain. The first and last points are directly mapped. Other points are mapped to the
    nearest point on the longer chain. The interpolated chain has at least two points, even when the specified point
    distance is greater than the chain length.
    :param contour: The closed 3D contour as a 2D array of points.
    :type contour: np.ndarray
    :param dcm_path: The path to the rt struct.
    :type dcm_path: str
    :param point_distance: The positive approximate distance between two interpolated points.
    :type point_distance: float
    :return: The closed interpolated contour as a 2D array of points.
    """
    if point_distance <= 0:
        return contour

    contour_interpolated = []
    z_values = np.unique(contour[:, 2])
    for i in range(0, len(z_values) - 2, 2):
        slice_below = contour[contour[:, 2] == z_values[i]]
        slice_mid = contour[contour[:, 2] == z_values[i + 1]]
        slice_above = contour[contour[:, 2] == z_values[i + 2]]

        slice_mid_interpolated = _interpolate_slices_in_z(slice_below, slice_mid, slice_above, point_distance, dcm_path)

        contour_interpolated.extend(slice_below)
        contour_interpolated.extend(slice_mid_interpolated)

    # The slices are processed in steps of 2. Therefore, the last 1-2 slices might not have been processed yet.
    number_remaining_z_values = 1 + (len(z_values) - 1) % 2
    remaining_z_values = z_values[-number_remaining_z_values:]
    remaining_contours = contour[np.isin(contour[:, 2], remaining_z_values)]
    contour_interpolated.extend(remaining_contours)

    return np.array(contour_interpolated)


def clip_corners(contour, corners, radius=2, angle=np.pi/2):
    """
    Clip the corners of a 3D contour that have an angle smaller than specified.

    This function processes each 2D xy-slice of the 3D contour separately. For each corner point that is part of the
    contour, it checks if the angle formed at that corner is smaller than the specified angle. If so, it removes the
    portion of the contour within a circle of the given radius centered at the corner, and replaces it with the circle's
    intersection points with the contour.

    A new contour with the corners clipped is returned. The original contour is not modified.
    :param contour: The closed 3D contour as a 2D array of points.
    :type contour: np.ndarray
    :param corners: The corner points to consider for clipping, as a 2D array of points.
    :type corners: np.ndarray
    :param radius: The radius of the circle used for clipping.
    :type radius: float
    :param angle: The threshold angle in radians. Corners with angles smaller than this will be clipped.
    :type angle: float
    :return: A new 3D contour with the specified corners clipped. The original contour remains unchanged.
    :rtype: np.ndarray
    """
    clipped_contour = []
    for z in np.unique(contour[:, 2]):
        contour_slice = contour[contour[:, 2] == z]
        for corner in corners[corners[:, 2] == z]:
            corner_index = _find_index(corner, contour_slice)
            if corner_index is not None:
                left_neighbor, right_neighbor = _find_nearest_distinct_neighbors(contour_slice, corner_index)
                if (left_neighbor is not None and right_neighbor is not None
                        and _angle(left_neighbor, corner, right_neighbor) < angle):
                    intersection, start_intersection, end_intersection = _get_contour_segment_within_circle(
                        contour_slice, corner_index, radius)
                    substitute_points = np.array([intersection[0], intersection[-1]])
                    contour_slice = _replace_section_wrap_around(contour_slice, start_intersection, end_intersection,
                                                                 substitute_points)
                    contour_slice = _close_contour(contour_slice)  # _delete_section_wrap_around can open contour_slice
        clipped_contour += contour_slice.tolist()
    return np.array(clipped_contour)


def interpolate_contour_in_xy(contour, corners=None, radius=2, num_interpolation_points=10):
    """
    Interpolate specified points of a closed 3D contour in the xy-plane.

    This function processes each 2D slice of the 3D contour independently. For each specified corner point that is part
    of the contour, it draws a circle of the given radius around the corner. The function then follows the contour in
    both directions from the corner until it intersects the circle. The points found on the contour and the two
    intersection points are used to define a Bézier curve. The original points within the circle are removed, and
    `num_interpolation_points` points along the Bézier curve are added to the contour.

    A new contour with the specified points interpolated is returned. The original contour is not modified.
    :param contour: The closed 3D contour as a 2D array of points.
    :type contour: np.ndarray
    :param corners: The points to interpolate over as a 2D array of points. Defaults to `contour`.
    :type corners: np.ndarray
    :param radius: The radius of the circle. Must be > 0. Defaults to 2.
    :type radius: float
    :param num_interpolation_points: The number of points to set on the Bézier curve. Defaults to 10.
    :type num_interpolation_points: int
    :return: A new contour with interpolated corner points as a 2D array of points.
    :rtype: np.ndarray
    """
    if corners is None:
        corners = contour

    interpolated_contour = []
    for z in np.unique(contour[:, 2]):
        contour_slice = contour[contour[:, 2] == z]
        for corner in corners[corners[:, 2] == z]:
            corner_index = _find_index(corner, contour_slice)
            if corner_index is not None:
                intersection, start_intersection, end_intersection = _get_contour_segment_within_circle(
                    contour_slice, corner_index, radius)
                # Multiple occurrences of a point pull the Bézier curve in this direction.
                intersection = _unique_array(intersection)
                interpolation = _interpolate_points(intersection, num_interpolation_points)
                interpolation[:, 2] = z  # revert rounding errors
                contour_slice = _replace_section_wrap_around(contour_slice, start_intersection, end_intersection,
                                                             interpolation)
                contour_slice = _close_contour(contour_slice)  # _delete_section_wrap_around can open contour_slice
        interpolated_contour += contour_slice.tolist()
    return np.array(interpolated_contour)


def extract_structure_endpoints(contour, dcm_path):
    """
    Extract endpoints of anatomical structures in the contour for each axial slice.

    The contour is represented as a closed polygonal chain in each axial slice. This function identifies the first
    and last points on the contour that lie on each anatomical structure defined in a DICOM file within the slice.

    :param contour: The contour to extract the points from.
    :type contour: np.ndarray
    :param dcm_path: The path to the RT-struct of the DICOM file.
    :type dcm_path: str
    :return: The endpoints of the anatomical structures.
    :rtype: np.ndarray
    """
    corners = []
    for z in np.unique(contour[:, 2]):
        contour_slice = contour[contour[:, 2] == z]
        contour_slice_open = _open_contour(contour_slice)
        labels = _extract_labels(contour_slice, dcm_path)

        for label in labels:
            structure = _extract_points(contour_slice_open, label, dcm_path)
            corners.append(structure[0])
            corners.append(structure[-1])
    return np.array(corners)
