from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
import os
from os.path import join
import numpy as np

#####
# Written by Daniel Luckey
#####

CT_IMAGE_STORAGE_SOP_CLASS_UID = '1.2.840.10008.5.1.4.1.1.2'
RT_STRUCTURE_SET_STORAGE_SOP_CLASS_UID = '1.2.840.10008.5.1.4.1.1.481.3'
DICOM_EXTENSION = '.dcm'

_ROI_GENERATION_ALGORITHM = 'AUTOMATIC'
_ROI_DISPLAY_COLOR = [255, 0, 0]  # red
_CONTOUR_GEOMETRIC_TYPE = 'CLOSED_PLANAR'
_RT_ROI_INTERPRETED_TYPE = 'CTV'
_ROI_INTERPRETER = ''


def _load_dicom(input_path):
    """
    Extracts the RT struct and the SOP instance UIDs of the CT images.

    :param str input_path: the path to a folder containing CT images and exactly one RT struct in DICOM format.
    :return: the RT struct, the path to the RT struct and the SOP instance UIDs as dictionary from z value to SOP
    instance UID
    """
    rt_struct = None
    rt_struct_path = None
    sop_instance_uids = {}
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith(DICOM_EXTENSION):
                file_path = join(root, file)
                ds = dcmread(file_path)
                sop_class_uid = ds.SOPClassUID
                if sop_class_uid == CT_IMAGE_STORAGE_SOP_CLASS_UID:
                    # ds is a ct image
                    z_coordinate = ds.ImagePositionPatient[2]
                    sop_instance_uids[z_coordinate] = ds.SOPInstanceUID
                elif sop_class_uid == RT_STRUCTURE_SET_STORAGE_SOP_CLASS_UID:
                    # ds is a rt struct
                    rt_struct = ds
                    rt_struct_path = file_path
    return rt_struct, rt_struct_path, sop_instance_uids


def _add_structure_set_roi(rt_struct, roi_name, roi_generation_algorithm):
    structure_set_roi = Dataset()
    structure_set_roi.ROINumber = max([structure_set_roi.ROINumber
                                       for structure_set_roi in rt_struct.StructureSetROISequence]) + 1
    structure_set_roi.ReferencedFrameOfReferenceUID = rt_struct.FrameOfReferenceUID
    structure_set_roi.ROIName = roi_name
    structure_set_roi.ROIGenerationAlgorithm = roi_generation_algorithm
    rt_struct.StructureSetROISequence.append(structure_set_roi)

    return structure_set_roi.ROINumber


def _add_roi_contour(rt_struct, roi_contour_points, sop_instance_uids, roi_display_color, contour_geometric_type,
                     referenced_roi_number):
    roi_contour = Dataset() # the 3D contour
    roi_contour.ROIDisplayColor = roi_display_color
    roi_contour.ReferencedROINumber = referenced_roi_number
    roi_contour.ContourSequence = Sequence()
    for i, z in enumerate(np.sort(np.unique(roi_contour_points[:, 2]))):
        contour = Dataset() # the 2D contour
        contour.ContourGeometricType = contour_geometric_type
        contour_points = roi_contour_points[roi_contour_points[:, 2] == z]
        contour.NumberOfContourPoints = len(contour_points)
        contour.ContourNumber = i
        contour.ContourData = contour_points.flatten().tolist()

        contour_image = Dataset() # the image containing the contour
        contour_image.ReferencedSOPClassUID = CT_IMAGE_STORAGE_SOP_CLASS_UID
        contour_image.ReferencedSOPInstanceUID = sop_instance_uids[z]
        contour.ContourImageSequence = Sequence()
        contour.ContourImageSequence.append(contour_image)

        roi_contour.ContourSequence.append(contour)
    rt_struct.ROIContourSequence.append(roi_contour)


def _add_rt_roi_observation(rt_struct, referenced_roi_number, roi_observation_label, rt_roi_interpreted_type,
                            roi_interpreter):
    rt_roi_observation = Dataset()
    rt_roi_observation.ObservationNumber = max([rt_roi_observation.ObservationNumber
                                                 for rt_roi_observation in rt_struct.RTROIObservationsSequence]) + 1
    rt_roi_observation.ReferencedROINumber = referenced_roi_number
    rt_roi_observation.ROIObservationLabel = roi_observation_label
    rt_roi_observation.RTROIInterpretedType = rt_roi_interpreted_type
    rt_roi_observation.ROIInterpreter = roi_interpreter

    rt_struct.RTROIObservationsSequence.append(rt_roi_observation)


def add_contour(input_path, contour_points, roi_name, roi_generation_algorithm=_ROI_GENERATION_ALGORITHM,
                roi_display_color=_ROI_DISPLAY_COLOR, contour_geometric_type=_CONTOUR_GEOMETRIC_TYPE,
                rt_roi_interpreted_type=_RT_ROI_INTERPRETED_TYPE, roi_interpreter=_ROI_INTERPRETER):
    """
    Adds a new contour to the RT struct.

    This function takes an input path to a folder containing CT images and exactly one RT struct in DICOM format, along
    with a set of 3D contour points that describe a closed polygon for specific z values. It adds all polygons to the
    RT struct.

    :param str input_path: path to a folder containing CT images and exactly one RT struct in DICOM format
    :param np.ndarray contour_points: 3D points describing closed polygons for specific z values
    :param str roi_name: the name of the contour
    :param str roi_generation_algorithm: the algorithm the contour was generated with
    :param list[int] roi_display_color: the color the contour is displayed with as RGB triplet
    :param str contour_geometric_type: the type of the contour
    :param str rt_roi_interpreted_type: the type of the contour
    :param str roi_interpreter: the name of the person performing the interpretation of the contour
    :return: None
    """

    rt_struct, rt_struct_path, sop_instance_uids = _load_dicom(input_path)

    roi_number = _add_structure_set_roi(rt_struct, roi_name, roi_generation_algorithm)
    _add_roi_contour(rt_struct, contour_points, sop_instance_uids, roi_display_color, contour_geometric_type, roi_number)
    _add_rt_roi_observation(rt_struct, roi_number, roi_name, rt_roi_interpreted_type, roi_interpreter)

    rt_struct.save_as(rt_struct_path, write_like_original=False)


def extract_rt_ct_paths(input_path):
    """
    Extract the path to the RT-struct and a CT-image from a path to a CT-scan in DICOM format.

    :param input_path: The path to the CT-scan.
    :type input_path: str
    :return: The paths to the RT-struct and a CT-image.
    :rtype: (str, str)
    """
    rt_path, ct_path = None, None
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith(DICOM_EXTENSION):
                file_path = join(root, file)
                ds = dcmread(file_path)
                sop_class_uid = ds.SOPClassUID
                if sop_class_uid == CT_IMAGE_STORAGE_SOP_CLASS_UID:
                    ct_path = file_path
                elif sop_class_uid == RT_STRUCTURE_SET_STORAGE_SOP_CLASS_UID:
                    rt_path = file_path
    return rt_path, ct_path


def get_slice_thickness(ct_path):
    """
    Retrieve the slice thickness, which is the distance between two axial slices.

    :param ct_path:  The path to a single CT-image.
    :type ct_path: str
    :return: The slice thickness.
    :rtype: float
    """
    ds = dcmread(ct_path)
    return float(ds.SliceThickness)