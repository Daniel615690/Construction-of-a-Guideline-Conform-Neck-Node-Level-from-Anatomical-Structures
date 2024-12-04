# Construction-of-a-Guideline-Conform-Neck-Node-Level-from-Anatomical-Structures

This repository provides the code to generate guideline-conform neck node levels from CT scans in DICOM format, 
following the guidelines by [Grégoire et al. (2013)](https://doi.org/10.1016/j.radonc.2013.10.010).

A neck node level is a distinct, connected, three-dimensional volume defined relative to bordering anatomical 
structures. In this repository, a neck node level is represented for each xy-plane by its boundary, which is described 
as a closed polygonal chain.

## Key components
* **neck_node_level.py**: The main script containing the **AnatomicalStructure** and **NeckNodeLevel** classes.
* **AnatomicalStructure**: This class extracts the boundary points of an anatomical structure from a DICOM file and 
  provides methods to extract specific points, such as the rightmost point.
* **NeckNodeLevel**: This abstract class defines the method **_initialize_contour** for constructing an initial neck 
  node level. It includes methods for post-processing the initial neck node level. To create a new neck node level, 
  extend this class and implement **_initialize_contour**, which can be achieved by extracting specific points from 
  bordering anatomical structures and combining them to form a closed polygonal chain for each z-value using the 
  **AnatomicalStructure** class.

## Usage
```python
from neck_node_level import NeckNodeLevel4aLeft

input_path = './HN_P001_pre'
output_path = './HN_P001_pre'

# Create an initial neck node level
neck_node_level = NeckNodeLevel4aLeft(input_path)

# Post-process the neck node level
neck_node_level.remove_self_intersections()
neck_node_level.interpolate_in_z()
neck_node_level.clip_corners(neck_node_level.extract_structure_endpoints())
neck_node_level.interpolate_in_xy()
neck_node_level.remove_intersections()

# Plot the neck node level
neck_node_level.plot()

# Save the neck node level to a DICOM file
neck_node_level.save(output_path)
```

## Future work
Currently, this repository implements the left IVa neck node level. Future work should involve extending the 
**NeckNodeLevel** class to construct the remaining neck node levels (I-X).

## References
* [Guidelines](https://doi.org/10.1016/j.radonc.2013.10.010)
