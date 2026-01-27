import numpy as np
from numpyradiomics.mod_shape_3d import shape_3d, shape_3d_units
from numpyradiomics.mod_shape_2d import shape_2d, shape_2d_units
from numpyradiomics.mod_ski_shape_3d import ski_shape_3d, ski_shape_3d_units


def shape(mask:np.ndarray, spacing=(1.0, 1.0, 1.0), extend=True):
    """
    Compute shape descriptors for a binary mask in 2D or 3D.
    
    This wrapper automatically detects the dimensionality of the input mask and 
    dispatches the calculation to the appropriate 2D or 3D sub-module.

    Parameters
    ----------
    mask : np.ndarray
        A binary mask array where non-zero pixels represent the Region of Interest (ROI).
        - For 2D: Format (Y, X).
        - For 3D: Format (Z, Y, X).
    spacing : tuple of float, optional
        Pixel or voxel spacing in physical base units (e.g., mm).
        - For 2D: (y_spacing, x_spacing).
        - For 3D: (z_spacing, y_spacing, x_spacing).
        Defaults to unit spacing (1.0, 1.0, 1.0).
    extend : bool, optional
        If True (default), and the input is 3D, the returned dictionary includes 
        additional shape metrics derived from `skimage.measure.regionprops` 
        (e.g., Convex Hull Volume, Solidity, Moments of Inertia) that may not be 
        part of the standard PyRadiomics set. Ignored for 2D inputs.

    Returns
    -------
    dict
        A dictionary containing shape feature names (keys) and their calculated 
        values (float). 

        For 2D data the following are returned:
        - **MeshSurface**: Area of the ROI defined by the mesh (marching squares).
        - **PixelSurface**: Area defined by the count of non-zero pixels.
        - **Perimeter**: Perimeter length of the ROI mesh.
        - **PerimeterSurfaceRatio**: Ratio of Perimeter to MeshSurface.
        - **Sphericity**: Measure of roundness (1.0 is a perfect circle).
        - **SphericalDisproportion**: Inverse of sphericity.
        - **MaximumDiameter**: Largest Euclidean distance between contour vertices.
        - **MajorAxisLength**: Principal axis length derived from image moments.
        - **MinorAxisLength**: Secondary axis length derived from image moments.
        - **Elongation**: Ratio of Minor to Major axis length.

        For 3D data the following are returned:
        - **MeshVolume**: Volume calculated from the surface mesh (Divergence theorem).
        - **VoxelVolume**: Volume calculated by counting voxels multiplied by voxel spacing.
        - **SurfaceArea**: Total area of the surface mesh.
        - **SurfaceVolumeRatio**: Ratio of Surface Area to Volume.
        - **Sphericity**: Measure of roundness (0 to 1), where 1 is a perfect sphere.
        - **Maximum3DDiameter**: Largest Euclidean distance between vertices on the convex hull.
        - **Maximum2DDiameterSlice**: Maximum diameter in the axial plane (X-Y).
        - **Maximum2DDiameterColumn**: Maximum diameter in the coronal plane (Z-X).
        - **Maximum2DDiameterRow**: Maximum diameter in the sagittal plane (Z-Y).
        - **MajorAxisLength**: Length of the largest principal axis (PCA).
        - **MinorAxisLength**: Length of the second largest principal axis (PCA).
        - **LeastAxisLength**: Length of the smallest principal axis (PCA).
        - **Elongation**: Ratio of major to minor axis components (sqrt(lambda_minor / lambda_major)).
        - **Flatness**: Ratio of major to least axis components (sqrt(lambda_least / lambda_major)).

        
        If `extend=True` (3D only), additional keys include:
        - **Solidity**: Ratio of region volume to convex hull volume.
        - **Extent**: Ratio of region volume to bounding box volume.
        - **MaximumDepth**: Radius of the largest inscribed sphere (Chebyshev radius).
        - **LongestCaliperDiameter**: Maximum Feret diameter.
        - **FractionalAnisotropyOfInertia**: Measures how elongated/flat the shape is (0 to 1).
        - **MomentsOfInertia**: First, Second, and Third moments along principal axes.

    
    Raises
    ------
    ValueError
        If `mask` is not 2D or 3D.

    Examples
    --------
    >>> from numpyradiomics import dro
    >>> # Create a synthetic 3D cube (e.g., 6x6x6 pixels centered in 10x10x10)
    >>> spacing = (1.0, 1.0, 1.0)
    >>> mask = dro.cuboid(radii_mm=(10.0, 5.0, 2.5), spacing=spacing)
    
    >>> # Calculate features
    >>> features = shape(mask, spacing, extend=True)
    
    >>> # Access standard and extended metrics
    >>> print(f"Volume: {features['VoxelVolume']}")
    Volume: 216.0
    >>> print(f"Solidity: {features.get('Solidity', 'N/A')}")
    Solidity: 1.0
    """
    if np.ndim(mask) == 2:
        return shape_2d(mask, spacing)
    if np.ndim(mask) == 3:
        shape = shape_3d(mask, spacing)
        if extend:
            shape = shape | ski_shape_3d(mask, spacing)
        return shape
    

def shape_units(dim:int, voxel_unit='mm'):
    """Units of returned shape metrics

    Args:
        dim (int): nr of dimensions (2 or 3)
        voxel_unit (str, optional): unit of voxel length

    Returns:
        dict: features and their units
    """
    if dim == 2:
        return shape_2d_units(voxel_unit)
    if dim == 3:
        return shape_3d_units(voxel_unit) | ski_shape_3d_units(voxel_unit)