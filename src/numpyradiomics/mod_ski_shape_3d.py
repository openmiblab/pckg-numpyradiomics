import numpy as np
import scipy.ndimage as ndi
from skimage import measure


def ski_shape_3d(image, spacing=(1.0, 1.0, 1.0)):
    """
    Computes additional 3D shape metrics using `skimage.measure.regionprops`.
    
    This function acts as an extension to standard PyRadiomics shape features. 
    It calculates metrics that require an isotropic grid representation, such as 
    Convex Hull Volume, Solidity, and Moments of Inertia. It handles anisotropic 
    input by resampling the mask to the smallest spacing dimension before calculation.

    Parameters
    ----------
    image : np.ndarray
        Binary mask array with shape (Z, Y, X).
    spacing : tuple of float, optional
        Voxel spacing (Sz, Sy, Sx) in physical base units (e.g., mm).
        Defaults to (1.0, 1.0, 1.0).

    Returns
    -------
    dict
        A dictionary of shape metrics. Key features include:
        
        - **Solidity**: Ratio of region volume to convex hull volume.
        - **Extent**: Ratio of region volume to bounding box volume.
        - **MaximumDepth**: Radius of the largest inscribed sphere (Chebyshev radius).
        - **LongestCaliperDiameter**: Maximum Feret diameter.
        - **FractionalAnisotropyOfInertia**: Measures how elongated/flat the shape is (0 to 1).
        - **MomentsOfInertia**: First, Second, and Third moments along principal axes.

    Examples
    --------
    >>> from numpyradiomics import dro
    >>> spacing = (1.0, 1.0, 1.0)
    >>> # Create a synthetic cuboid
    >>> mask = dro.cuboid(radii_mm=(10.0, 5.0, 2.5), spacing=spacing)
    
    >>> # Calculate extended features
    >>> ext_feats = ski_shape_3d(mask, spacing)
    
    >>> print(f"Solidity: {ext_feats['Solidity']:.2f}")
    Solidity: 1.00
    >>> print(f"Max Depth: {ext_feats['MaximumDepth']} mm")
    Max Depth: 2.5 mm
    """
    
    # 1. Zero-pad to ensure the ROI is not touching the boundary
    # This ensures Marching Cubes and Distance Transforms work correctly at edges.
    pad_width = 4
    padded_arr = np.pad(image, pad_width=pad_width, mode='constant', constant_values=0)

    # 4. Interpolate to Isotropic Spacing
    # regionprops calculates properties in "pixels". To get accurate physical 
    # metrics (like Solidity, FA), we treat the object as if it were in an isotropic grid.
    spacing = np.array(spacing)
    
    if np.amin(spacing) != np.amax(spacing):
        # Target the smallest spacing dimension to preserve resolution
        iso_spacing = np.amin(spacing)
        # Use order=0 (Nearest Neighbor) for Masks to preserve binary nature
        # order=1 (Linear) would create float values and change the volume.
        iso_arr, _ = _interpolate3d_isotropic(padded_arr, spacing, iso_spacing, order=0)
    else:
        iso_spacing = np.mean(spacing)
        iso_arr = padded_arr

    isotropic_voxel_vol = iso_spacing ** 3

    # 5. Region Props (on Isotropic Mask)
    # Ensure binary integer type for regionprops
    iso_arr = (iso_arr > 0.5).astype(int)
    
    if np.count_nonzero(iso_arr) == 0:
        return {} # Return empty if mask vanished

    # regionprops returns a list, we take the first (and should be only) region
    props = measure.regionprops(iso_arr)[0]

    # --- Derived Metrics ---
    # Moments of Inertia & Fractional Anisotropy (FA)
    # regionprops inertia_tensor_eigvals are \sum (dist_pixels)^2
    # To convert to physical units (mm^2), we multiply by iso_spacing^2
    eigvals = np.array(props['inertia_tensor_eigvals'])
    
    m0, m1, m2 = eigvals
    m_mean = np.mean(eigvals)
    
    # FA formula (0 to 1)
    numerator = np.sqrt((m0 - m_mean)**2 + (m1 - m_mean)**2 + (m2 - m_mean)**2)
    denominator = np.sqrt(m0**2 + m1**2 + m2**2)
    fa = np.sqrt(3/2) * (numerator / denominator) if denominator > 0 else 0

    # Max Depth (Distance Transform)
    # Calculates the distance from the centroid pixel to the nearest zero pixel
    dist_map = ndi.distance_transform_edt(iso_arr)
    max_depth_mm = np.amax(dist_map) * iso_spacing

    return {
        'MaximumDepth': max_depth_mm, # mm
        'BoundingBoxVolume': props['area_bbox'] * isotropic_voxel_vol, # mm^3
        'ConvexHullVolume': props['area_convex'] * isotropic_voxel_vol,# mm^3
        'FirstMomentOfInertia': m0 * (iso_spacing**2),
        'SecondMomentOfInertia': m1 * (iso_spacing**2),
        'ThirdMomentOfInertia': m2 * (iso_spacing**2),
        'MeanMomentOfInertia': m_mean * (iso_spacing**2),
        'FractionalAnisotropyOfInertia': fa, # Dimensionless (0-1)
        'Extent': props['extent'],      # % (Vol / BoundingBoxVol)
        'Solidity': props['solidity'],  # % (Vol / ConvexVol)
    
        # Note: removing features that are a direct function of a single other feature 
        # alread in the list, or differs only in implementation detail (e.g. resolution-dependent)

        # Same as maximum3DDiameter in pyradiomics
        # 'LongestCaliperDiameter': props['feret_diameter_max'] * iso_spacing,
            
        # Same as SurfaceArea - remove
        # 'surface_area': surface_area,                              # mm^2
        
        # Same as VoxelVolume
        # 'voxel_volume': volume,                                    # mm^3
        
        # Difference with VoxelVolume is numerical - resolution dependent: remove
        # 'IsoVoxelVolume': props['area'] * isotropic_voxel_vol,   # mm^3 (Sanity check vs voxel_volume)
        
        # Same as sphericity 
        # 'compactness': compactness,           # % (Relative to Sphere)

        # These (ellipsoid approx) have a fixed relation with PyRadiomics version (statistical shape approx)
        # 'MajorAxisSize': props['axis_major_length'] * iso_spacing, # mm
        # 'LeastAxisSize': props['axis_minor_length'] * iso_spacing,# mm

        #  Direct function of volume
        # 'equivalent_diameter': props['equivalent_diameter'] * iso_spacing, # mm

    }

def _interpolate3d_isotropic(array, spacing, isotropic_spacing=None, order=0):
    """
    Interpolate a 3D array to isotropic spacing.
    
    Args:
        order (int): 0 for Nearest Neighbor (Labels/Masks), 1 for Linear (Images).
    """
    if isotropic_spacing is None:
        isotropic_spacing = np.amin(spacing)

    zoom_factors = [s / isotropic_spacing for s in spacing]
    
    # Use standard scipy zoom. 
    # prefilter=False is faster and sharper for order=0 or 1
    resampled_array = ndi.zoom(array, zoom_factors, order=order, prefilter=False)

    return resampled_array, isotropic_spacing


def ski_shape_3d_units(base_unit='mm'):
    """
    Return the physical units for the extended (skimage) 3D shape metrics.

    Parameters
    ----------
    base_unit : str, optional
        The string representation of the physical length unit (e.g., 'mm').
        Defaults to 'mm'.

    Returns
    -------
    dict
        A dictionary mapping feature names to unit strings.

    Examples
    --------
    >>> units = ski_shape_3d_units('mm')
    >>> print(units['ConvexHullVolume'])
    'mm^3'
    >>> print(units['FractionalAnisotropyOfInertia'])
    ''
    """
    return {
        'BoundingBoxVolume': f"{base_unit}^3", # mm^3
        'ConvexHullVolume': f"{base_unit}^3",# mm^3
        'Extent': '',      # % (Vol / BoundingBoxVol)
        'Solidity': '',  # % (Vol / ConvexVol)
        'MaximumDepth': base_unit, # mm
        'FirstMomentOfInertia': f"{base_unit}^2",
        'SecondMomentOfInertia': f"{base_unit}^2",
        'ThirdMomentOfInertia': f"{base_unit}^2",
        'MeanMomentOfInertia': f"{base_unit}^2",
        'FractionalAnisotropyOfInertia': '', # Dimensionless (0-1)
    }
