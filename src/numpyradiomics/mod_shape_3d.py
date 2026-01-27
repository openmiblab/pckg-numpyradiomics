import logging

import numpy as np
from skimage import measure
from scipy.spatial import ConvexHull, distance_matrix


def shape_3d(input_mask, spacing=(1.0, 1.0, 1.0)):
    """
    Returns the 3D shape features following PyRadiomics conventions.

    This function calculates geometric shape descriptors for a 3D Region of Interest (ROI).
    It computes features based on both the mesh representation (Surface Area, Mesh Volume) 
    and the voxel representation (Voxel Volume, Axis Lengths).

    Parameters
    ----------
    input_mask : np.ndarray
        Binary mask with shape (Z, Y, X). Non-zero elements are considered the ROI.
    spacing : tuple of float, optional
        Voxel spacing (dz, dy, dx) in physical base units (e.g., mm).
        Defaults to (1.0, 1.0, 1.0).

    Returns
    -------
    dict
        Dictionary of calculated features in base units.
        
        The following metrics are returned:
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

    Raises
    ------
    AssertionError
        If `input_mask` is not 3-dimensional.

    Examples
    --------
    >>> from numpyradiomics import dro
    >>> spacing = (1.0, 1.0, 1.0)
    >>> # Create a synthetic cuboid (Radii: 10, 5, 2.5 -> Dimensions: 20x10x5 mm)
    >>> mask = dro.cuboid(radii_mm=(10.0, 5.0, 2.5), spacing=spacing)
    
    >>> # Calculate features
    >>> feats = shape_3d(mask, spacing)
    
    >>> print(f"Mesh Volume: {feats['MeshVolume']:.1f}")
    Mesh Volume: 1000.0
    >>> print(f"Sphericity: {feats['Sphericity']:.2f}")
    Sphericity: 0.72
    """
    assert input_mask.ndim == 3, "Shape features are only available in 3D."
    
    # 1. Zero-pad mask to ensure the mesh is closed (water-tight) at boundaries
    mask = np.pad(input_mask, pad_width=1, mode='constant', constant_values=0)
    
    # 2. Extract Surface Mesh using Marching Cubes
    #    If spacing is provided, 'verts' are returned in physical coordinates.
    #    skimage uses the same axis order as input: (z, y, x).
    verts, faces, normals, _ = measure.marching_cubes(mask, spacing=spacing)
    
    # --- Feature: Surface Area ---
    def mesh_surface_area(vertices, faces):
        # Vectorized cross product for speed
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        # Area is sum of half the magnitude of cross products
        return 0.5 * np.sum(np.linalg.norm(cross, axis=1))

    surface_area = mesh_surface_area(verts, faces)
    
    # --- Feature: Mesh Volume ---
    def mesh_volume(vertices, faces):
        # Divergence theorem
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        return abs(np.sum(np.einsum('ij,ij->i', v0, np.cross(v1, v2))) / 6.0)

    volume = mesh_volume(verts, faces)
    
    # --- Helper: Max Diameter ---
    def max_diameter(points):
        if len(points) <= 1: return 0.0
        # Optimization: The farthest points must be on the Convex Hull
        try:
            hull = ConvexHull(points)
            candidates = points[hull.vertices]
        except Exception:
            # Fallback for degenerate shapes (lines/planes)
            candidates = points
        
        # Calculate pairwise distance matrix
        dist_mat = distance_matrix(candidates, candidates)
        return np.max(dist_mat)

    # --- Feature: Maximum 3D Diameter ---
    max_3D_diameter = max_diameter(verts)
    
    # --- Feature: Maximum 2D Diameters ---
    # Since input is (Z, Y, X), the columns of 'verts' are 0=Z, 1=Y, 2=X.
    # PyRadiomics definitions:
    # - Slice (Axial):     Plane X-Y (indices 1, 2)
    # - Column (Coronal):  Plane Z-X (indices 0, 2)
    # - Row (Sagittal):    Plane Z-Y (indices 0, 1)

    max_2D_slice = max_diameter(verts[:, [1, 2]])  # Axial
    max_2D_column = max_diameter(verts[:, [0, 2]]) # Coronal
    max_2D_row = max_diameter(verts[:, [0, 1]])    # Sagittal

    # --- Feature: Axis Lengths (PCA) ---
    # PyRadiomics uses the SOLID volume (all voxel centers), not the hollow mesh.
    # Get indices of all non-zero voxels
    z, y, x = np.where(input_mask != 0)
    
    # Convert to physical coordinates using spacing
    # Stack columns to get N x 3 array of physical points
    voxel_coords = np.vstack([z * spacing[0], y * spacing[1], x * spacing[2]]).T
    
    if len(voxel_coords) > 1:
        # Covariance matrix of the physical coordinates
        cov_matrix = np.cov(voxel_coords.T)
        # Eigenvalues represent variance along principal axes
        eigenvalues, _ = np.linalg.eigh(cov_matrix)
        eigenvalues = np.sort(eigenvalues) # Sort ascending [small, medium, large]
    else:
        eigenvalues = np.array([0, 0, 0])

    # PyRadiomics Formula: Length = 4 * sqrt(eigenvalue)
    # Ensure non-negative (float precision errors can cause -1e-18)
    eig_safe = np.maximum(eigenvalues, 0)
    
    least_axis_length = 4 * np.sqrt(eig_safe[0])
    minor_axis_length = 4 * np.sqrt(eig_safe[1])
    major_axis_length = 4 * np.sqrt(eig_safe[2])
    
    # --- Derived Parameters ---
    voxel_volume = len(voxel_coords) * np.prod(spacing)
    
    # Avoid division by zero
    surface_volume_ratio = surface_area / volume if volume > 0 else 0
    sphericity = (36 * np.pi * volume**2) ** (1.0 / 3.0) / surface_area if surface_area > 0 else 0
    elongation = np.sqrt(eig_safe[1] / eig_safe[2]) if eig_safe[2] > 0 else 0
    flatness = np.sqrt(eig_safe[0] / eig_safe[2]) if eig_safe[2] > 0 else 0

    return {
        "SurfaceArea": surface_area,
        "MeshVolume": volume,
        "VoxelVolume": voxel_volume,
        "Maximum2DDiameterSlice": max_2D_slice,
        "Maximum2DDiameterColumn": max_2D_column,
        "Maximum2DDiameterRow": max_2D_row,
        "Maximum3DDiameter": max_3D_diameter,
        "SurfaceVolumeRatio": surface_volume_ratio,
        "Sphericity": sphericity,
        "MajorAxisLength": major_axis_length,
        "MinorAxisLength": minor_axis_length,
        "LeastAxisLength": least_axis_length,
        "Elongation": elongation,
        "Flatness": flatness,
    }


def shape_3d_units(base_unit="mm"):
    """
    Return the physical units for the 3D shape metrics.

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
    >>> units = shape_3d_units(base_unit='cm')
    >>> print(units['MeshVolume'])
    'cm^3'
    >>> print(units['SurfaceArea'])
    'cm^2'
    """
    return {
        "Elongation": "",                         # Dimensionless (ratio)
        "Flatness": "",                           # Dimensionless (ratio)
        "LeastAxisLength": base_unit,             # Length
        "MajorAxisLength": base_unit,             # Length
        "Maximum2DDiameterColumn": base_unit,     # Length
        "Maximum2DDiameterRow": base_unit,        # Length
        "Maximum2DDiameterSlice": base_unit,      # Length
        "Maximum3DDiameter": base_unit,           # Length
        "MeshVolume": f"{base_unit}^3",           # Volume
        "MinorAxisLength": base_unit,             # Length
        "Sphericity": "",                         # Dimensionless (ratio)
        "SurfaceArea": f"{base_unit}^2",          # Area
        "SurfaceVolumeRatio": f"{base_unit}^-1",  # Area / Volume (L^2 / L^3 = 1/L)
        "VoxelVolume": f"{base_unit}^3",          # Volume
    }
