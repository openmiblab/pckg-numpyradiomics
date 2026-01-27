import numpy as np

def _create_grid(radii_mm, spacing, padding_mm):
    """
    Helper to create centered physical coordinate grids.
    Ensures 0.0 is exactly at the center of the array.
    """
    radii_mm = np.array(radii_mm, dtype=np.float64)
    spacing = np.array(spacing, dtype=np.float64)
    
    # Field of View limits (radii + padding)
    limits = radii_mm + padding_mm
    
    # Create coordinate arrays (centered at 0)
    # Using arange ensures steps are exactly 'spacing'
    z_mm = np.arange(-limits[0], limits[0], spacing[0])
    y_mm = np.arange(-limits[1], limits[1], spacing[1])
    x_mm = np.arange(-limits[2], limits[2], spacing[2])
    
    # Create 3D Meshgrid
    # indexing='ij' ensures Matrix/Image ordering (Z, Y, X)
    Z, Y, X = np.meshgrid(z_mm, y_mm, x_mm, indexing='ij')
    
    return Z, Y, X

def cuboid(radii_mm=(30.0, 10.0, 2.5), spacing=(1.0, 1.0, 1.0), padding_mm=5.0):
    """
    Creates a binary mask of a solid cuboid.
    
    Args:
        radii_mm (tuple): Physical half-lengths (Rx, Ry, Rz) in mm.
                          Total size will be 2*Rx, 2*Ry, 2*Rz.
        spacing (tuple): Voxel spacing (Sz, Sy, Sx) in mm.
        padding_mm (float): Padding around the object in mm.
        
    Returns:
        image (np.ndarray): Dummy intensity image (value 100 inside).
        mask (np.ndarray): Binary mask (uint8).

    Example:
        >>> from numpyradiomics.dro import cuboid
        >>> # Create a 10x10x10mm cube (5mm radii) with 1mm spacing
        >>> image, mask = cuboid(radii_mm=(5, 5, 5), spacing=(1.0, 1.0, 1.0))
        >>> print(mask.shape)
        (20, 20, 20)
    """
    Z, Y, X = _create_grid(radii_mm, spacing, padding_mm)
    
    # Cuboid Logic: Intersection of linear bounds
    # Using half-open intervals [-R, R) prevents "fencepost" errors where
    # centered grids include one too many pixels for even integer dimensions.
    mask = (
        (Z >= -radii_mm[0]) & (Z < radii_mm[0]) &
        (Y >= -radii_mm[1]) & (Y < radii_mm[1]) &
        (X >= -radii_mm[2]) & (X < radii_mm[2])
    ).astype(np.uint8)
    
    image = mask.astype(np.float32) * 100.0
    return image, mask

def ellipsoid(radii_mm=(30.0, 10.0, 2.5), spacing=(1.0, 1.0, 1.0), padding_mm=5.0):
    """
    Creates a binary mask of a solid ellipsoid.
    
    Args:
        radii_mm (tuple): Physical radii (Rx, Ry, Rz) in mm.
        spacing (tuple): Voxel spacing (Sz, Sy, Sx) in mm.
        padding_mm (float): Padding around the object in mm.
        
    Returns:
        image (np.ndarray): Dummy intensity image (value 100 inside).
        mask (np.ndarray): Binary mask (uint8).

    Example:
        >>> from numpyradiomics.dro import ellipsoid
        >>> # Create an isotropic sphere (radius 10mm) with 0.5mm spacing
        >>> image, mask = ellipsoid(radii_mm=(10, 10, 10), spacing=(0.5, 0.5, 0.5))
        >>> print(mask.sum())  # Number of voxels in the sphere
        33489
    """
    Z, Y, X = _create_grid(radii_mm, spacing, padding_mm)
    
    # Ellipsoid Logic: Sum of squared normalized distances <= 1
    normalized_dist_sq = (
        (Z / radii_mm[0])**2 + 
        (Y / radii_mm[1])**2 + 
        (X / radii_mm[2])**2
    )
    mask = (normalized_dist_sq <= 1.0).astype(np.uint8)
    
    image = mask.astype(np.float32) * 100.0
    return image, mask

def noisy_ellipsoid(radii_mm=(30.0, 10.0, 2.5), spacing=(1.0, 1.0, 1.0), padding_mm=5.0, intensity_range=(0, 100)):
    """
    Creates a binary mask of an ellipsoid filled with random noise.
    Useful for testing texture features.
    
    Args:
        radii_mm (tuple): Physical radii (Rx, Ry, Rz) in mm.
        spacing (tuple): Voxel spacing (Sz, Sy, Sx) in mm.
        padding_mm (float): Padding around the object in mm.
        intensity_range (tuple): (min, max) intensity values for noise.
        
    Returns:
        image (np.ndarray): Noisy image (float32).
        mask (np.ndarray): Binary mask (uint8).

    Example:
        >>> from numpyradiomics.dro import noisy_ellipsoid
        >>> import numpy as np
        >>> # Create a noisy ellipsoid for texture analysis
        >>> img, mask = noisy_ellipsoid(radii_mm=(20, 10, 5))
        >>> print(f"Mean Intensity in ROI: {np.mean(img[mask==1]):.2f}")
        Mean Intensity in ROI: 49.87
    """
    Z, Y, X = _create_grid(radii_mm, spacing, padding_mm)
    
    # Ellipsoid Logic
    normalized_dist_sq = (
        (Z / radii_mm[0])**2 + 
        (Y / radii_mm[1])**2 + 
        (X / radii_mm[2])**2
    )
    mask = (normalized_dist_sq <= 1.0).astype(np.uint8)
    
    # Generate Noise
    np.random.seed(42)
    image = np.random.uniform(intensity_range[0], intensity_range[1], mask.shape).astype(np.float32)
    
    # Apply Mask (Background = 0)
    image[mask == 0] = 0
    
    return image, mask