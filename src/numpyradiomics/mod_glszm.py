from typing import Optional, Union

import numpy as np
from scipy.ndimage import label

from .mod_preproc import _discretize_image

def glszm(
    image: np.ndarray, 
    mask: np.ndarray, 
    binWidth: float = 25, 
    binCount: Optional[int] = None,
    levels: Optional[int] = None, 
    connectivity: Optional[int] = None
):
    """
    Compute 16 Pyradiomics-style GLSZM (Gray Level Size Zone Matrix) features.

    GLSZM quantifies gray level zones in an image. A gray level zone is defined as a
    number of connected voxels that share the same gray level intensity.
    
    Args:
        image (np.ndarray): 3D image array containing voxel intensities.
        mask (np.ndarray): 3D mask array (same shape as image), where non-zero values indicate the ROI.
        binWidth (float, optional): Width of bins for 'Fixed Bin Width' discretization. Default is 25.
        binCount (int, optional): Number of bins for 'Fixed Bin Count' discretization. 
                                  If specified, overrides binWidth logic. Default is None.
        levels (int, optional): (Deprecated) Number of levels. If None, calculated dynamically.
        connectivity (int, optional): Connectivity kernel (e.g., 6, 18, 26 for 3D). 
                                      Default is None (26-connected in 3D, 8-connected in 2D).

    Returns:
        dict: Dictionary containing the 16 GLSZM features:
            - **SmallAreaEmphasis**: Measures the distribution of small zones.
            - **LargeAreaEmphasis**: Measures the distribution of large zone sizes.
            - **GrayLevelNonUniformity**: Measures the variability of gray-level values in the image.
            - **GrayLevelNonUniformityNormalized**: Normalized version of GLN.
            - **ZoneSizeNonUniformity**: Measures the variability of zone size values.
            - **ZoneSizeNonUniformityNormalized**: Normalized version of ZSN.
            - **ZonePercentage**: Measures the coarseness of the texture.
            - **LowGrayLevelZoneEmphasis**: Measures the distribution of lower gray-level values.
            - **HighGrayLevelZoneEmphasis**: Measures the distribution of higher gray-level values.
            - **SmallAreaLowGrayLevelEmphasis**: Emphasis on small zones with low gray levels.
            - **SmallAreaHighGrayLevelEmphasis**: Emphasis on small zones with high gray levels.
            - **LargeAreaLowGrayLevelEmphasis**: Emphasis on large zones with low gray levels.
            - **LargeAreaHighGrayLevelEmphasis**: Emphasis on large zones with high gray levels.
            - **GrayLevelVariance**: Variance of gray level intensities in the zones.
            - **ZoneSizeVariance**: Variance of zone size volumes.
            - **ZoneEntropy**: Uncertainty/Randomness in the distribution of zone sizes and gray levels.

    Example:
        >>> import numpyradiomics as npr
        >>> # Generate a noisy ellipsoid
        >>> img, mask = npr.dro.noisy_ellipsoid(radii_mm=(15, 15, 15), intensity_range=(0, 100))
        >>> 
        >>> # Compute GLSZM features
        >>> feats = npr.glszm(img, mask, binWidth=10)
        >>> 
        >>> print(f"ZonePercentage: {feats['ZonePercentage']:.4f}")
        ZonePercentage: 0.8912
    """
    roi_mask = mask > 0
    if not np.any(roi_mask):
        raise ValueError("Mask contains no voxels.")

    # --- Step 1: Discretization ---
    img_q = _discretize_image(image, mask, binWidth=binWidth, binCount=binCount)
    
    # 4. Handle Levels (Legacy/Safety)
    if levels is None:
        levels = int(img_q.max())
    else:
        img_q = np.clip(img_q, 0, levels)

    # --- Step 2: Zone Counting ---
    dims = image.ndim
    
    # Default connectivity: 8 (2D) or 26 (3D) -> All neighbors
    if connectivity is None:
        connectivity = 26 if dims == 3 else 8
        
    structure = _get_connectivity_structure(dims, connectivity)
    
    # Initialize Matrix: Rows (Gray Levels) x Cols (Zone Sizes)
    # Max potential zone size is the total ROI volume (all voxels connected)
    max_zone = np.sum(roi_mask)
    
    # We build the matrix. Note: zone size 1 is index 0. Gray level 1 is index 0.
    glszm_mat = np.zeros((levels, max_zone + 1), dtype=np.float64)
    
    # Optimization: Only process levels present in the ROI
    active_levels = np.unique(img_q[roi_mask])
    
    for g in active_levels:
        if g == 0: continue
        
        # Binary mask for this specific gray level
        mask_g = (img_q == g)
        
        # Connected components labeling
        # labeled_map is an integer array where each zone has a unique ID
        labeled_map, num_features = label(mask_g, structure=structure)
        
        if num_features > 0:
            # Count sizes of each zone
            # bincount returns counts for label 0 (background), 1, 2...
            # We skip index 0 (background of the mask_g).
            sizes = np.bincount(labeled_map.ravel())[1:]
            
            # Accumulate into matrix
            # g-1 because matrix is 0-indexed for gray levels
            # sizes-1 because matrix is 0-indexed for sizes (size 1 -> index 0)
            np.add.at(glszm_mat, (g - 1, sizes - 1), 1)

    # --- Step 3: Compute Features ---
    # Np is the total number of voxels in the ROI (used for ZonePercentage)
    Np = max_zone 
    return _compute_glszm_features(glszm_mat, Np)

def _get_connectivity_structure(dims, connectivity):
    """Generate the scipy.ndimage.label structure element."""
    if dims == 2:
        if connectivity == 4:
            return np.array([[0,1,0],[1,1,1],[0,1,0]])
        else: # 8
            return np.ones((3,3), dtype=int)
    elif dims == 3:
        struct = np.zeros((3,3,3), dtype=int)
        c = (1,1,1) # Center
        struct[c] = 1
        
        # 6-connectivity (Faces)
        faces = [(0,1,1), (2,1,1), (1,0,1), (1,2,1), (1,1,0), (1,1,2)]
        for p in faces: struct[p] = 1
        if connectivity == 6: return struct
        
        # 18-connectivity (Faces + Edges)
        # Edges are coordinates varying in 2 dimensions
        # Easier: 18 = 3x3x3 minus corners
        if connectivity == 18:
            s18 = np.ones((3,3,3), dtype=int)
            corners = [(0,0,0),(0,0,2),(0,2,0),(0,2,2),
                       (2,0,0),(2,0,2),(2,2,0),(2,2,2)]
            for p in corners: s18[p] = 0
            return s18
            
        # 26-connectivity (All neighbors)
        return np.ones((3,3,3), dtype=int)
    return None

def _compute_glszm_features(P, Np):
    Nz = np.sum(P)
    if Nz == 0:
        return {}
    
    # Normalize
    P_norm = P / Nz
    Ng, Ns = P.shape
    
    # Indices (1-based) -- FORCE FLOAT64 to prevent int32 overflow
    i = np.arange(1, Ng + 1, dtype=np.float64).reshape(-1, 1) 
    j = np.arange(1, Ns + 1, dtype=np.float64).reshape(1, -1) 
    
    # Marginals
    pg = np.sum(P_norm, axis=1) # (Ng,)
    ps = np.sum(P_norm, axis=0) # (Ns,)
    
    # Flatten for 1D sums
    i_flat = i.flatten()
    j_flat = j.flatten()
    
    # ... (Rest of function remains exactly the same) ...
    # Now (i**2 * j**2) will happen in float64 space, safely handling 10^10+ values.
    
    # Means
    mu_i = np.sum(pg * i_flat)
    mu_j = np.sum(ps * j_flat)
    
    # Features
    sae = np.sum(ps / (j_flat**2))
    lae = np.sum(ps * (j_flat**2))
    gln = np.sum(np.sum(P, axis=1)**2) / Nz
    glnn = gln / Nz
    zsn = np.sum(np.sum(P, axis=0)**2) / Nz
    zsnn = zsn / Nz
    zp = Nz / Np
    lglze = np.sum(pg / (i_flat**2))
    hglze = np.sum(pg * (i_flat**2))
    salgle = np.sum(P_norm / ((i**2) * (j**2)))
    sahgle = np.sum(P_norm * (i**2) / (j**2))
    lalgle = np.sum(P_norm * (j**2) / (i**2))
    lahgle = np.sum(P_norm * (i**2) * (j**2))
    glv = np.sum(pg * (i_flat - mu_i)**2)
    zsv = np.sum(ps * (j_flat - mu_j)**2)
    
    eps = 2e-16
    ze = -np.sum(P_norm * np.log2(P_norm + eps))
    
    return {
        'SmallAreaEmphasis': sae, 'LargeAreaEmphasis': lae,
        'GrayLevelNonUniformity': gln, 'GrayLevelNonUniformityNormalized': glnn,
        'ZoneSizeNonUniformity': zsn, 'ZoneSizeNonUniformityNormalized': zsnn,
        'ZonePercentage': zp,
        'LowGrayLevelZoneEmphasis': lglze, 'HighGrayLevelZoneEmphasis': hglze,
        'SmallAreaLowGrayLevelEmphasis': salgle, 'SmallAreaHighGrayLevelEmphasis': sahgle,
        'LargeAreaLowGrayLevelEmphasis': lalgle, 'LargeAreaHighGrayLevelEmphasis': lahgle,
        'GrayLevelVariance': glv, 'ZoneSizeVariance': zsv, 'ZoneEntropy': ze
    }

def glszm_units(base_unit=""):
    """
    Returns units for GLSZM features.

    Args:
        intensity_unit (str, optional): The unit of pixel intensity (e.g., 'HU', 'GV', ''). Default is 'HU'.

    Returns:
        dict: Dictionary mapping feature names to their units.

    Example:
        >>> from numpyradiomics import glszm_units
        >>> units = glszm_units(intensity_unit='SUV')
        >>> print(units['LowGrayLevelZoneEmphasis'])
        SUV^-2
    """
    return {
        # 1. Zone Size Emphasis (j) -> Dimensionless (Counts)
        "SmallAreaEmphasis": "",
        "LargeAreaEmphasis": "",
        
        # 2. Non-Uniformity -> Dimensionless
        "GrayLevelNonUniformity": "",
        "GrayLevelNonUniformityNormalized": "",
        "ZoneSizeNonUniformity": "",
        "ZoneSizeNonUniformityNormalized": "",
        "ZonePercentage": "",
        
        # 3. Gray Level Emphasis (i) -> Intensity Units
        "LowGrayLevelZoneEmphasis": f"{base_unit}^-2",      # 1/I^2
        "HighGrayLevelZoneEmphasis": f"{base_unit}^2",      # I^2
        
        # 4. Mixed Emphasis
        "SmallAreaLowGrayLevelEmphasis": f"{base_unit}^-2",
        "SmallAreaHighGrayLevelEmphasis": f"{base_unit}^2",
        "LargeAreaLowGrayLevelEmphasis": f"{base_unit}^-2",
        "LargeAreaHighGrayLevelEmphasis": f"{base_unit}^2",
        
        # 5. Variance / Entropy
        "GrayLevelVariance": f"{base_unit}^2",
        "ZoneSizeVariance": "",
        "ZoneEntropy": ""
    }