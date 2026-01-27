import numpy as np

def gldm(image, mask, binWidth=25, alpha=0, levels=None):
    """
    Compute 15 Pyradiomics-style GLDM (Gray Level Dependence Matrix) features.

    The GLDM quantifies gray level dependencies in an image. A gray level dependency is defined 
    as the number of connected voxels within distance $\delta=1$ that are dependent on the center voxel.
    A neighboring voxel is dependent if the absolute difference in gray level is $\le \alpha$.

    Args:
        image (np.ndarray): 3D image array containing voxel intensities.
        mask (np.ndarray): 3D mask array (same shape as image), where non-zero values indicate the ROI.
        binWidth (float, optional): Width of bins for discretization. Default is 25.
        alpha (int, optional): Cutoff for dependence. Neighbors are "dependent" if |val - center| <= alpha. 
                               Default is 0 (exact match).
        levels (int, optional): Number of levels for discretization. If None, calculated from binWidth.

    Returns:
        dict: Dictionary containing the 15 GLDM features:
            - **SmallDependenceEmphasis**: Distribution of small dependencies.
            - **LargeDependenceEmphasis**: Distribution of large dependencies.
            - **GrayLevelNonUniformity**: Variability of gray-level values in the image.
            - **DependenceNonUniformity**: Variability of dependence counts.
            - **DependenceNonUniformityNormalized**: Normalized version of DN.
            - **GrayLevelNonUniformityNormalized**: Normalized version of GLN.
            - **LowGrayLevelEmphasis**: Distribution of low gray-level values.
            - **HighGrayLevelEmphasis**: Distribution of high gray-level values.
            - **SmallDependenceLowGrayLevelEmphasis**: Joint distribution of small dependence and low gray levels.
            - **SmallDependenceHighGrayLevelEmphasis**: Joint distribution of small dependence and high gray levels.
            - **LargeDependenceLowGrayLevelEmphasis**: Joint distribution of large dependence and low gray levels.
            - **LargeDependenceHighGrayLevelEmphasis**: Joint distribution of large dependence and high gray levels.
            - **GrayLevelVariance**: Variance of gray levels.
            - **DependenceVariance**: Variance of dependence counts.
            - **DependenceEntropy**: Randomness/variability in dependence counts.

    Example:
        >>> import numpyradiomics as npr
        >>> # Generate a noisy ellipsoid
        >>> img, mask = npr.dro.noisy_ellipsoid(radii_mm=(12, 12, 12), intensity_range=(0, 100))
        >>> 
        >>> # Compute GLDM features (alpha=0 implies exact match dependency)
        >>> feats = npr.gldm(img, mask, binWidth=10, alpha=0)
        >>> 
        >>> print(f"DependenceEntropy: {feats['DependenceEntropy']:.4f}")
        DependenceEntropy: 2.1543
    """
    roi_mask = mask > 0
    if not np.any(roi_mask):
        raise ValueError("Mask contains no voxels.")

    # 1. Quantization
    roi_image = image.copy()
    min_val = np.min(roi_image[roi_mask])
    
    if levels is None:
        max_val = np.max(roi_image[roi_mask])
        levels = int(np.floor((max_val - min_val) / binWidth)) + 1
    
    # 1-based indexing for levels
    # Clamp to avoid potential floor() underflow/overflow at boundaries
    diff = roi_image - min_val
    diff[roi_mask] = np.maximum(diff[roi_mask], 0)
    
    img_q = np.floor(diff / binWidth).astype(np.int32) + 1
    img_q[~roi_mask] = 0
    img_q = np.clip(img_q, 0, levels) 

    # 2. Vectorized Neighbor Counting
    dims = image.ndim
    if dims == 2:
        offsets = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    elif dims == 3:
        offsets = []
        for z in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for x in [-1, 0, 1]:
                    if not (z==0 and y==0 and x==0):
                        offsets.append((z, y, x))
    
    dependency_map = np.zeros(img_q.shape, dtype=np.int32)
    
    for shift in offsets:
        src_slices = []
        dst_slices = []
        for s in shift:
            if s > 0:
                src_slices.append(slice(0, -s))
                dst_slices.append(slice(s, None))
            elif s < 0:
                src_slices.append(slice(-s, None))
                dst_slices.append(slice(0, s))
            else:
                src_slices.append(slice(None))
                dst_slices.append(slice(None))
        
        src_vals = img_q[tuple(src_slices)]
        dst_vals = img_q[tuple(dst_slices)]
        
        match = (src_vals > 0) & (dst_vals > 0) & (np.abs(src_vals - dst_vals) <= alpha)
        
        temp_map = np.zeros_like(dependency_map)
        temp_map[tuple(src_slices)] = match
        dependency_map += temp_map

    # 3. Build Matrix
    valid_mask = (img_q > 0)
    flat_gray = img_q[valid_mask]      
    flat_dep  = dependency_map[valid_mask] 
    
    Ng = levels
    max_dep = len(offsets)
    
    P = np.histogram2d(
        flat_gray - 1, 
        flat_dep, 
        bins=[Ng, max_dep + 1], 
        range=[[0, Ng], [0, max_dep + 1]]
    )[0]

    return _compute_gldm_features(P)

def _compute_gldm_features(P):
    Nz = np.sum(P)
    if Nz == 0:
        return {}
        
    P_norm = P / Nz
    Ng, Nd = P.shape
    
    # Grid Indices (1-based)
    # i: Gray Levels, j: Dependency Counts
    i = np.arange(1, Ng + 1).reshape(-1, 1)
    j = np.arange(1, Nd + 1).reshape(1, -1)
    
    # Marginal probabilities
    pg = np.sum(P_norm, axis=1) # (Ng,)
    pd = np.sum(P_norm, axis=0) # (Nd,)
    
    # --- FIX 2: Broadcasting Safety ---
    # Flatten indices for 1D weighted sums to prevent outer-product behavior
    i_flat = i.flatten()
    j_flat = j.flatten()

    # 1. Small Dependence Emphasis (SDE)
    sde = np.sum(pd / (j_flat**2))
    
    # 2. Large Dependence Emphasis (LDE)
    lde = np.sum(pd * (j_flat**2))
    
    # 3. Gray Level Non-Uniformity (GLN)
    # --- FIX 1: Scaling ---
    # PyRadiomics divides the sum-of-squares-counts by Nz
    gln = np.sum(np.sum(P, axis=1)**2) / Nz
    
    # 4. Dependence Non-Uniformity (DN)
    dn = np.sum(np.sum(P, axis=0)**2) / Nz
    
    # 5. Dependence Non-Uniformity Normalized (DNN)
    # Formula: sum(pd^2). Equivalently: (sum(counts^2) / Nz^2)
    # Since 'dn' is already (counts^2 / Nz), we just divide by Nz again.
    dnn = dn / Nz
    
    # 6. Gray Level Non-Uniformity Normalized (GLNN)
    glnn = gln / Nz
    
    # 7. Low Gray Level Emphasis (LGLE)
    lgle = np.sum(pg / (i_flat**2))
    
    # 8. High Gray Level Emphasis (HGLE)
    hgle = np.sum(pg * (i_flat**2))
    
    # 9. Small Dependence Low Gray Level Emphasis (SDLGLE)
    # 2D sums use the original 2D (Ng,1) and (1,Nd) arrays which broadcast correctly
    sdlgle = np.sum(P_norm / ((i**2) * (j**2)))
    
    # 10. Small Dependence High Gray Level Emphasis (SDHGLE)
    sdhgle = np.sum(P_norm * (i**2) / (j**2))
    
    # 11. Large Dependence Low Gray Level Emphasis (LDLGLE)
    ldlgle = np.sum(P_norm * (j**2) / (i**2))
    
    # 12. Large Dependence High Gray Level Emphasis (LDHGLE)
    ldhgle = np.sum(P_norm * (i**2) * (j**2))
    
    # 13. Gray Level Variance (GLV)
    mu_i = np.sum(pg * i_flat)
    glv = np.sum(pg * (i_flat - mu_i)**2)
    
    # 14. Dependence Variance (DV)
    mu_j = np.sum(pd * j_flat)
    dv = np.sum(pd * (j_flat - mu_j)**2)
    
    # 15. Dependence Entropy (DE)
    eps = 2e-16
    de = -np.sum(P_norm * np.log2(P_norm + eps))

    return {
        'SmallDependenceEmphasis': sde,
        'LargeDependenceEmphasis': lde,
        'GrayLevelNonUniformity': gln,
        'DependenceNonUniformity': dn,
        'DependenceNonUniformityNormalized': dnn,
        'GrayLevelNonUniformityNormalized': glnn,
        'LowGrayLevelEmphasis': lgle,
        'HighGrayLevelEmphasis': hgle,
        'SmallDependenceLowGrayLevelEmphasis': sdlgle,
        'SmallDependenceHighGrayLevelEmphasis': sdhgle,
        'LargeDependenceLowGrayLevelEmphasis': ldlgle,
        'LargeDependenceHighGrayLevelEmphasis': ldhgle,
        'GrayLevelVariance': glv,
        'DependenceVariance': dv,
        'DependenceEntropy': de
    }


def gldm_units(intensity_unit='HU'):
    """
    Returns units for GLDM features.

    Args:
        intensity_unit (str, optional): The unit of pixel intensity (e.g., 'HU', 'GV', ''). Default is 'HU'.

    Returns:
        dict: Dictionary mapping feature names to their units.

    Example:
        >>> from numpyradiomics import gldm_units
        >>> units = gldm_units(intensity_unit='HU')
        >>> print(units['LowGrayLevelEmphasis'])
        HU^-2
    """
    base_unit = intensity_unit
    return {
        # 1. Dependence Counts (j) -> Dimensionless (Neighbor counts)
        "SmallDependenceEmphasis": "",             # sum(P / j^2)
        "LargeDependenceEmphasis": "",             # sum(P * j^2)
        
        # 2. Non-Uniformity (Probabilities/Counts) -> Dimensionless
        "GrayLevelNonUniformity": "",              # sum(counts^2) / N
        "DependenceNonUniformity": "",
        "GrayLevelNonUniformityNormalized": "",
        "DependenceNonUniformityNormalized": "",
        
        # 3. Gray Level Emphasis (i) -> Intensity Units
        "LowGrayLevelEmphasis": f"{base_unit}^-2",                # sum(P / i^2)
        "HighGrayLevelEmphasis": f"{base_unit}^2",                # sum(P * i^2)
        
        # 4. Mixed Emphasis
        "SmallDependenceLowGrayLevelEmphasis": f"{base_unit}^-2", # sum(P / (i^2 * j^2)) -> 1/I^2
        "SmallDependenceHighGrayLevelEmphasis": f"{base_unit}^2", # sum(P * i^2 / j^2)   -> I^2
        "LargeDependenceLowGrayLevelEmphasis": f"{base_unit}^-2", # sum(P * j^2 / i^2)   -> 1/I^2
        "LargeDependenceHighGrayLevelEmphasis": f"{base_unit}^2", # sum(P * i^2 * j^2)   -> I^2
        
        # 5. Variance/Entropy
        "GrayLevelVariance": f"{base_unit}^2",     # Variance of I
        "DependenceVariance": "",                  # Variance of counts
        "DependenceEntropy": "",                   # Bits
    }