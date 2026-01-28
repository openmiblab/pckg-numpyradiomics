from typing import Optional, Union
import numpy as np
from numpy.lib.stride_tricks import as_strided

from .mod_preproc import _discretize_image

def glrlm(
    image: np.ndarray, 
    mask: np.ndarray, 
    binWidth: float = 25, 
    binCount: Optional[int] = None, 
    levels: Optional[int] = None
):
    """
    Compute 16 Pyradiomics-style GLRLM (Gray Level Run Length Matrix) features.

    The GLRLM quantifies gray level runs, which are defined as the length in number of
    pixels, of consecutive pixels that have the same gray level value. The matrix
    is computed for the 13 principal directions in 3D (or 4 in 2D) and the features
    are averaged over these angles.

    Args:
        image (np.ndarray): 3D image array containing voxel intensities.
        mask (np.ndarray): 3D mask array (same shape as image), where non-zero values indicate the ROI.
        binWidth (float, optional): Width of bins for 'Fixed Bin Width' discretization. Default is 25.
        binCount (int, optional): Number of bins for 'Fixed Bin Count' discretization. 
                                  If specified, overrides binWidth logic. Default is None.
        levels (int, optional): (Deprecated) Number of levels. If None, calculated dynamically from the discretized image.

    Returns:
        dict: Dictionary containing the 16 GLRLM features (averaged over all directions):
            - **ShortRunEmphasis**: Emphasis on short runs.
            - **LongRunEmphasis**: Emphasis on long runs.
            - **GrayLevelNonUniformity**: Variability of gray-level values in the image.
            - **GrayLevelNonUniformityNormalized**: Normalized version of GLN.
            - **RunLengthNonUniformity**: Variability of run lengths.
            - **RunLengthNonUniformityNormalized**: Normalized version of RLN.
            - **RunPercentage**: Fraction of realized runs versus potential runs (coarseness).
            - **LowGrayLevelRunEmphasis**: Distribution of low gray-level values.
            - **HighGrayLevelRunEmphasis**: Distribution of high gray-level values.
            - **ShortRunLowGrayLevelRunEmphasis**: Joint distribution of short runs and low gray levels.
            - **ShortRunHighGrayLevelRunEmphasis**: Joint distribution of short runs and high gray levels.
            - **LongRunLowGrayLevelRunEmphasis**: Joint distribution of long runs and low gray levels.
            - **LongRunHighGrayLevelRunEmphasis**: Joint distribution of long runs and high gray levels.
            - **GrayLevelVariance**: Variance of gray levels in runs.
            - **RunLengthVariance**: Variance of run lengths.
            - **RunEntropy**: Randomness/variability in run lengths and gray levels.

    Example:
        >>> import numpyradiomics as npr
        >>> # Generate a noisy ellipsoid
        >>> img, mask = npr.dro.noisy_ellipsoid(radii_mm=(15, 15, 15), intensity_range=(0, 100))
        >>> 
        >>> # Compute GLRLM features
        >>> feats = npr.glrlm(img, mask, binWidth=10)
        >>> 
        >>> print(f"RunPercentage: {feats['RunPercentage']:.4f}")
        RunPercentage: 0.8241
    """
    if not np.any(mask > 0):
        raise ValueError("Mask contains no voxels.")

    # --- Step 1: Discretization ---
    img_q = _discretize_image(image, mask, binWidth=binWidth, binCount=binCount)
    
    # 4. Determine Levels
    # If levels were manually passed (legacy), we can cap, but standard behavior
    # is to derive it from the discretized max value.
    if levels is None:
        levels = int(img_q.max())
    else:
        # If user forces levels, we clip (though discretize usually handles this)
        img_q = np.clip(img_q, 0, levels)

    # --- Step 2: Compute Features per Angle ---
    dims = image.ndim
    angles = _get_angles(dims)
    max_dim = max(img_q.shape)
    
    # Number of voxels in ROI
    Np = np.sum(mask > 0)
    
    feature_sums = {}
    valid_angles = 0
    
    for angle in angles:
        # Build Matrix for THIS angle
        # Rows: Gray Levels (1..levels)
        # Cols: Run Lengths (1..max_dim)
        glrlm_mat = np.zeros((levels, max_dim + 1), dtype=np.float64)
        
        # _compute_runs_skewed is a highly optimized helper using stride tricks
        # It expects 0 as background and 1..N as foreground.
        runs, lengths = _compute_runs_skewed(img_q, angle)
        
        if len(runs) > 0:
            # Clip lengths to matrix dimensions (safety)
            safe_lengths = np.clip(lengths, 1, max_dim + 1)
            
            # Populate Matrix (subtract 1 for 0-based indexing)
            np.add.at(glrlm_mat, (runs - 1, safe_lengths - 1), 1)
            
            # Compute features for this specific angle
            feats = _compute_glrlm_features(glrlm_mat, Np)
            
            # Accumulate
            if not feature_sums:
                feature_sums = {k: 0.0 for k in feats}
            
            for k, v in feats.items():
                feature_sums[k] += v
            valid_angles += 1

    # --- Step 3: Average Features ---
    if valid_angles == 0:
        return {} 
        
    return {k: v / valid_angles for k, v in feature_sums.items()}

def _get_angles(dims):
    if dims == 2:
        return [(0,1), (1,0), (1,1), (1,-1)]
    elif dims == 3:
        return [
            (0,0,1), (0,1,0), (1,0,0),
            (0,1,1), (0,1,-1), (1,0,1), (1,0,-1), (1,1,0), (1,-1,0),
            (1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1)
        ]
    return []

def _compute_runs_skewed(img, angle):
    """
    Compute Run Lengths for a given direction using vectorized skewing.
    FIX: Uses double-sided padding to prevent Access Violations at array bounds.
    """
    angle = np.array(angle)
    primary_axis = np.argmax(np.abs(angle))
    
    # 1. Align primary axis to 0 (Depth)
    img_perm = np.moveaxis(img, primary_axis, 0)
    angle_perm = np.roll(angle, -primary_axis)
    
    # 2. Ensure positive strides (Flip axes if needed)
    d0, d1, d2 = 1, angle_perm[1], angle_perm[2]
    
    if angle[primary_axis] < 0:
        img_perm = np.flip(img_perm, axis=0)
    if d1 < 0:
        img_perm = np.flip(img_perm, axis=1)
        d1 = abs(d1)
    if img_perm.ndim == 3 and d2 < 0:
        img_perm = np.flip(img_perm, axis=2)
        d2 = abs(d2)

    if img_perm.ndim == 2:
        img_perm = img_perm[:, :, None]
        d2 = 0
        
    nz, ny, nx = img_perm.shape

    # 3. Calculate Padding (Double-sided)
    # Start Padding: Captures diagonals entering from the side.
    # End Padding: Prevents rays starting at the bottom from going OOB.
    pad_y = nz * d1
    pad_x = nz * d2
    
    # Pad Axis 0 (Depth) with 1 zero at END (Run breaker)
    # Pad Axis 1, 2 (Spatial) at BOTH Start and End
    padded = np.pad(img_perm, 
                   ((0, 1), (pad_y, pad_y), (pad_x, pad_x)),
                   mode='constant', constant_values=0)
    
    # 4. Create Skewed View
    s0, s1, s2 = padded.strides
    stride_diag = s0 + d1*s1 + d2*s2
    
    # View dimensions:
    # We iterate starting points that cover the original image width + the start padding
    h = ny + pad_y
    w = nx + pad_x
    depth_pad = nz + 1
    
    # Check that memory footprint is safe (Conceptual check)
    # Max index accessed = (h-1)*s1 + (w-1)*s2 + (depth_pad-1)*stride_diag
    # With double padding, 'padded' is large enough to contain this.
    
    skewed_vol = as_strided(padded, 
                           shape=(h, w, depth_pad), 
                           strides=(s1, s2, stride_diag))
    
    # 5. Flatten and RLE
    flat = skewed_vol.reshape(-1, depth_pad).ravel()
    
    # Standard Vectorized RLE
    
    # Start: Value is non-zero, and different from previous
    # We shift 'flat' right by 1 (prepend 0) to compare
    flat_prev = np.concatenate(([0], flat[:-1]))
    starts_mask = (flat != 0) & (flat != flat_prev)
    starts = np.where(starts_mask)[0]
    
    # End: Value is non-zero, and different from next
    # We shift 'flat' left by 1 (append 0) to compare
    flat_next = np.concatenate((flat[1:], [0]))
    ends_mask = (flat != 0) & (flat != flat_next)
    ends = np.where(ends_mask)[0] + 1 
    
    lengths = ends - starts
    values = flat[starts]
    
    return values, lengths

def _compute_glrlm_features(P, Np):
    """
    Compute features from P matrix.
    """
    Nr = np.sum(P)
    if Nr == 0:
        return {}
        
    P_norm = P / Nr
    Ng, Ns = P.shape
    i = np.arange(1, Ng + 1).reshape(-1, 1)
    j = np.arange(1, Ns + 1).reshape(1, -1)
    
    pg = np.sum(P_norm, axis=1)
    pr = np.sum(P_norm, axis=0)
    
    i_flat = i.flatten()
    j_flat = j.flatten()
    
    mu_i = np.sum(pg * i_flat)
    mu_j = np.sum(pr * j_flat)
    
    sre = np.sum(pr / (j_flat**2))
    lre = np.sum(pr * (j_flat**2))
    
    # Non-Uniformity (Scaled by Nr)
    gln = np.sum(np.sum(P, axis=1)**2) / Nr
    rln = np.sum(np.sum(P, axis=0)**2) / Nr
    
    glnn = gln / Nr
    rlnn = rln / Nr
    
    rp = Nr / Np
    
    lglre = np.sum(pg / (i_flat**2))
    hglre = np.sum(pg * (i_flat**2))
    
    srlgle = np.sum(P_norm / ((i**2) * (j**2)))
    srhgle = np.sum(P_norm * (i**2) / (j**2))
    lrlgle = np.sum(P_norm * (j**2) / (i**2))
    lrhgle = np.sum(P_norm * (i**2) * (j**2))
    
    glv = np.sum(pg * (i_flat - mu_i)**2)
    rlv = np.sum(pr * (j_flat - mu_j)**2)
    
    eps = 2e-16
    ent = -np.sum(P_norm * np.log2(P_norm + eps))
    
    return {
        'ShortRunEmphasis': sre, 'LongRunEmphasis': lre,
        'GrayLevelNonUniformity': gln, 'GrayLevelNonUniformityNormalized': glnn,
        'RunLengthNonUniformity': rln, 'RunLengthNonUniformityNormalized': rlnn,
        'RunPercentage': rp,
        'LowGrayLevelRunEmphasis': lglre, 'HighGrayLevelRunEmphasis': hglre,
        'ShortRunLowGrayLevelRunEmphasis': srlgle, 'ShortRunHighGrayLevelRunEmphasis': srhgle,
        'LongRunLowGrayLevelRunEmphasis': lrlgle, 'LongRunHighGrayLevelRunEmphasis': lrhgle,
        'GrayLevelVariance': glv, 'RunLengthVariance': rlv, 'RunEntropy': ent
    }


def glrlm_units(intensity_unit='HU'):
    """
    Returns units for GLRLM features.

    Args:
        intensity_unit (str, optional): The unit of pixel intensity (e.g., 'HU', 'GV', ''). Default is 'HU'.

    Returns:
        dict: Dictionary mapping feature names to their units.

    Example:
        >>> from numpyradiomics import glrlm_units
        >>> units = glrlm_units(intensity_unit='HU')
        >>> print(units['ShortRunLowGrayLevelRunEmphasis'])
        HU^-2
    """
    base_unit = intensity_unit
    return {
        # 1. Run Length Emphasis (j) -> Dimensionless (Lengths)
        "ShortRunEmphasis": "",
        "LongRunEmphasis": "",
        
        # 2. Non-Uniformity -> Dimensionless (Counts/Probabilities)
        "GrayLevelNonUniformity": "", 
        "GrayLevelNonUniformityNormalized": "",
        "RunLengthNonUniformity": "",
        "RunLengthNonUniformityNormalized": "",
        "RunPercentage": "",
        
        # 3. Gray Level Emphasis (i) -> Intensity Units
        "LowGrayLevelRunEmphasis": f"{base_unit}^-2",      # sum(P / i^2) -> 1/I^2
        "HighGrayLevelRunEmphasis": f"{base_unit}^2",      # sum(P * i^2) -> I^2
        
        # 4. Mixed Emphasis
        "ShortRunLowGrayLevelRunEmphasis": f"{base_unit}^-2", # sum(P / (i^2 * j^2))
        "ShortRunHighGrayLevelRunEmphasis": f"{base_unit}^2", # sum(P * i^2 / j^2)
        "LongRunLowGrayLevelRunEmphasis": f"{base_unit}^-2",  # sum(P * j^2 / i^2)
        "LongRunHighGrayLevelRunEmphasis": f"{base_unit}^2",  # sum(P * i^2 * j^2)
        
        # 5. Variance / Entropy
        "GrayLevelVariance": f"{base_unit}^2",             # Variance of Intensity
        "RunLengthVariance": "",                           # Variance of Lengths
        "RunEntropy": ""                                   # Entropy (bits)
    }