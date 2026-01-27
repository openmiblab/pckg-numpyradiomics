import numpy as np
from scipy.ndimage import convolve

def ngtdm(image, mask, binWidth=25, distance=1):
    """
    Compute 5 Pyradiomics-style NGTDM (Neighborhood Gray Tone Difference Matrix) features.

    The NGTDM quantifies the difference between a gray value and the average gray value
    of its neighbors within a defined distance.

    Args:
        image (np.ndarray): 3D image array containing voxel intensities.
        mask (np.ndarray): 3D mask array (same shape as image), where non-zero values indicate the ROI.
        binWidth (float, optional): Width of bins for discretization. Default is 25.
        distance (int, optional): The distance (radius) of the neighborhood kernel. Default is 1.

    Returns:
        dict: Dictionary containing the 5 NGTDM features:
            - **Coarseness**: Measures the average difference between the center voxel and its neighborhood.
                              Higher values indicate a lower spatial change rate and locally more uniform texture.
            - **Contrast**: Measures the spatial intensity change rate.
                              Higher values indicate large changes in intensity between neighboring areas.
            - **Busyness**: Measures the rapid changes of intensity between pixels and their neighborhood.
                              High values indicate a "busy" image (rapid changes).
            - **Complexity**: Measures the information content of the image.
                              High values indicate non-uniform and rapid changes in gray levels.
            - **Strength**: Measures the distinctness of the primitives in the image.
                              High values indicate easily definable and visible structures.

    Example:
        >>> import numpyradiomics as npr
        >>> # Generate a noisy ellipsoid
        >>> img, mask = npr.dro.noisy_ellipsoid(radii_mm=(12, 12, 12), intensity_range=(0, 100))
        >>> 
        >>> # Compute NGTDM features
        >>> feats = npr.ngtdm(img, mask, binWidth=10, distance=1)
        >>> 
        >>> print(f"Coarseness: {feats['Coarseness']:.6f}")
        Coarseness: 0.001234
    """
    roi_mask = mask > 0
    if not np.any(roi_mask):
        raise ValueError("Mask contains no voxels.")

    roi_image = image.copy()
    
    # Restrict ROI for min/max to match PyRadiomics ROI-based binning
    masked_pixels = roi_image[roi_mask]
    min_val = np.min(masked_pixels)
    max_val = np.max(masked_pixels)
    
    # Handle degenerate case (Flat Region)
    if max_val == min_val:
        # Coarseness caps at 1e6 for flat regions, others are 0
        return {
            "Coarseness": 1000000.0, 
            "Contrast": 0.0, 
            "Busyness": 0.0, 
            "Complexity": 0.0, 
            "Strength": 0.0
        }

    # --- 1. Quantization (PyRadiomics Style) ---
    # PyRadiomics uses floor((x - min) / binWidth) + 1
    # resulting in 1-based indices: 1, 2, ..., N_bins
    
    # Calculate exact number of bins needed
    N_bins = int(np.floor((max_val - min_val) / binWidth)) + 1
    
    # Discretize
    # We use float temporarily to handle the division
    pixel_bins = np.floor((roi_image - min_val) / binWidth) + 1
    
    # Clip to ensure numerical stability at the max_val edge
    pixel_bins = np.clip(pixel_bins, 1, N_bins).astype(int)
    
    # Apply Mask (0 is background/ignored now, 1..N are valid bins)
    img_quant = pixel_bins * roi_mask

    # --- 2. Neighborhood Pre-processing ---
    # Kernel: 3D cube with 0 at center
    k_size = 2 * distance + 1
    kernel = np.ones((k_size, k_size, k_size), dtype=np.float64)
    kernel[distance, distance, distance] = 0
    
    # Calculate sum and count of neighbors
    # Note: img_quant has values 1..N inside mask, 0 outside.
    # convolve with cval=0 treats outside as 0 (ignored).
    neighbor_sum = convolve(img_quant.astype(float), kernel, mode='constant', cval=0)
    neighbor_count = convolve(roi_mask.astype(float), kernel, mode='constant', cval=0)

    # Valid neighborhood: inside mask AND has neighbors
    valid_mask = (neighbor_count > 0) & roi_mask
    
    # Calculate average neighborhood intensity (A_bar)
    # Note: neighbor_sum sums indices (1..N). This matches PyRadiomics 
    # which calculates texture features on the discretized indices.
    local_mean = np.zeros_like(neighbor_sum)
    local_mean[valid_mask] = neighbor_sum[valid_mask] / neighbor_count[valid_mask]

    # --- 3. Build NGTDM Matrix (N_i and S_i) ---
    # We need vectors of length N_bins (indices 0..N-1 mapped to levels 1..N)
    N_i = np.zeros(N_bins, dtype=np.float64)
    S_i = np.zeros(N_bins, dtype=np.float64)

    valid_voxels = img_quant[valid_mask]     # Values are 1..N
    valid_means = local_mean[valid_mask]
    
    # Vectorized accumulation
    # Subtract 1 from voxel values to get 0-based array indices
    voxel_indices = valid_voxels - 1 
    
    # Count occurrences (N_i)
    N_i = np.bincount(voxel_indices, minlength=N_bins).astype(np.float64)
    
    # Calculate S_i: Sum of absolute differences |i - mean|
    # We iterate because bincount cannot sum a derived float difference easily
    for i in range(N_bins):
        # i is 0-based index, corresponding to Gray Level (i + 1)
        level_val = i + 1
        matches = (valid_voxels == level_val)
        if np.any(matches):
            # Difference between Integer Level and Float Mean
            S_i[i] = np.sum(np.abs(level_val - valid_means[matches]))

    # --- 4. Compute Features ---
    
    N_total = np.sum(N_i) # N_vp (Valid Pixels)
    if N_total == 0:
        return {"Coarseness": 1e6, "Contrast": 0.0, "Busyness": 0.0, "Complexity": 0.0, "Strength": 0.0}

    p_i = N_i / N_total
    
    # Gray Level Vector: PyRadiomics uses 1-based indices (1, 2, 3...) for features
    i_vec = np.arange(1, N_bins + 1, dtype=np.float64)
    
    # Identify non-zeros for efficiency (Ngp calculation)
    nz_indices = np.where(p_i > 0)[0]
    Ngp = len(nz_indices)
    
    # --- Coarseness ---
    sum_pi_si = np.sum(p_i * S_i)
    coarseness = 1.0 / sum_pi_si if sum_pi_si != 0 else 1000000.0

    # --- Contrast ---
    if Ngp > 1:
        # Standard deviation term involves double sum over p_i * p_j * (i-j)^2
        # Vectorized: Outer product
        i_diff = i_vec[:, None] - i_vec[None, :]
        p_prod = np.outer(p_i, p_i)
        
        term1 = np.sum(p_prod * (i_diff**2)) / (Ngp * (Ngp - 1))
        term2 = np.sum(S_i) / N_total
        contrast = term1 * term2
    else:
        contrast = 0.0

    # --- Busyness ---
    # Denom: sum |i*p_i - j*p_j|
    ip = i_vec * p_i
    denom_busy = np.sum(np.abs(ip[:, None] - ip[None, :]))
    
    busyness = np.sum(p_i * S_i) / denom_busy if denom_busy != 0 else 0.0

    # --- Complexity ---
    # Term: |i-j| * ( (p_i s_i + p_j s_j) / (p_i + p_j) )
    p_sum = p_i[:, None] + p_i[None, :]
    p_sum[p_sum == 0] = 1.0 # Avoid div by zero (numerator will be 0 anyway)
    
    pi_si = p_i * S_i
    num_complex = pi_si[:, None] + pi_si[None, :]
    
    term_complex = (np.abs(i_diff) * num_complex) / p_sum
    complexity = np.sum(term_complex) / N_total

    # --- Strength ---
    # Num: (p_i + p_j) * (i-j)^2
    # Denom: sum(S_i)
    # Force p_sum to be clean again just in case
    p_sum = p_i[:, None] + p_i[None, :]
    
    strength_num = np.sum(p_sum * (i_diff**2))
    strength_denom = np.sum(S_i)
    
    strength = strength_num / strength_denom if strength_denom != 0 else 0.0

    return {
        "Coarseness": coarseness,
        "Contrast": contrast,
        "Busyness": busyness,
        "Complexity": complexity,
        "Strength": strength
    }


def ngtdm_units(intensity_unit='HU'):
    """
    Returns units for NGTDM features.

    Args:
        intensity_unit (str, optional): The unit of pixel intensity (e.g., 'HU', 'GV', ''). Default is 'HU'.

    Returns:
        dict: Dictionary mapping feature names to their units.

    Example:
        >>> from numpyradiomics import ngtdm_units
        >>> units = ngtdm_units(intensity_unit='HU')
        >>> print(units['Contrast'])
        HU^3
    """
    base_unit = intensity_unit
    
    # Variables:
    # i  = Intensity (base_unit)
    # P  = Probability (dimensionless)
    # S  = Sum of absolute differences (base_unit)
    
    return {
        "Coarseness": f"{base_unit}^-1",       # 1 / sum(P * S) -> 1 / (1 * U) -> U^-1
        "Contrast": f"{base_unit}^3",          # Variance(U^2) * NormSum(S)(U) -> U^3
        "Busyness": "",                        # sum(P*S) / sum(|iP - jP|) -> U / U -> Dimensionless
        "Complexity": f"{base_unit}^2",        # sum(|i-j| * (PS+PS)/(P+P)) -> U * U / 1 -> U^2
        "Strength": base_unit                  # sum((P+P)*(i-j)^2) / sum(S) -> (1 * U^2) / U -> U
    }