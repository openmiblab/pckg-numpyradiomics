import numpy as np
from typing import Optional

from .mod_preproc import _discretize_image

def ngtdm(
    image: np.ndarray, 
    mask: np.ndarray, 
    binWidth: float = 25, 
    binCount: Optional[int] = None,
    distance: int = 1
):
    """
    Compute 5 Pyradiomics-style NGTDM (Neighborhood Gray Tone Difference Matrix) features.

    The NGTDM quantifies the difference between a gray value and the average gray value
    of its neighbors within a defined distance.

    Args:
        image (np.ndarray): 3D image array containing voxel intensities.
        mask (np.ndarray): 3D mask array (same shape as image), where non-zero values indicate the ROI.
        binWidth (float, optional): Width of bins for 'Fixed Bin Width' discretization. Default is 25.
        binCount (int, optional): Number of bins for 'Fixed Bin Count' discretization. 
                                  If specified, overrides binWidth logic. Default is None.
        distance (int, optional): The distance (radius) of the neighborhood kernel. Default is 1.

    Returns:
        dict: Dictionary containing the 5 NGTDM features:
            - **Coarseness**: Measures the average difference between the center voxel and its neighborhood.
            - **Contrast**: Measures the spatial intensity change rate.
            - **Busyness**: Measures the rapid changes of intensity between pixels and their neighborhood.
            - **Complexity**: Measures the information content of the image.
            - **Strength**: Measures the distinctness of the primitives in the image.

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

    # --- Step 1: Discretization ---
    img_quant = _discretize_image(image, mask, binWidth=binWidth, binCount=binCount)
    
    N_bins = int(img_quant.max())
    
    if N_bins <= 1:
        return {
            "Coarseness": 1000000.0, "Contrast": 0.0, "Busyness": 0.0, 
            "Complexity": 0.0, "Strength": 0.0
        }

    # --- Step 2: Calculate Neighborhood Mean (Dynamic) ---
    
    # PyRadiomics iterates over all angles defined by 'distance'.
    # In 3D with d=1, this is the 26-connectivity (Chebyshev distance 1).
    # We simulate this by shifting the array in all 26 directions.
    
    dims = image.ndim
    
    # Generate offsets (Chebyshev distance)
    # Exclude (0,0,0)
    offsets = []
    if dims == 3:
        for z in range(-distance, distance + 1):
            for y in range(-distance, distance + 1):
                for x in range(-distance, distance + 1):
                    if z == 0 and y == 0 and x == 0:
                        continue
                    offsets.append((z, y, x))
    elif dims == 2:
        for y in range(-distance, distance + 1):
            for x in range(-distance, distance + 1):
                if y == 0 and x == 0:
                    continue
                offsets.append((y, x))
                
    # Accumulators for the mean calculation
    sum_neighbor_int = np.zeros_like(image, dtype=np.float64)
    count_neighbors = np.zeros_like(image, dtype=np.float64)
    
    # Vectorized Sliding Window using Slicing
    for shift in offsets:
        # Create source (shifted) and destination slices
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
                
        src_vals = img_quant[tuple(src_slices)]
        dst_vals = img_quant[tuple(dst_slices)] # This is the center pixel location
        
        # Condition: Neighbor must be inside ROI (val > 0)
        # We accumulate at the destination (center) only if the source (neighbor) is valid
        valid_neighbor = (src_vals > 0)
        
        # Add neighbor intensity to the center pixel's accumulator
        # We only add to pixels that are themselves in the ROI (dst_vals > 0)
        # But for efficiency, we can just add to the whole array view 
        # because we will mask the final result later.
        
        # Accumulate Sum
        # Note: img_quant has values 1..N.
        # PyRadiomics sums the GRAY LEVELS (indices), not the raw intensities.
        # Since img_quant matches PyRadiomics 'gray levels' (1-based), we sum img_quant.
        
        # We use simple += on the sliced views
        sum_neighbor_int[tuple(dst_slices)][valid_neighbor] += src_vals[valid_neighbor]
        count_neighbors[tuple(dst_slices)][valid_neighbor] += 1

    # --- Step 3: Identify Valid Centers and Compute Means ---
    
    # A voxel is valid if:
    # 1. It is in the ROI (img_quant > 0)
    # 2. It has at least one valid neighbor (count_neighbors > 0)
    #    (Edge cases with 0 neighbors are excluded from NGTDM)
    
    valid_mask = (img_quant > 0) & (count_neighbors > 0)
    
    if not np.any(valid_mask):
         return {"Coarseness": 1e6, "Contrast": 0.0, "Busyness": 0.0, "Complexity": 0.0, "Strength": 0.0}

    # Extract valid data
    valid_centers = img_quant[valid_mask]
    valid_sums = sum_neighbor_int[valid_mask]
    valid_counts = count_neighbors[valid_mask]
    
    # A_bar
    neighborhood_means = valid_sums / valid_counts

    # --- Step 4: Build NGTDM Matrix (N_i and S_i) ---
    
    # 0-based indices
    idx_values = valid_centers - 1
    
    # N_i: Count of voxels with gray level i
    N_i = np.bincount(idx_values, minlength=N_bins).astype(np.float64)
    
    # S_i: Sum of absolute differences |i - mean|
    S_i = np.zeros(N_bins, dtype=np.float64)
    abs_diffs = np.abs(valid_centers - neighborhood_means)
    np.add.at(S_i, idx_values, abs_diffs)

    # --- Step 5: Compute Features ---
    
    N_total = np.sum(N_i) # N_vp
    if N_total <= 0:
        return {"Coarseness": 1e6, "Contrast": 0.0, "Busyness": 0.0, "Complexity": 0.0, "Strength": 0.0}

    p_i = N_i / N_total
    i_vec = np.arange(1, N_bins + 1, dtype=np.float64)
    
    # Filter non-zeros
    nz_mask = (p_i > 0)
    p_i_nz = p_i[nz_mask]
    S_i_nz = S_i[nz_mask]
    i_vec_nz = i_vec[nz_mask]
    Ngp = len(p_i_nz)
    
    # --- Coarseness ---
    sum_pi_si = np.sum(p_i * S_i)
    coarseness = 1.0 / sum_pi_si if sum_pi_si != 0 else 1000000.0

    # --- Contrast ---
    if Ngp > 1:
        i_diff = i_vec_nz[:, None] - i_vec_nz[None, :]
        p_prod = np.outer(p_i_nz, p_i_nz)
        
        term1 = np.sum(p_prod * (i_diff**2)) / (Ngp * (Ngp - 1))
        term2 = np.sum(S_i) / N_total
        contrast = term1 * term2
    else:
        contrast = 0.0

    # --- Busyness ---
    ip = i_vec_nz * p_i_nz
    denom_busy = np.sum(np.abs(ip[:, None] - ip[None, :]))
    busyness = np.sum(p_i * S_i) / denom_busy if denom_busy != 0 else 0.0

    # --- Complexity ---
    i_diff_abs = np.abs(i_vec_nz[:, None] - i_vec_nz[None, :])
    
    p_sum = p_i_nz[:, None] + p_i_nz[None, :]
    p_sum[p_sum == 0] = 1.0 
    
    pi_si = p_i_nz * S_i_nz
    num_complex = pi_si[:, None] + pi_si[None, :]
    
    term_complex = (i_diff_abs * num_complex) / p_sum
    complexity = np.sum(term_complex) / N_total

    # --- Strength ---
    p_sum_clean = p_i_nz[:, None] + p_i_nz[None, :]
    strength_num = np.sum(p_sum_clean * ((i_vec_nz[:, None] - i_vec_nz[None, :])**2))
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