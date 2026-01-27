import numpy as np

def glcm(image, mask, binWidth=25, distances=[1], symmetricalGLCM=True, weightingNorm=None):
    """
    Compute 24 Pyradiomics-style GLCM (Gray Level Co-occurrence Matrix) features.

    The GLCM describes the second-order joint probability function of an image region,
    constrained by the mask. It counts how often pairs of voxels with specific gray
    levels and specific spatial relationships occur.

    Args:
        image (np.ndarray): 3D image array containing voxel intensities.
        mask (np.ndarray): 3D mask array (same shape as image), where non-zero values indicate the ROI.
        binWidth (float, optional): Width of bins for discretization. Default is 25.
        distances (list, optional): List of integer distances (offsets) to compute GLCMs for. Default is [1].
        symmetricalGLCM (bool, optional): If True, counts co-occurrences in both directions (i->j and j->i).
                                          If False, only counts i->j. Default is True.
        weightingNorm (str, optional): Method to weight GLCMs if multiple distances are used.
                                       Options: None, 'manhattan', 'euclidean', 'infinity'. Default is None.

    Returns:
        dict: Dictionary containing the 24 GLCM features:
            - **Autocorrelation**: Magnitude of the fineness and coarseness of texture.
            - **ClusterProminence**: Asymmetry of the GLCM.
            - **ClusterShade**: Skewness of the matrix.
            - **ClusterTendency**: Grouping of voxels with similar gray-level values.
            - **Contrast**: Local intensity variation (favors contributions away from the diagonal).
            - **Correlation**: Linear dependency of gray levels to their neighbors.
            - **DifferenceAverage**: Mean of the difference distribution.
            - **DifferenceEntropy**: Randomness/variability in neighborhood intensity differences.
            - **DifferenceVariance**: Variance of the difference distribution.
            - **JointAverage**: Mean gray level intensity of the $i$ distribution.
            - **JointEnergy**: Homogeneity of the texture (sum of squared elements).
            - **JointEntropy**: Randomness/variability in neighborhood intensity values.
            - **Id**: Inverse Difference (homogeneity 1).
            - **Idm**: Inverse Difference Moment (homogeneity 2).
            - **Idmn**: Normalized Inverse Difference Moment.
            - **Idn**: Normalized Inverse Difference.
            - **Imc1**: Informational Measure of Correlation 1.
            - **Imc2**: Informational Measure of Correlation 2.
            - **InverseVariance**: Inverse variance of the difference distribution.
            - **MCC**: Maximal Correlation Coefficient.
            - **MaximumProbability**: Occurrence of the most predominant pair of neighboring intensity values.
            - **SumAverage**: Mean of the sum distribution.
            - **SumEntropy**: Sum of entropies.
            - **SumSquares**: Variance of the distribution (Cluster Tendency is a specific case of this).

    Example:
        >>> import numpyradiomics as npr
        >>> # Generate a noisy ellipsoid
        >>> img, mask = npr.dro.noisy_ellipsoid(radii_mm=(10, 10, 10), intensity_range=(0, 100))
        >>> 
        >>> # Compute GLCM features (distance=1)
        >>> feats = npr.glcm(img, mask, binWidth=10, distances=[1])
        >>> 
        >>> print(f"Contrast: {feats['Contrast']:.4f}")
        Contrast: 12.4501
    """
    if not np.any(mask > 0):
        raise ValueError("Mask contains no voxels")

    roi = image[mask > 0]
    Imin = roi.min()
    
    # --- FIX: Clamp discretization to avoid epsilon errors ---
    # If image[i] == Imin, (image - Imin) might be -1e-16, causing floor() to be -1
    # This would map valid pixels to bin 0 (background), shifting the mean/autocorr.
    diff = image - Imin
    diff[mask > 0] = np.maximum(diff[mask > 0], 0)
    
    img_q = np.floor(diff / binWidth).astype(np.int32) + 1
    img_q[mask == 0] = 0
    levels = int(img_q.max())

    # --- FIX 2: Handle Degenerate (Single Gray Level) ROIs ---
    if levels <= 1:
        # Returns standard PyRadiomics defaults for flat regions
        return {
            'Autocorrelation': 1.0, 'ClusterProminence': 0.0, 'ClusterShade': 0.0, 
            'ClusterTendency': 0.0, 'Contrast': 0.0, 'Correlation': 1.0, 
            'DifferenceAverage': 0.0, 'DifferenceEntropy': 0.0, 'DifferenceVariance': 0.0, 
            'Id': 1.0, 'Idm': 1.0, 'Idmn': 1.0, 'Idn': 1.0, 'Imc1': 0.0, 'Imc2': 0.0, 
            'InverseVariance': 0.0, 'JointAverage': 1.0, 'JointEnergy': 1.0, 
            'JointEntropy': 0.0, 'MCC': 1.0, 'MaximumProbability': 1.0, 
            'SumAverage': 2.0, 'SumEntropy': 0.0, 'SumSquares': 0.0
        }

    # ... (Step 2: Generate Angles - SAME AS BEFORE) ...
    dims = image.ndim
    if dims == 3:
        angles = [
            (0,0,1), (0,1,0), (1,0,0), 
            (0,1,1), (0,1,-1), (1,0,1), (1,0,-1), (1,1,0), (1,-1,0), 
            (1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1)
        ]
    else:
        angles = [(0, 1), (1, 1), (1, 0), (1, -1)]

    final_offsets = []
    d_seq = distances if isinstance(distances, list) else [distances]
    for d_list in d_seq:
        d_sub = d_list if isinstance(d_list, list) else [d_list]
        for d in d_sub:
            for a in angles:
                final_offsets.append(np.array(a) * d)

    # ... (Step 3: Accumulate P Matrix - SAME AS BEFORE) ...
    P_accum = np.zeros((levels, levels), dtype=np.float64)
    
    weights = np.ones(len(final_offsets))
    if weightingNorm == 'manhattan':
        weights = np.array([1.0 / np.sum(np.abs(o)) for o in final_offsets])
    elif weightingNorm == 'euclidean':
        weights = np.array([1.0 / np.linalg.norm(o) for o in final_offsets])
    elif weightingNorm == 'infinity':
        weights = np.array([1.0 / np.max(np.abs(o)) for o in final_offsets])
    
    weights /= np.sum(weights)

    for offset, w in zip(final_offsets, weights):
        P_accum += w * _glcm_offset(img_q, mask, offset, levels, symmetricalGLCM)

    if P_accum.sum() > 0:
        P_accum /= P_accum.sum()

    return _compute_features_from_matrix(P_accum)


def _glcm_offset(img, mask, offset, levels, symmetric):
    slices_src = []
    slices_dst = []
    for shift in offset:
        if shift > 0:
            slices_src.append(slice(0, -shift))
            slices_dst.append(slice(shift, None))
        elif shift < 0:
            slices_src.append(slice(-shift, None))
            slices_dst.append(slice(0, shift))
        else:
            slices_src.append(slice(None))
            slices_dst.append(slice(None))
            
    src_vals = img[tuple(slices_src)]
    dst_vals = img[tuple(slices_dst)]
    valid = (src_vals > 0) & (dst_vals > 0)
    
    rows = src_vals[valid] - 1
    cols = dst_vals[valid] - 1
    
    P = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(P, (rows, cols), 1)
    
    if symmetric:
        P += P.T
    return P

def _compute_features_from_matrix(P):
    Ng = P.shape[0]
    eps = 2e-16
    
    # Grid indices (1-based)
    i, j = np.indices((Ng, Ng)) + 1 
    
    # Marginal probabilities
    px = np.sum(P, axis=1) # Row sums
    py = np.sum(P, axis=0) # Col sums
    
    # --- FIX 1: Correct Mean Calculation ---
    # Use 1D index vector for means to avoid 2D broadcasting errors
    k_values = np.arange(1, Ng + 1)
    ux = np.sum(k_values * px) 
    uy = np.sum(k_values * py)
    
    # Variances
    # Here broadcasting works correctly because (N,N) * (N,) aligns on columns?
    # No, to be safe, use explicit reshaping or the grid 'i'.
    # i is (Ng, Ng) where rows are constant 1, 2... 
    # px is (Ng,). 
    # (i - ux)**2 * px REQUIRES checking broadcast alignment.
    # It's safer to use the 1D vector:
    sigx2 = np.sum(((k_values - ux)**2) * px)
    sigy2 = np.sum(((k_values - uy)**2) * py)
    sigx = np.sqrt(sigx2)
    sigy = np.sqrt(sigy2)

    # ... (Rest of features use 'i' and 'j' grids which is correct) ...
    
    autocorr = np.sum(i * j * P)
    joint_avg = ux 
    cluster_prom = np.sum(((i + j - ux - uy)**4) * P)
    cluster_shade = np.sum(((i + j - ux - uy)**3) * P)
    cluster_tend = np.sum(((i + j - ux - uy)**2) * P)
    contrast = np.sum(((i - j)**2) * P)
    
    if sigx * sigy == 0:
        correlation = 1.0
    else:
        correlation = (np.sum((i - ux) * (j - uy) * P) / (sigx * sigy))
        
    k_vals_diff = np.abs(i - j)
    px_minus_y = np.bincount(k_vals_diff.ravel(), weights=P.ravel())
    k_indices = np.arange(len(px_minus_y))
    
    diff_avg = np.sum(k_indices * px_minus_y)
    diff_ent = -np.sum(px_minus_y * np.log2(px_minus_y + eps))
    diff_var = np.sum(((k_indices - diff_avg)**2) * px_minus_y)
    
    joint_energy = np.sum(P**2)
    joint_ent = -np.sum(P * np.log2(P + eps))
    
    i_minus_j = np.abs(i - j)
    i_minus_j_sq = i_minus_j**2
    
    Id = np.sum(P / (1 + i_minus_j))
    Idm = np.sum(P / (1 + i_minus_j_sq))
    
# ... inside _compute_features_from_matrix ...

    # MCC (Maximal Correlation Coefficient)
    try:
        px_safe = px.copy(); px_safe[px==0]=1
        py_safe = py.copy(); py_safe[py==0]=1
        
        # Calculate Q matrix
        Q = (P / px_safe[:, None]) @ (P / py_safe[None, :]).T
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(Q)
        
        # Sort and take the second largest magnitude
        # We sort by abs because slight complex noise can happen
        eigenvalues = np.sort(np.abs(eigenvalues))
        
        # The largest eigenval should be approx 1.0 (trivial solution).
        # We want the one just before it.
        if len(eigenvalues) >= 2:
            second_largest = eigenvalues[-2]
            mcc = np.sqrt(second_largest) if second_largest >= 0 else 0.0
        else:
            mcc = 0.0
    except Exception:
        mcc = 0.0

    Idmn = np.sum(P / (1 + (i_minus_j_sq / (Ng**2))))
    Idn = np.sum(P / (1 + (i_minus_j / Ng)))
    
    mask_neq = (i != j)
    inv_var = np.sum(P[mask_neq] / i_minus_j_sq[mask_neq]) if np.any(mask_neq) else 0.0
    
    max_prob = np.max(P)
    
    k_vals_sum = i + j
    # Min sum is 2 (1+1). bincount index 0 corresponds to sum=2.
    px_plus_y = np.bincount(k_vals_sum.ravel() - 2, weights=P.ravel())
    k_indices_sum = np.arange(len(px_plus_y)) + 2 
    
    sum_avg = np.sum(k_indices_sum * px_plus_y)
    sum_ent = -np.sum(px_plus_y * np.log2(px_plus_y + eps))
    
    # Sum Squares (Variance relative to mean)
    # Use 1D logic again for safety
    sum_squares = np.sum(((k_values - ux)**2) * px)
    
    hx = -np.sum(px * np.log2(px + eps))
    hy = -np.sum(py * np.log2(py + eps))
    hxy = joint_ent
    
    p_log_px_py = np.log2(px_safe[i-1] * py_safe[j-1] + eps)
    hxy1 = -np.sum(P * p_log_px_py)
    
    px_py = np.outer(px, py)
    hxy2 = -np.sum(px_py * np.log2(px_py + eps))
    
    imc1 = (hxy - hxy1) / max(hx, hy) if max(hx, hy) != 0 else 0
    imc2_term = np.exp(-2 * (hxy2 - hxy))
    imc2 = np.sqrt(1 - imc2_term) if imc2_term <= 1 else 0

    return {
        'Autocorrelation': autocorr,
        'ClusterProminence': cluster_prom,
        'ClusterShade': cluster_shade,
        'ClusterTendency': cluster_tend,
        'Contrast': contrast,
        'Correlation': correlation,
        'DifferenceAverage': diff_avg,
        'DifferenceEntropy': diff_ent,
        'DifferenceVariance': diff_var,
        'Id': Id, 'Idm': Idm, 'Idmn': Idmn, 'Idn': Idn, 'Imc1': imc1, 'Imc2': imc2,
        'InverseVariance': inv_var,
        'JointAverage': joint_avg,
        'JointEnergy': joint_energy,
        'JointEntropy': joint_ent,
        'MCC': mcc,
        'MaximumProbability': max_prob,
        'SumAverage': sum_avg,
        'SumEntropy': sum_ent,
        'SumSquares': sum_squares
    }


def glcm_units(intensity_unit='HU'):
    """
    Returns units for GLCM features.

    Args:
        intensity_unit (str, optional): The unit of pixel intensity (e.g., 'HU', 'GV', ''). Default is 'HU'.

    Returns:
        dict: Dictionary mapping feature names to their units.

    Example:
        >>> from numpyradiomics import glcm_units
        >>> units = glcm_units(intensity_unit='HU')
        >>> print(units['Contrast'])
        HU^2
    """
    base_unit = intensity_unit
    
    return {
        "Autocorrelation": f"{base_unit}^2",       # sum(i * j * P) -> I * I * 1
        "ClusterProminence": f"{base_unit}^4",     # sum((i+j-u)^4 * P) -> I^4
        "ClusterShade": f"{base_unit}^3",          # sum((i+j-u)^3 * P) -> I^3
        "ClusterTendency": f"{base_unit}^2",       # sum((i+j-u)^2 * P) -> I^2
        "Contrast": f"{base_unit}^2",              # sum((i-j)^2 * P) -> I^2
        "Correlation": "",                         # (Covariance / (std * std)) -> I^2 / I^2 = 1
        "DifferenceAverage": base_unit,            # sum(k * P_diff) -> I * 1
        "DifferenceEntropy": "",                   # Entropy (bits)
        "DifferenceVariance": f"{base_unit}^2",    # Variance of difference -> I^2
        "Id": "",                                  # Inverse Difference (1 / (1 + |i-j|)) -> Dimensionless
        "Idm": "",                                 # Inverse Difference Moment (1 / (1 + (i-j)^2)) -> Dimensionless
        "Idmn": "",                                # Normalized -> Dimensionless
        "Idn": "",                                 # Normalized -> Dimensionless
        "Imc1": "",                                # Correlation measure -> Dimensionless
        "Imc2": "",                                # Correlation measure -> Dimensionless
        "InverseVariance": f"{base_unit}^-2",      # sum(P / |i-j|^2) -> 1 / I^2
        "JointAverage": base_unit,                 # Mean intensity -> I
        "JointEnergy": "",                         # sum(P^2) -> Dimensionless
        "JointEntropy": "",                        # Entropy (bits)
        "MCC": "",                                 # Correlation coeff -> Dimensionless
        "MaximumProbability": "",                  # Probability value -> Dimensionless
        "SumAverage": base_unit,                   # sum(k * P_sum) -> I * 1
        "SumEntropy": "",                          # Entropy (bits)
        "SumSquares": f"{base_unit}^2",            # Variance from mean -> I^2
    }