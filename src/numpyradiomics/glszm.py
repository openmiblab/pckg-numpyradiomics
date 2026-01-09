import numpy as np
from scipy.ndimage import label

def glszm_features(img, mask, binWidth=25, levels=None, connectivity=None):
    roi_mask = mask > 0
    if not np.any(roi_mask):
        raise ValueError("Mask contains no voxels.")

    roi_image = img.copy()
    roi_image[~roi_mask] = 0

    min_val = np.min(roi_image[roi_mask])
    max_val = np.max(roi_image[roi_mask])
    if max_val == min_val:
        raise ValueError("ROI has constant intensity; GLSZM cannot be computed.")

    if levels is None:
        levels = int(np.ceil((max_val - min_val) / binWidth)) + 1

    # Quantize intensities (1-based for PyRadiomics)
    img_quant = np.floor((roi_image - min_val) / binWidth).astype(int) + 1
    img_quant = np.clip(img_quant, 1, levels)

    dims = img.ndim
    if connectivity is None:
        connectivity = 8 if dims == 2 else 26
    structure = _get_connectivity_structure_nd(dims, connectivity)

    max_zone_size = np.sum(roi_mask)
    glszm = np.zeros((levels, max_zone_size), dtype=np.int64)

    for g in range(1, levels + 1):
        mask_g = (img_quant == g) & roi_mask
        labeled, num_features = label(mask_g, structure=structure)
        if num_features == 0:
            continue
        sizes = np.bincount(labeled.ravel())[1:]  # skip background
        for size in sizes:
            glszm[g-1, size-1] += 1  # 0-based index for Python

    glszm = glszm[:, np.any(glszm, axis=0)]
    features = _compute_glszm_features_pyradiomics(glszm, mask.size)
    return features

def _get_connectivity_structure_nd(dims, connectivity):
    if dims == 2:
        return np.ones((3,3), dtype=int) if connectivity == 8 else np.array([[0,1,0],[1,1,1],[0,1,0]], int)
    elif dims == 3:
        if connectivity == 6:
            s = np.zeros((3,3,3), int)
            s[1,1,0]=s[1,1,2]=s[1,0,1]=s[1,2,1]=s[0,1,1]=s[2,1,1]=1
            return s
        elif connectivity == 18:
            s = np.ones((3,3,3), int)
            for idx in [(0,0,0),(0,0,2),(0,2,0),(0,2,2),(2,0,0),(2,0,2),(2,2,0),(2,2,2)]: s[idx]=0
            return s
        else:  # 26
            return np.ones((3,3,3), int)
    else:
        raise ValueError("Input must be 2D or 3D")

def _compute_glszm_features_pyradiomics(P, mask_size):
    """
    Compute 16 Pyradiomics GLSZM features from GLSZM matrix P.
    mask_size: total number of voxels in ROI mask (used for ZonePercentage)
    """
    P = P.astype(np.float64)
    Nz = P.sum()  # total number of zones
    Ng, Ns = P.shape
    P_norm = P / (Nz + 1e-12)

    i = np.arange(1, Ng+1).reshape(-1,1)
    j = np.arange(1, Ns+1).reshape(1,-1)

    Ps = P.sum(axis=0)  # zone size probabilities
    Pg = P.sum(axis=1)  # gray level probabilities

    mean_gray = np.sum(Pg * i.flatten())
    mean_size = np.sum(Ps * j.flatten())

    # Features using Pyradiomics formulas
    f = {}
    f['SmallAreaEmphasis'] = np.sum(P / (j**2)) / Nz
    f['LargeAreaEmphasis'] = np.sum(P * (j**2)) / Nz
    f['GrayLevelNonUniformity'] = np.sum(Pg**2)
    f['GrayLevelNonUniformityNormalized'] = np.sum(Pg**2) / (Nz**2)
    f['ZoneSizeNonUniformity'] = np.sum(Ps**2)
    f['ZoneSizeNonUniformityNormalized'] = np.sum(Ps**2) / (Nz**2)
    f['ZonePercentage'] = Nz / mask_size
    f['LowGrayLevelZoneEmphasis'] = np.sum(P / (i**2)) / Nz
    f['HighGrayLevelZoneEmphasis'] = np.sum(P * (i**2)) / Nz
    f['SmallAreaLowGrayLevelEmphasis'] = np.sum(P / (i**2 * j**2)) / Nz
    f['SmallAreaHighGrayLevelEmphasis'] = np.sum(P * (i**2 / j**2)) / Nz
    f['LargeAreaLowGrayLevelEmphasis'] = np.sum(P * (j**2 / i**2)) / Nz
    f['LargeAreaHighGrayLevelEmphasis'] = np.sum(P * (i**2 * j**2)) / Nz
    f['GrayLevelVariance'] = np.sum(Pg * (i.flatten() - mean_gray)**2)
    f['ZoneSizeVariance'] = np.sum(Ps * (j.flatten() - mean_size)**2)
    f['ZoneEntropy'] = -np.sum(P_norm * np.log2(P_norm + 1e-12))
    return f
