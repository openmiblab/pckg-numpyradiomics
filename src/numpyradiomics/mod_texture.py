from .mod_glcm import glcm, glcm_units
from .mod_gldm import gldm, gldm_units
from .mod_glrlm import glrlm, glrlm_units
from .mod_glszm import glszm, glszm_units
from .mod_ngtdm import ngtdm, ngtdm_units



def texture(image, mask, binWidth=25, distances=[1], distance=1, alpha=0, levels=None, connectivity=None, symmetricalGLCM=True, weightingNorm=None):
    """
    Wrapper to compute all texture features (GLCM, GLDM, GLRLM, GLSZM, NGTDM) in a single call.
    
    This function aggregates the 5 standard texture matrices, handling the prefixing of keys 
    (e.g., 'Contrast' becomes 'glcm_Contrast', 'gldm_Contrast', etc.) to avoid name collisions.

    Args:
        image (np.ndarray): 3D image array containing voxel intensities.
        mask (np.ndarray): 3D mask array (same shape as image), where non-zero values indicate the ROI.
        binWidth (float, optional): Bin width for discretization (used by all matrices). Default is 25.
        distances (list, optional): List of pixel distances for GLCM. Default is [1].
        distance (int, optional): Integer pixel distance for NGTDM. Default is 1.
        alpha (int, optional): Alpha cutoff for GLDM dependence. Default is 0.
        levels (int, optional): Manual number of levels (overrides binWidth if set). Default is None.
        connectivity (int, optional): Connectivity kernel for GLSZM (e.g., 6, 18, 26). Default is None (26).
        symmetricalGLCM (bool, optional): Whether to symmetrize the GLCM. Default is True.
        weightingNorm (str, optional): GLCM weighting norm ('manhattan', 'euclidean', 'infinity'). Default is None.
        
    Returns:
        dict: Combined dictionary of all texture features (~75 features) with standard prefixes:
            - **glcm_**: Features from Gray Level Co-occurrence Matrix.
            - **gldm_**: Features from Gray Level Dependence Matrix.
            - **glrlm_**: Features from Gray Level Run Length Matrix.
            - **glszm_**: Features from Gray Level Size Zone Matrix.
            - **ngtdm_**: Features from Neighborhood Gray Tone Difference Matrix.

    Example:
        >>> import numpyradiomics as npr
        >>> # Generate a noisy ellipsoid
        >>> img, mask = npr.dro.noisy_ellipsoid(radii_mm=(15, 15, 15), intensity_range=(0, 100))
        >>> 
        >>> # Compute all texture features
        >>> feats = npr.texture(img, mask, binWidth=10)
        >>> 
        >>> print(f"GLCM Contrast: {feats['glcm_Contrast']:.4f}")
        GLCM Contrast: 12.4501
        >>> print(f"GLSZM Zone %:  {feats['glszm_ZonePercentage']:.4f}")
        GLSZM Zone %:  0.8912
    """
    results = {}
    
    # --- 1. GLCM ---
    # Signature: glcm(img, mask, binWidth, distances, symmetricalGLCM, weightingNorm)
    try:
        res = glcm(image, mask, 
                   binWidth=binWidth, 
                   distances=distances, 
                   symmetricalGLCM=symmetricalGLCM, 
                   weightingNorm=weightingNorm)
        results.update({f"glcm_{k}": v for k, v in res.items()})
    except Exception as e:
        print(f"GLCM failed: {e}")

    # --- 2. GLDM ---
    try:
        res = gldm(image, mask, binWidth=binWidth, alpha=alpha, levels=levels)
        results.update({f"gldm_{k}": v for k, v in res.items()})
    except Exception as e:
        print(f"GLDM failed: {e}")

    # --- 3. GLRLM ---
    try:
        res = glrlm(image, mask, binWidth=binWidth, levels=levels)
        results.update({f"glrlm_{k}": v for k, v in res.items()})
    except Exception as e:
        print(f"GLRLM failed: {e}")

    # --- 4. GLSZM ---
    try:
        res = glszm(image, mask, binWidth=binWidth, levels=levels, connectivity=connectivity)
        results.update({f"glszm_{k}": v for k, v in res.items()})
    except Exception as e:
        print(f"GLSZM failed: {e}")

    # --- 5. NGTDM ---
    try:
        res = ngtdm(image, mask, binWidth=binWidth, distance=distance)
        results.update({f"ngtdm_{k}": v for k, v in res.items()})
    except Exception as e:
        print(f"NGTDM failed: {e}")

    return results


def texture_units(intensity_unit='HU'):
    """
    Returns units of returned texture metrics with correct prefixes.

    Args:
        intensity_unit (str, optional): units of signal intensity (e.g., 'HU', 'SUV'). Default is 'HU'.

    Returns:
        dict: Dictionary mapping prefixed feature names to their units.

    Example:
        >>> from numpyradiomics import texture_units
        >>> units = texture_units(intensity_unit='HU')
        >>> print(units['glcm_Contrast'])
        HU^2
        >>> print(units['ngtdm_Coarseness'])
        HU^-1
    """
    units = {}

    # --- 1. GLCM ---
    glcm_u = glcm_units(intensity_unit)
    units.update({f"glcm_{k}": v for k, v in glcm_u.items()})

    # --- 2. GLDM ---
    gldm_u = gldm_units(intensity_unit)
    units.update({f"gldm_{k}": v for k, v in gldm_u.items()})

    # --- 3. GLRLM ---
    glrlm_u = glrlm_units(intensity_unit)
    units.update({f"glrlm_{k}": v for k, v in glrlm_u.items()})

    # --- 4. GLSZM ---
    glszm_u = glszm_units(intensity_unit)
    units.update({f"glszm_{k}": v for k, v in glszm_u.items()})

    # --- 5. NGTDM ---
    ngtdm_u = ngtdm_units(intensity_unit)
    units.update({f"ngtdm_{k}": v for k, v in ngtdm_u.items()})

    return units
