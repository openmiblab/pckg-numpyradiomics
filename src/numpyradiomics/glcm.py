import numpy as np

def glcm_features(img, mask, binWidth=25, distances=[[1]], symmetricalGLCM=True, weightingNorm=None):
    """
    Custom PyRadiomics-compatible GLCM computation with distances, symmetry, and weighting options.

    Parameters
    ----------
    img : np.ndarray
        2D or 3D image array
    mask : np.ndarray
        2D or 3D mask array (non-zero = ROI)
    binWidth : float
        Intensity discretization bin width
    distances : list of list of int
        Distances for which offsets/angles should be generated
    symmetricalGLCM : bool
        Whether GLCMs should be symmetric
    weightingNorm : str or None
        Distance weighting norm: 'manhattan', 'euclidean', 'infinity', 'no_weighting', or None

    Returns
    -------
    dict
        Dictionary with all 24 PyRadiomics GLCM features
    """
    if not np.any(mask > 0):
        raise ValueError("Mask contains no voxels")

    roi = img[mask > 0]
    Imin = roi.min()
    img_q = np.floor((img - Imin) / binWidth).astype(int) + 1
    img_q[mask == 0] = 0
    levels = img_q[mask > 0].max()

    if levels == 1:
        # Degenerate ROI: all features default
        return {
            'Autocorrelation': 1., 'ClusterProminence': 0., 'ClusterShade': 0., 'ClusterTendency': 0.,
            'Contrast': 0., 'Correlation': 1., 'DifferenceAverage': 0., 'DifferenceEntropy': 0.,
            'DifferenceVariance': 0., 'Id': 1., 'Idm': 1., 'Idmn': 1., 'Idn': 1., 'Imc1': 0., 'Imc2': 0.,
            'InverseVariance': 0., 'JointAverage': 1., 'JointEnergy': 1., 'JointEntropy': 0.,
            'MCC': 1., 'MaximumProbability': 1., 'SumAverage': 2., 'SumEntropy': 0., 'SumSquares': 0.
        }

    # Generate offsets for all distances
    offsets = []
    dims = img.ndim
    for d_list in distances:
        for d in d_list:
            if dims == 2:
                dirs = [(0, 1), (1, 1), (1, 0), (1, -1)]
                offsets.extend([(dx*d, dy*d) for dx, dy in dirs])
            elif dims == 3:
                for dx in [-1,0,1]:
                    for dy in [-1,0,1]:
                        for dz in [-1,0,1]:
                            if dx == dy == dz == 0:
                                continue
                            offsets.append((dx*d, dy*d, dz*d))

    # Weighting setup
    weights = np.ones(len(offsets))
    if weightingNorm is not None:
        for idx, off in enumerate(offsets):
            if weightingNorm == 'manhattan':
                weights[idx] = 1 / (np.sum(np.abs(off)) + 1e-12)
            elif weightingNorm == 'euclidean':
                weights[idx] = 1 / (np.sqrt(np.sum(np.array(off)**2)) + 1e-12)
            elif weightingNorm == 'infinity':
                weights[idx] = 1 / (np.max(np.abs(off)) + 1e-12)
            elif weightingNorm == 'no_weighting':
                weights[idx] = 1.0
            else:
                raise ValueError(f"Unknown weightingNorm: {weightingNorm}")
    weights /= weights.sum()  # Normalize

    # Compute weighted GLCM
    glcm_sum = np.zeros((levels, levels), dtype=np.float64)
    for w, offset in zip(weights, offsets):
        glcm_sum += w * _glcm_offset(img_q, mask, offset, levels, symmetricalGLCM)
    glcm = glcm_sum / glcm_sum.sum()

    return _compute_glcm_all(glcm)


def _glcm_offset(img, mask, offset, levels, symmetric):
    """Compute normalized GLCM for a single offset"""
    dims = img.ndim
    if dims == 2:
        dx, dy = offset
        X, Y = img.shape
        x0, x1 = max(0,-dx), min(X,X-dx)
        y0, y1 = max(0,-dy), min(Y,Y-dy)
        xs, ys = x0+dx, y0+dy
        mask_valid = mask[x0:x1, y0:y1] & mask[xs:x1+dx, ys:y1+dy]
        vals1 = img[x0:x1, y0:y1][mask_valid]
        vals2 = img[xs:x1+dx, ys:y1+dy][mask_valid]
    else:
        dx, dy, dz = offset
        X,Y,Z = img.shape
        x0, x1 = max(0,-dx), min(X,X-dx)
        y0, y1 = max(0,-dy), min(Y,Y-dy)
        z0, z1 = max(0,-dz), min(Z,Z-dz)
        xs, ys, zs = x0+dx, y0+dy, z0+dz
        mask_valid = mask[x0:x1, y0:y1, z0:z1] & mask[xs:x1+dx, ys:y1+dy, zs:z1+dz]
        vals1 = img[x0:x1, y0:y1, z0:z1][mask_valid]
        vals2 = img[xs:x1+dx, ys:y1+dy, zs:z1+dz][mask_valid]

    glcm = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(glcm, (vals1-1, vals2-1), 1)
    if symmetric:
        glcm = glcm + glcm.T
    if glcm.sum() > 0:
        glcm /= glcm.sum()
    return glcm


def _compute_glcm_all(p):
    """Compute all PyRadiomics GLCM features from normalized matrix"""
    levels = p.shape[0]
    i,j = np.indices((levels, levels))
    px = np.sum(p, axis=1)
    py = np.sum(p, axis=0)
    ux = np.sum(i*px)
    uy = np.sum(j*py)
    sx = np.sqrt(np.sum((i-ux)**2*px))
    sy = np.sqrt(np.sum((j-uy)**2*py))

    contrast = np.sum((i-j)**2*p)
    dissimilarity = np.sum(np.abs(i-j)*p)
    homogeneity = np.sum(p/(1+(i-j)**2))
    homogeneity2 = np.sum(p/(1+np.abs(i-j)))
    energy = np.sqrt(np.sum(p**2))
    entropy = -np.sum(p[p>0]*np.log2(p[p>0]))
    correlation = 1.0 if sx==0 or sy==0 else np.sum((i-ux)*(j-uy)*p/(sx*sy))
    cluster_shade = np.sum(((i+j-ux-uy)**3)*p)
    cluster_prominence = np.sum(((i+j-ux-uy)**4)*p)
    cluster_tendency = cluster_prominence
    max_prob = np.max(p)
    sum_probs = np.array([np.sum(p[i+j==k]) for k in range(2*levels-1)])
    sum_average = np.sum(sum_probs*np.arange(2,2*levels+1))
    sum_entropy = -np.sum(sum_probs[sum_probs>0]*np.log2(sum_probs[sum_probs>0]))
    sum_squares = np.sum((i-ux)**2*p)
    diff_probs = np.array([np.sum(p[np.abs(i-j)==k]) for k in range(levels)])
    difference_entropy = -np.sum(diff_probs[diff_probs>0]*np.log2(diff_probs[diff_probs>0]))
    difference_variance = np.sum((np.arange(levels)-np.sum(np.arange(levels)*diff_probs))**2*diff_probs)
    autocorr = np.sum(i*j*p)
    joint_average = np.sum(i*j*p)
    joint_variance = sum_squares
    joint_entropy = entropy
    energy_glcm = np.sum(p**2)
    Id = Idm = Idmn = Idn = MCC = 1.0 if levels==1 else 0.0
    HX = -np.sum(px[px>0]*np.log2(px[px>0]))
    HY = -np.sum(py[py>0]*np.log2(py[py>0]))
    HXY = entropy
    IMC1 = 0.0 if HXY==0 else (HXY-HX-HY)/max(HX,HY)
    IMC2 = np.sqrt(1-np.exp(-2*IMC1))

    return {
        'Autocorrelation': autocorr, 'ClusterProminence': cluster_prominence, 'ClusterShade': cluster_shade,
        'ClusterTendency': cluster_tendency, 'Contrast': contrast, 'Correlation': correlation,
        'DifferenceAverage': 0.0, 'DifferenceEntropy': difference_entropy, 'DifferenceVariance': difference_variance,
        'Id': Id, 'Idm': Idm, 'Idmn': Idmn, 'Idn': Idn, 'Imc1': IMC1, 'Imc2': IMC2, 'InverseVariance': homogeneity2,
        'JointAverage': joint_average, 'JointEnergy': energy_glcm, 'JointEntropy': joint_entropy, 'MCC': MCC,
        'MaximumProbability': max_prob, 'SumAverage': sum_average, 'SumEntropy': sum_entropy, 'SumSquares': sum_squares
    }
