from typing import Optional, Union
import numpy as np
from scipy.stats import skew, kurtosis


from .mod_preproc import _discretize

def firstorder(
    image: np.ndarray, 
    mask: np.ndarray, 
    voxelVolume: float = 1.0, 
    binWidth: float = 25, 
    binCount: Optional[int] = None,
    voxelArrayShift: float = 0.0, 
    extend: bool = True
):
    """
    Compute first-order (intensity-based) statistics for a given image and mask,
    replicating Pyradiomics first-order features.

    Args:
        image (np.ndarray): 3D image array containing voxel intensities.
        mask (np.ndarray): 3D mask array (same shape as image), where non-zero values indicate the ROI.
        voxelVolume (float, optional): Volume of a single voxel (used to scale TotalEnergy). Default is 1.
        binWidth (float, optional): Width of bins for 'Fixed Bin Width' discretization (used for Entropy/Uniformity). Default is 25.
        binCount (int, optional): Number of bins for 'Fixed Bin Count' discretization. 
                                  If specified, overrides binWidth logic. Default is None.
        voxelArrayShift (float, optional): Value to add to intensities before computing Energy/RMS. Default is 0.
        extend (bool, optional): If True, returns extended features not strictly in the PyRadiomics standard set 
                                 (05/95 Percentile, CoeffOfVar, Heterogeneity). Default is True.

    Returns:
        dict: Dictionary containing first-order feature names and their computed values.

            Base Features:
                - **Energy**: Sum of squared voxel intensities.
                - **TotalEnergy**: Energy scaled by voxelVolume.
                - **Entropy**: Histogram-based Shannon entropy.
                - **Minimum**: Minimum intensity.
                - **Maximum**: Maximum intensity.
                - **Mean**: Mean intensity.
                - **Median**: Median intensity.
                - **Range**: Maximum - Minimum.
                - **Variance**: Variance of intensities.
                - **StandardDeviation**: Standard deviation of intensities.
                - **Skewness**: Measure of asymmetry of the distribution.
                - **Kurtosis**: Measure of the "tailedness" of the distribution.
                - **MeanAbsoluteDeviation**: Mean distance of all intensity values from the Mean.
                - **RobustMeanAbsoluteDeviation**: Mean distance of intensities (10-90th percentile) from the Mean.
                - **RootMeanSquared**: Square root of the mean of all the squared intensity values.
                - **10Percentile**: 10th percentile of intensities.
                - **90Percentile**: 90th percentile of intensities.
                - **InterquartileRange**: 75th percentile - 25th percentile.
                - **Uniformity**: Sum of squared histogram probabilities.
            
            Extended Features (if ``extend=True``):
                - **05Percentile**: 5th percentile of intensities.
                - **95Percentile**: 95th percentile of intensities.
                - **CoefficientOfVariation**: Standard Deviation / Mean.
                - **Heterogeneity**: Interquartile Range / Median.

    Example:
        >>> import numpyradiomics as npr
        >>> # Generate a noisy ellipsoid (simulation of a tumor)
        >>> img, mask = npr.dro.noisy_ellipsoid(radii_mm=(20, 10, 5), intensity_range=(20, 80))
        >>> 
        >>> # Compute first order statistics
        >>> feats = npr.firstorder(img, mask)
        >>> 
        >>> print(f"Mean: {feats['Mean']:.2f}")
        Mean: 50.02
        >>> print(f"Range: {feats['Range']:.2f}")
        Range: 59.91
    """

    # Extract the voxels inside the mask
    roi = image[mask > 0].astype(np.float64)
    
    if roi.size == 0:
        raise ValueError("The mask does not contain any voxels.")

    # --- Basic Statistics (Calculated on Continuous/Raw Values) ---
    minimum = np.min(roi)
    maximum = np.max(roi)
    mean = np.mean(roi)
    median = np.median(roi)
    variance = np.var(roi)
    sdev = np.std(roi)
    
    # Energy terms use the shifted array (if shift is provided)
    roi_shifted = roi + voxelArrayShift
    rms = np.sqrt(np.mean(roi_shifted**2))
    energy = np.sum(roi_shifted**2)
    total_energy = voxelVolume * energy 
    
    # Percentiles
    perc05 = np.percentile(roi, 5)
    perc10 = np.percentile(roi, 10)
    perc90 = np.percentile(roi, 90)
    perc95 = np.percentile(roi, 95)
    iqr = np.percentile(roi, 75) - np.percentile(roi, 25)
    
    mad = np.mean(np.abs(roi - mean))  # Mean absolute deviation
    range_val = maximum - minimum
    skewness = skew(roi)
    
    # Fisher=False returns Pearson kurtosis (normal distribution = 3.0), matching PyRadiomics
    kurt = kurtosis(roi, fisher=False)
    
    coefficient_of_variation = sdev / mean if mean != 0 else 0.0
    heterogeneity = iqr / median if median != 0 else 0.0

    # Robust MAD: only voxels between 10th and 90th percentile
    # Note: PyRadiomics definition excludes voxels strictly outside the range (inclusive?)
    # Usually: p10 <= x <= p90
    roi_robust = roi[(roi >= perc10) & (roi <= perc90)]
    if roi_robust.size > 0:
        rmad = np.mean(np.abs(roi_robust - np.mean(roi_robust)))
    else:
        rmad = 0.0

    # --- Histogram-based Features (Calculated on Discretized Values) ---
    # We use the centralized discretize function to ensure consistency
    # This aligns 0-based bins correctly and handles binCount dynamic sizing
    discretized_roi = _discretize(roi, binWidth=binWidth, binCount=binCount)
    
    # Get counts of unique discretized values (bins)
    _, counts = np.unique(discretized_roi, return_counts=True)
    
    # Calculate probabilities
    probs = counts.astype(np.float64) / counts.sum()

    # Entropy: -sum(p * log2(p))
    entropy = -np.sum(probs * np.log2(probs + 2e-16))

    # Uniformity: sum(p^2)
    uniformity = np.sum(probs**2)

    features = {
        "Energy": energy,
        "TotalEnergy": total_energy,
        "Entropy": entropy,
        "Minimum": minimum,
        "Maximum": maximum,
        "10Percentile": perc10,
        "90Percentile": perc90,
        "Mean": mean,
        "Median": median,
        "InterquartileRange": iqr,
        "Range": range_val,
        "MeanAbsoluteDeviation": mad,
        "RobustMeanAbsoluteDeviation": rmad,
        "RootMeanSquared": rms,
        "StandardDeviation": sdev,
        "Skewness": skewness,
        "Kurtosis": kurt,
        "Variance": variance,
        "Uniformity": uniformity,
    }
    
    extended_features = {
        "05Percentile": perc05,
        "95Percentile": perc95,
        "CoefficientOfVariation": coefficient_of_variation,
        "Heterogeneity": heterogeneity,
    }

    if extend:
        features.update(extended_features)

    return features


def firstorder_units(intensity_unit='HU', voxel_unit='mm'):
    """
    Returns units of returned first-order metrics.

    Args:
        intensity_unit (str, optional): Units of signal intensity (e.g., 'HU', 'SUV'). Default is 'HU'.
        voxel_unit (str, optional): Unit of voxel length (e.g., 'mm'). Default is 'mm'.

    Returns:
        dict: Dictionary mapping feature names to their units.

    Example:
        >>> from numpyradiomics import firstorder_units
        >>> units = firstorder_units(intensity_unit='SUV')
        >>> print(units['Energy'])
        SUV^2
    """

    voxelVolume = f"{voxel_unit}^3"
    unit = intensity_unit

    unit_sq = '' if unit=='' else f"{unit}^2"
    if voxelVolume=='':
        unit_sq_vol = unit_sq
    elif unit=='':
        unit_sq_vol = voxelVolume
    else:
        unit_sq_vol = f"{unit_sq}*{voxelVolume}"

    return {
        "Energy": unit_sq,
        "TotalEnergy": unit_sq_vol,
        "Entropy": '',
        "Minimum": unit,
        "Maximum": unit,
        "10Percentile": unit,
        "90Percentile": unit,
        "Mean": unit,
        "Median": unit,
        "InterquartileRange": unit,
        "Range": unit,
        "MeanAbsoluteDeviation": unit,
        "RobustMeanAbsoluteDeviation": unit,
        "RootMeanSquared": unit,
        "StandardDeviation": unit,
        "Skewness": '',
        "Kurtosis": '',
        "Variance": unit_sq,
        "Uniformity": '',
        # Extended
        "05Percentile": unit,
        "95Percentile": unit,
        "CoefficientOfVariation": '',
        "Heterogeneity": '',
    }


