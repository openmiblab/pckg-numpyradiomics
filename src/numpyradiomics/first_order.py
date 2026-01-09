import numpy as np
from scipy.stats import skew, kurtosis


def first_order_features(input_image, input_mask, voxelVolume=1, binWidth=25, voxelArrayShift=0):
    """
    Compute first-order (intensity-based) statistics for a given image and mask,
    replicating Pyradiomics first-order features.

    Parameters:
        input_image (np.ndarray): 3D image array containing voxel intensities.
        input_mask (np.ndarray): 3D mask array (same shape as input_image), where non-zero values indicate the region of interest (ROI).
        voxelVolume (float, optional): Volume of a single voxel (used to scale total energy). Default is 1.
        binWidth (float, optional): Width of bins for histogram-based features (entropy, uniformity). Default is 1.0.
        voxelArrayShift (float, optional): Value to add to voxel intensities before computing energy/RMS. Useful for negative intensity images. Default is 0.

    Returns:
        dict: Dictionary containing first-order feature names and their computed values:
            - Energy: Sum of squared voxel intensities (after applying voxelArrayShift).
            - TotalEnergy: Energy scaled by voxelVolume.
            - Entropy: Histogram-based Shannon entropy of the ROI.
            - Minimum: Minimum intensity in the ROI.
            - Maximum: Maximum intensity in the ROI.
            - 10Percentile: 10th percentile of the ROI intensities.
            - 90Percentile: 90th percentile of the ROI intensities.
            - Mean: Mean intensity of the ROI.
            - Median: Median intensity of the ROI.
            - InterQuartileRange: 75th percentile minus 25th percentile (IQR).
            - Range: Maximum minus minimum intensity.
            - MeanAbsoluteDeviation: Mean absolute deviation from the mean.
            - RobustMeanAbsoluteDeviation: Mean absolute deviation of voxels between the 10th and 90th percentile.
            - RootMeanSquared: Root mean squared intensity (after applying voxelArrayShift).
            - StandardDeviation: Standard deviation of the ROI.
            - Skewness: Skewness of the intensity distribution.
            - Kurtosis: Kurtosis of the intensity distribution.
            - Variance: Variance of the ROI intensities.
            - Uniformity: Histogram-based uniformity (sum of squared probabilities).
    """

    # Extract the voxels inside the mask
    roi = input_image[input_mask > 0].astype(np.float64)
    
    if roi.size == 0:
        raise ValueError("The mask does not contain any voxels.")

    # Basic statistics
    minimum = np.min(roi)
    maximum = np.max(roi)
    mean = np.mean(roi)
    median = np.median(roi)
    variance = np.var(roi)
    sdev = np.std(roi)
    rms = np.sqrt(np.mean((roi + voxelArrayShift)**2))
    perc05 = np.percentile(roi, 5)
    perc10 = np.percentile(roi, 10)
    perc90 = np.percentile(roi, 90)
    perc95 = np.percentile(roi, 95)
    iqr = np.percentile(roi, 75) - np.percentile(roi, 25)
    mad = np.mean(np.abs(roi - mean))  # Mean absolute deviation
    range_val = maximum - minimum
    skewness = skew(roi)
    kurt = kurtosis(roi, fisher=False)
    energy = np.sum((roi + voxelArrayShift)**2)
    total_energy = voxelVolume * energy  # same as energy in pyradiomics
    coefficient_of_variation = sdev / mean
    heterogeneity = iqr / median

    # Robust MAD: only voxels between 10th and 90th percentile
    roi_robust = roi[(roi >= perc10) & (roi <= perc90)]
    rmad = np.mean(np.abs(roi_robust - np.mean(roi_robust)))

    # Histogram-based features
    discretized = np.floor((roi - minimum) / binWidth).astype(np.int32)
    _, counts = np.unique(discretized, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()

    # Entropy: -sum(p * log2(p))
    entropy = -np.sum(probs * np.log2(probs + 1e-12))

    # Uniformity: sum(p^2)
    uniformity = np.sum(probs**2)

    return {
        "Energy": energy,
        "TotalEnergy": total_energy,
        "Entropy": entropy,
        "Minimum": minimum,
        "Maximum": maximum,
        "05Percentile": perc05,
        "10Percentile": perc10,
        "90Percentile": perc90,
        "95Percentile": perc95,
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
        "CoefficientOfVariation": coefficient_of_variation,
        "Heterogeneity": heterogeneity,
    }

