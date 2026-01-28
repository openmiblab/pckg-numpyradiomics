from typing import Optional
import numpy as np
import scipy.ndimage


# Load -> Resample -> Normalize -> Discretize -> Filter -> Features

def resample_image_numpy(image_arr: np.ndarray, 
                         input_spacing: tuple, 
                         target_spacing: tuple = (1.0, 1.0, 1.0), 
                         is_label: bool = False) -> np.ndarray:
    """
    Resamples a 3D image array to a new voxel spacing using spline interpolation.
    
    Args:
        image_arr: 3D numpy array (Z, Y, X).
        input_spacing: Tuple of original spacing (z_mm, y_mm, x_mm).
        target_spacing: Tuple of target spacing (z_mm, y_mm, x_mm).
        is_label: If True, uses Nearest Neighbor interpolation (order=0) 
                  to preserve integer labels. If False, uses Cubic Spline (order=3).
                  
    Returns:
        Resampled 3D numpy array.
    """
    # 1. Calculate the Zoom Factor
    # formula: zoom = original_spacing / target_spacing
    # Example: If moving from 0.5mm -> 1.0mm, we need half as many pixels.
    #          Zoom = 0.5 / 1.0 = 0.5 (Downsampling)
    zoom_factors = [
        input_spacing[0] / target_spacing[0],
        input_spacing[1] / target_spacing[1],
        input_spacing[2] / target_spacing[2]
    ]
    
    # 2. Set Interpolation Order
    # Order 3 (Cubic) is standard for continuous images (CT/MRI intensities)
    # Order 0 (Nearest) is mandatory for masks (labels) to avoid creating new classes (e.g. 1.5)
    order = 0 if is_label else 3
    
    # 3. Apply Zoom
    # We generally do not use 'prefilter' for labels to keep values strict
    resampled_img = scipy.ndimage.zoom(image_arr, zoom=zoom_factors, order=order, prefilter=not is_label)
    
    return resampled_img



def normalize_zscore_numpy(image_arr: np.ndarray, mask_arr: np.ndarray = None) -> np.ndarray:
    """
    Standardizes the image intensities (Z-Score normalization).
    
    Args:
        image_arr: 3D numpy array of image intensities.
        mask_arr: (Optional) Boolean mask array (same shape). 
                  If provided, mean/std are calculated only from masked pixels.
    """
    if mask_arr is not None:
        # Extract only the pixels inside the mask to calculate stats
        # We assume mask_arr is boolean or 0/1
        pixels_in_roi = image_arr[mask_arr > 0]
        
        # Avoid empty mask errors
        if pixels_in_roi.size == 0:
            return image_arr
            
        mean = np.mean(pixels_in_roi)
        std = np.std(pixels_in_roi)
    else:
        # Calculate stats on the whole volume
        mean = np.mean(image_arr)
        std = np.std(image_arr)
        
    # Prevent division by zero if flat image
    if std == 0:
        return image_arr - mean

    # Apply (X - mu) / sigma
    normalized_img = (image_arr - mean) / std
    return normalized_img



def _discretize_image(
    image: np.ndarray, 
    mask: np.ndarray, 
    binWidth: Optional[float] = 25, 
    binCount: Optional[int] = None
) -> np.ndarray:
    """
    Discretizes an image based on the ROI statistics and maps it back to the original image shape.
    
    This wrapper extracts the voxel intensities within the mask, performs discretization
    (using either fixed bin width or fixed bin count), and returns a new 3D array where 
    the background is 0 and the ROI contains integer bin indices starting at 1.

    Args:
        image (np.ndarray): The input image array (e.g., 3D volume).
        mask (np.ndarray): The binary mask array matching the image shape (ROI > 0).
        binWidth (float, optional): The width of bins for 'Fixed Bin Width' discretization. 
                                    Default is 25.
        binCount (int, optional): The desired number of bins for 'Fixed Bin Count' discretization.
                                  If specified, this overrides ``binWidth``. Default is None.

    Returns:
        np.ndarray: An integer array of the same shape as ``image``. 
                    Pixels outside the mask are 0. 
                    Pixels inside the mask are mapped to bins 1..N.
    """
    # 1. Extract ROI pixels
    roi_voxels = image[mask > 0]
    
    # 2. Run Discretization
    # Assuming '_discretize' handles the logic (min subtraction, bin calculation, etc.)
    discretized_roi = _discretize(roi_voxels, binWidth=binWidth, binCount=binCount)
    
    # 3. Map back to spatial grid
    # Initialize with 0 (background)
    img_quantized = np.zeros_like(image, dtype=np.int32)
    img_quantized[mask > 0] = discretized_roi

    return img_quantized

def _discretize(
    roi_voxels: np.ndarray, 
    binWidth: Optional[float] = 25, 
    binCount: Optional[int] = None
) -> np.ndarray:
    """
    Discretizes a collection of voxel intensities (ROI) using either a Fixed Bin Width 
    or Fixed Bin Count strategy. 
    
    This function strictly replicates the discretization logic used by PyRadiomics, 
    including specific edge-case handling and 1-based indexing.

    Parameters
    ----------
    roi_voxels : np.ndarray
        A 1D array containing only the pixel/voxel intensities within the Region of Interest (ROI).
        (e.g. ``image[mask == 1]``).
    binWidth : float, optional
        The size of each bin. Uses an 'Absolute' discretization strategy where bins are 
        aligned to 0. Recommended for quantitative scales (CT, PET). Default is 25.
    binCount : int, optional
        The desired number of bins. Uses a 'Relative' discretization strategy where bins 
        are defined dynamically by the ROI's minimum and maximum values. 
        Recommended for arbitrary scales (MRI, Ultrasound).

    Returns
    -------
    np.ndarray
        An array of discretized integer values (1-based indices) with the same shape as input.

    Raises
    ------
    ValueError
        If neither or both `binWidth` and `binCount` are specified.
    """

    if binCount is not None:
        return _discretize_bincount(roi_voxels, int(binCount))
       
    if binWidth is not None:
        return _discretize_binwidth(roi_voxels, float(binWidth))
         
    raise ValueError("Either 'binWidth' or 'binCount' must be specified.")


def _discretize_binwidth(roi_voxels: np.ndarray, binWidth: float) -> np.ndarray:
    """
    Internal replication of PyRadiomics 'binWidth' (Absolute) logic.
    Formula: floor(X / W) - floor(min(X) / W) + 1
    """
    if roi_voxels.size == 0:
        return roi_voxels
        
    roi_min = np.min(roi_voxels)
    
    # Calculate bins anchored at 0, then offset so the ROI min becomes Bin 1
    discretized = np.floor(roi_voxels / binWidth) - np.floor(roi_min / binWidth) + 1
    
    return discretized.astype(int)


def _discretize_bincount(roi_voxels: np.ndarray, binCount: int) -> np.ndarray:
    """
    Internal replication of PyRadiomics 'binCount' (Relative) logic.
    Logic: Histogram Edges -> Digitize.
    """
    if roi_voxels.size == 0:
        return roi_voxels

    # 1. Calculate Edges using Numpy Histogram
    # PyRadiomics uses standard numpy histogram to determine the ranges
    _, bin_edges = np.histogram(roi_voxels, bins=binCount)
    
    # 2. Fix the Last Edge (The "Max Value" Fix)
    # PyRadiomics manually increments the last edge to ensure the maximum 
    # value is included in the last bin (N), rather than creating a new bin (N+1).
    bin_edges[-1] += 1.0
    
    # 3. Digitize
    # Maps values to bins based on the edges (returns 1-based indices)
    discretized = np.digitize(roi_voxels, bin_edges)
    
    return discretized.astype(int)