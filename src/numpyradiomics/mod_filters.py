import numpy as np
import scipy.ndimage
import pywt


def apply_square_numpy(image_arr: np.ndarray) -> np.ndarray:
    return np.square(image_arr)

def apply_square_root_numpy(image_arr: np.ndarray) -> np.ndarray:
    # Clip negative values to 0 to avoid NaNs
    return np.sqrt(np.maximum(image_arr, 0))

def apply_logarithm_numpy(image_arr: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    # Add epsilon to avoid log(0) and clip negatives
    return np.log(np.maximum(image_arr + epsilon, 1e-6))

def apply_exponential_numpy(image_arr: np.ndarray) -> np.ndarray:
    return np.exp(image_arr)

def apply_log_numpy(image_arr: np.ndarray, sigma_mm: float = 1.0, spacing: tuple = (1.0, 1.0, 1.0)) -> np.ndarray:
    """
    Applies Laplacian of Gaussian using SciPy.
    
    Args:
        image_arr: 3D numpy array.
        sigma_mm: Desired sigma in millimeters.
        spacing: Tuple of (z_spacing, y_spacing, x_spacing) in mm.
    """
    # CRITICAL STEP: Convert physical sigma (mm) to pixel sigma
    # We allow anisotropic spacing (different spacing per axis)
    sigma_pixels = [sigma_mm / s for s in spacing]
    
    # Apply the LoG filter
    # SciPy calculates the Laplacian of the Gaussian smoothed image
    return scipy.ndimage.gaussian_laplace(image_arr, sigma=sigma_pixels)



def apply_wavelet_numpy(image_arr: np.ndarray, wavelet_name: str = 'coif1'):
    """
    Applies Stationary Wavelet Transform returning a dict of 8 arrays (for 3D).
    """
    # Apply Stationary Wavelet Transform (SWT)
    # level=1 decomposition
    coeffs = pywt.swtn(image_arr, wavelet_name, level=1, start_level=0)
    
    # coeffs[0] contains the dictionary of subbands
    subbands_dict = coeffs[0]
    
    output_arrays = {}
    
    for key, subband_arr in subbands_dict.items():
        # PyWavelets keys use 'a' (approx) and 'd' (detail).
        # PyRadiomics/Standard convention uses 'L' (Low) and 'H' (High).
        # Mapping: 'a' -> 'L', 'd' -> 'H'
        new_key = key.replace('a', 'L').replace('d', 'H')
        
        # Note: PyWavelets returns keys in the order of axes (Axis 0, Axis 1, Axis 2)
        # which matches the (Z, Y, X) order of the numpy array.
        output_arrays[new_key] = subband_arr
        
    return output_arrays


def apply_gradient_numpy(image_arr: np.ndarray, sigma_mm: float = 1.0, spacing: tuple = (1.0, 1.0, 1.0)) -> np.ndarray:
    """
    Computes Gradient Magnitude using Gaussian derivatives.
    """
    # Convert physical sigma to pixel sigma
    sigma_pixels = [sigma_mm / s for s in spacing]
    
    # Computes the magnitude of the gradient of the image using Gaussian derivatives
    return scipy.ndimage.gaussian_gradient_magnitude(image_arr, sigma=sigma_pixels)