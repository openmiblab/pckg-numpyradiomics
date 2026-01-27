import pytest
import numpy as np
import SimpleITK as sitk
from radiomics import shape as pyrad_shape_3d
from radiomics import shape2D as pyrad_shape_2d

# Import your wrapper
from numpyradiomics import shape

# ==========================================
# 1. Helper Functions (Data Generation)
# ==========================================

def get_ellipsoid_mask(radii, spacing=(1.0, 1.0, 1.0), padding=10):
    """
    Generates a binary mask of a 3D ellipsoid.
    radii: tuple (rz, ry, rx) in mm
    spacing: tuple (sz, sy, sx) in mm
    """
    radii = np.array(radii)
    spacing = np.array(spacing)
    physical_size = radii * 2 + padding
    
    # Grid generation (centered at 0)
    z = np.arange(-physical_size[0]/2, physical_size[0]/2, spacing[0])
    y = np.arange(-physical_size[1]/2, physical_size[1]/2, spacing[1])
    x = np.arange(-physical_size[2]/2, physical_size[2]/2, spacing[2])
    
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    
    # Ellipsoid equation
    mask = ((Z / radii[0])**2 + (Y / radii[1])**2 + (X / radii[2])**2) <= 1.0
    return mask.astype(np.uint8)

def get_ellipse_mask(radii, spacing=(1.0, 1.0), padding=10):
    """
    Generates a binary mask of a 2D ellipse.
    radii: tuple (ry, rx) in mm
    spacing: tuple (sy, sx) in mm
    """
    radii = np.array(radii)
    spacing = np.array(spacing)
    physical_size = radii * 2 + padding
    
    y = np.arange(-physical_size[0]/2, physical_size[0]/2, spacing[0])
    x = np.arange(-physical_size[1]/2, physical_size[1]/2, spacing[1])
    
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    mask = ((Y / radii[0])**2 + (X / radii[1])**2) <= 1.0
    return mask.astype(np.uint8)

def run_pyradiomics_3d(mask_arr, spacing):
    """Run official PyRadiomics 3D extraction"""
    img_itk = sitk.GetImageFromArray(mask_arr)
    img_itk.SetSpacing(spacing[::-1]) # SITK expects (x,y,z)
    
    # Use mask as both image and mask
    extractor = pyrad_shape_3d.RadiomicsShape(img_itk, img_itk)
    extractor.enableAllFeatures()
    return extractor.execute()

def run_pyradiomics_2d(mask_arr, spacing):
    """Run official PyRadiomics 2D extraction"""
    img_itk = sitk.GetImageFromArray(mask_arr)
    img_itk.SetSpacing(spacing[::-1]) # SITK expects (x,y,z)
    
    extractor = pyrad_shape_2d.RadiomicsShape2D(img_itk, img_itk, force2D=True)
    extractor.enableAllFeatures()
    return extractor.execute()

# ==========================================
# 2. Comprehensive Comparison Tests
# ==========================================

class TestNumpyRadiomics:
    
    def _compare_results(self, my_res, pyrad_res, context=""):
        """
        Helper to iterate and compare all keys in the result dictionaries.
        """
        print(f"\n--- Comparison: {context} ---")
        
        # We only check keys that exist in PyRadiomics (our source of truth)
        # Your package might compute extra things (like 'Compactness'), which we skip here.
        common_keys = [k for k in my_res.keys() if k in pyrad_res]
        
        for key in common_keys:
            my_val = my_res[key]
            py_val = pyrad_res[key]
            
            # Define Tolerances:
            # 1. Mesh-based metrics (Surface Area, Volume, Sphericity)
            #    ITK and Scikit-Image use different variants of Marching Cubes/Squares.
            #    A 5% deviation is expected and acceptable.
            if any(x in key for x in ["Surface", "Mesh", "Compactness", "Sphericity", "Perimeter", "Flatness"]):
                rtol = 0.05
            # 2. Voxel/PCA metrics (Voxel Volume, Axis Lengths)
            #    These should match very closely (mathematically identical logic).
            else:
                rtol = 0.01

            # specific check for tiny values to avoid division-by-zero errors in relative checks
            atol = 1e-8

            try:
                np.testing.assert_allclose(
                    my_val, py_val, rtol=rtol, atol=atol,
                    err_msg=f"Mismatch in feature: {key}"
                )
            except AssertionError as e:
                print(f"FAILURE in {key}: My={my_val} | Py={py_val}")
                raise e

    def test_3d_ellipsoid_isotropic(self):
        """Standard 3D Ellipsoid with 1.0mm spacing"""
        spacing = (1.0, 1.0, 1.0)
        radii = (10, 20, 30) 
        mask = get_ellipsoid_mask(radii, spacing)
        
        pyrad_res = run_pyradiomics_3d(mask, spacing)
        my_res = shape(mask, spacing) 
        
        self._compare_results(my_res, pyrad_res, context="3D Isotropic")

    def test_3d_ellipsoid_anisotropic(self):
        """
        3D Ellipsoid with highly ANISOTROPIC spacing.
        This stresses the mesh generation and PCA logic to ensure spacing is applied correctly.
        """
        spacing = (3.0, 1.0, 0.5) 
        radii = (30, 20, 15)      
        mask = get_ellipsoid_mask(radii, spacing)
        
        pyrad_res = run_pyradiomics_3d(mask, spacing)
        my_res = shape(mask, spacing)

        self._compare_results(my_res, pyrad_res, context="3D Anisotropic")

    def test_2d_ellipse_isotropic(self):
        """Standard 2D Ellipse"""
        spacing = (1.0, 1.0)
        radii = (10, 20)
        mask = get_ellipse_mask(radii, spacing)
        
        pyrad_res = run_pyradiomics_2d(mask, spacing)
        my_res = shape(mask, spacing)
        
        self._compare_results(my_res, pyrad_res, context="2D Isotropic")

    def test_2d_ellipse_anisotropic(self):
        """
        2D Ellipse with ANISOTROPIC spacing.
        Verifies 2D perimeter and PCA axis scaling.
        """
        spacing = (2.0, 0.5) # 2mm vertical, 0.5mm horizontal
        radii = (15, 10)
        mask = get_ellipse_mask(radii, spacing)
        
        pyrad_res = run_pyradiomics_2d(mask, spacing)
        my_res = shape(mask, spacing)
        
        self._compare_results(my_res, pyrad_res, context="2D Anisotropic")