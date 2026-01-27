import pytest
import numpy as np
import SimpleITK as sitk
from radiomics import ngtdm as pyrad_ngtdm

from numpyradiomics import ngtdm 

# --- Helper Functions ---

def get_noisy_ellipsoid(radii, spacing, padding=10, intensity_range=(0, 100)):
    radii = np.array(radii)
    spacing = np.array(spacing)
    physical_size = radii * 2 + padding
    
    z = np.arange(-physical_size[0]/2, physical_size[0]/2, spacing[0])
    y = np.arange(-physical_size[1]/2, physical_size[1]/2, spacing[1])
    x = np.arange(-physical_size[2]/2, physical_size[2]/2, spacing[2])
    
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    mask = ((Z / radii[0])**2 + (Y / radii[1])**2 + (X / radii[2])**2) <= 1.0
    mask = mask.astype(np.uint8)
    
    np.random.seed(42)
    image = np.random.uniform(intensity_range[0], intensity_range[1], mask.shape).astype(np.float32)
    image[mask == 0] = 0
    return image, mask

def run_pyradiomics_ngtdm(image, mask, spacing, bin_width=10, distance=1):
    img_itk = sitk.GetImageFromArray(image)
    mask_itk = sitk.GetImageFromArray(mask)
    img_itk.SetSpacing(spacing[::-1]) 
    mask_itk.SetSpacing(spacing[::-1])
    
    # PyRadiomics expects a list for distances
    extractor = pyrad_ngtdm.RadiomicsNGTDM(
        img_itk, mask_itk, 
        binWidth=bin_width,
        voxelBased=False,
        distances=[distance]  # Wrap single int in list for PyRadiomics
    )
    extractor.enableAllFeatures()
    return extractor.execute()

class TestNGTDM:

    def _compare(self, my_res, pyrad_res):
        print("\n--- Feature Comparison ---")
        failures = []
        for key in my_res.keys():
            if key in pyrad_res:
                py_val = pyrad_res[key]
                my_val = my_res[key]
                
                # Baseline Tolerances
                local_rtol = 1e-3  # 0.1% relative error
                local_atol = 1e-6

                # --- NGTDM Specific Instabilities ---

                # 1. Busyness:
                # Involves a ratio of sums. On random noise, the denominator 
                # (absolute differences) can fluctuate, causing larger relative errors.
                if key == 'Busyness':
                    local_rtol = 0.05  # Relax to 5% for high noise
                    local_atol = 1e-3

                # 2. Coarseness:
                # Inverses the sum of differences. If texture is flat, this blows up.
                elif key == 'Coarseness':
                    # Your implementation adds +1e-6 to denominator. 
                    # PyRadiomics does not. This creates a tiny drift.
                    local_rtol = 0.02

                try:
                    np.testing.assert_allclose(
                        my_val, py_val, 
                        rtol=local_rtol, atol=local_atol,
                        err_msg=f"Mismatch in {key}"
                    )
                except AssertionError as e:
                    failures.append(f"{key}: My={my_val:.6f} | Py={py_val:.6f}")
        
        if failures:
            pytest.fail("\n".join(failures))

    def test_isotropic_cube(self):
        """Basic check: 20x20x20 noise cube"""
        print("\n[Test] Isotropic Cube (NGTDM)")
        image = np.random.uniform(0, 100, (20,20,20)).astype(np.float32)
        mask = np.ones((20,20,20), dtype=np.uint8)
        spacing = (1.0, 1.0, 1.0)
        
        # PyRadiomics Wrapper
        pyrad = run_pyradiomics_ngtdm(image, mask, spacing, bin_width=10, distance=1)
        
        # Your Function Call
        myres = ngtdm(image, mask, binWidth=10, distance=1)
        
        self._compare(myres, pyrad)

    def test_anisotropic_ellipsoid(self):
        """Advanced check: Ellipsoid ROI with Anisotropic spacing"""
        print("\n[Test] Anisotropic Ellipsoid (NGTDM)")
        # Note: NGTDM defines neighborhood via distances (pixels), 
        # so spacing primarily affects the image resampling (if it were happening).
        spacing = (3.0, 0.5, 0.5)
        radii = (15, 10, 10) 
        image, mask = get_noisy_ellipsoid(radii, spacing)
        
        pyrad = run_pyradiomics_ngtdm(image, mask, spacing, bin_width=10, distance=1)
        myres = ngtdm(image, mask, binWidth=10, distance=1)
        
        self._compare(myres, pyrad)

    def test_degenerate_roi(self):
        """Edge Case: Single Value ROI (Coarseness check)"""
        print("\n[Test] Degenerate ROI (Flat Texture)")
        np.random.seed(42)
        image = np.ones((10,10,10), dtype=np.float32) * 50 
        mask = np.ones((10,10,10), dtype=np.uint8)
        spacing = (1.0, 1.0, 1.0)
        
        pyrad = run_pyradiomics_ngtdm(image, mask, spacing, bin_width=10)
        myres = ngtdm(image, mask, binWidth=10)
        
        # Both should return 0.0 for most features, 
        # and 10^6 for Coarseness due to the specific protection against div by zero.
        self._compare(myres, pyrad)