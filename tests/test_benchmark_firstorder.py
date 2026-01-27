import pytest
import numpy as np
import SimpleITK as sitk
from radiomics import firstorder as pyrad_firstorder

from numpyradiomics import firstorder

# ==========================================
# 1. Comparison Test
# ==========================================

class TestFirstOrder:

    def setup_method(self):
        # Create a synthetic 20x20x20 volume
        # Gaussian distribution ensures meaningful Skew/Kurtosis
        np.random.seed(42)
        self.image_arr = np.random.normal(loc=100, scale=20, size=(20, 20, 20))
        
        # Create a binary mask (central cube)
        self.mask_arr = np.zeros((20, 20, 20), dtype=np.uint8)
        self.mask_arr[5:15, 5:15, 5:15] = 1
        
        self.spacing = (1.0, 1.0, 1.0)
        self.bin_width = 5.0 # Bin width for discretization

    def run_pyradiomics(self, image, mask, spacing, bin_width):
        """Run official PyRadiomics First Order extraction"""
        img_itk = sitk.GetImageFromArray(image)
        mask_itk = sitk.GetImageFromArray(mask)
        img_itk.SetSpacing(spacing)
        mask_itk.SetSpacing(spacing)
        
        # Initialize extractor with specific binWidth to force discretization
        # for Entropy/Uniformity, while keeping others continuous.
        settings = {'binWidth': bin_width}
        
        # We use the class directly to avoid calculating all other classes
        extractor = pyrad_firstorder.RadiomicsFirstOrder(
            img_itk, mask_itk, **settings
        )
        extractor.enableAllFeatures()
        return extractor.execute()

    def test_firstorder_compliance(self):
        """
        Benchmark your numpy implementation against PyRadiomics.
        """
        # 1. Run PyRadiomics
        pyrad_res = self.run_pyradiomics(
            self.image_arr, self.mask_arr, self.spacing, self.bin_width
        )
        
        # 2. Run NumpyRadiomics
        # Note: PyRadiomics doesn't use voxelVolume for intensity features 
        # except for 'TotalEnergy'. Default voxelVolume=1 is fine if spacing is (1,1,1).
        my_res = firstorder(
            self.image_arr, self.mask_arr, 
            voxelVolume=1.0, 
            binWidth=self.bin_width
        )
        
        print("\n--- First Order Comparison ---")
        
        # List of standard PyRadiomics keys to check
        # We verify your custom keys (CV, Heterogeneity) don't break the loop
        standard_keys = [
            "Energy", "TotalEnergy", "Entropy", "Minimum", "Maximum", 
            "Mean", "Median", "Range", "MeanAbsoluteDeviation", 
            "RobustMeanAbsoluteDeviation", "RootMeanSquared", 
            "StandardDeviation", "Skewness", "Kurtosis", "Variance", "Uniformity"
        ]
        
        for key in standard_keys:
            if key in pyrad_res:
                py_val = pyrad_res[key]
                my_val = my_res[key]
                
                # Tolerances:
                # 1. Standard features: Exact match expected (float precision)
                rtol = 1e-5
                
                # 2. Histogram features (Entropy, Uniformity):
                # Sensitive to "bin jitter" on floating point data (floor precision).
                # We allow 0.5% deviation to account for C++ vs Python binning differences.
                if key in ["Entropy", "Uniformity"]: 
                    rtol = 5e-3  # 0.005

                try:
                    np.testing.assert_allclose(
                        my_val, py_val, rtol=rtol, 
                        err_msg=f"Mismatch in {key}"
                    )
                except AssertionError as e:
                    print(f"FAILURE in {key}: My={my_val} | Py={py_val}")
                    raise e

    def test_percentiles(self):
        """
        Separate test for percentiles as naming conventions differ.
        """
        pyrad_res = self.run_pyradiomics(
            self.image_arr, self.mask_arr, self.spacing, self.bin_width
        )
        my_res = firstorder(self.image_arr, self.mask_arr)
        
        # PyRadiomics keys: "10Percentile", "90Percentile"
        # Your keys: "10Percentile", "90Percentile" (Matches!)
        
        np.testing.assert_allclose(
            my_res["10Percentile"], pyrad_res["10Percentile"], rtol=1e-5
        )
        np.testing.assert_allclose(
            my_res["90Percentile"], pyrad_res["90Percentile"], rtol=1e-5
        )