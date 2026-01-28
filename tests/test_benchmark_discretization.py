import numpy as np
import unittest
from radiomics import imageoperations
from numpyradiomics.mod_preproc import _discretize

# ==========================================
# Robust Gradient + Noise Benchmark
# ==========================================

class TestRadiomicsReplication(unittest.TestCase):
    
    def setUp(self):
        """
        Creates a dataset that transitions smoothly from low to high intensity (Gradient),
        with added Noise to test robustness near bin boundaries.
        """
        np.random.seed(42)
        n_samples = 2000
        min_val = 0.0
        max_val = 500.0
        
        # 1. The Signal: A Linear Gradient covering the full range
        #    This ensures we test every possible bin index, not just random ones.
        signal = np.linspace(min_val, max_val, n_samples)
        
        # 2. The Noise: Add Gaussian noise
        #    This tests if values near bin edges (e.g., 24.99 vs 25.01) are handled consistently.
        noise = np.random.normal(0, 5.0, n_samples)
        self.target_voxels = signal + noise
        
        # 3. Clip negative values (optional, but realistic for some imaging)
        #    PyRadiomics handles negatives, but let's keep it clean [0, max] or strictly math.
        #    Let's actually ALLOW negatives to test if logic holds for absolute scales (CT).
        
        # 4. CRITICAL: Add Deterministic Edge Cases
        #    We force specific values that sit exactly on bin boundaries for 'binWidth=25'
        edge_cases = np.array([
            0.0,            # Min anchor
            24.999999,      # Just before bin edge
            25.0,           # Exact bin edge
            25.000001,      # Just after bin edge
            50.0, 
            75.0,
            max_val         # The absolute max
        ])
        
        self.target_voxels = np.concatenate([self.target_voxels, edge_cases])

    def test_bin_width_exact_match(self):
        """Validates Fixed Bin Width (Math Formula) with Gradient Data"""
        bin_width = 25.0
        print(f"\nTesting Fixed Bin Width (Width={bin_width}) on Gradient+Noise data...")
        
        # A. Actual PyRadiomics
        pyrad_result, _ = imageoperations.binImage(
            self.target_voxels, 
            parameterMatrixCoordinates=None, 
            binWidth=bin_width
        )
            
        # B. Replication
        my_result = _discretize(self.target_voxels, binWidth=bin_width)
        
        # C. Compare
        # Check standard equality
        np.testing.assert_array_equal(my_result, pyrad_result)
        
        # D. Statistical check (sanity check on the data itself)
        unique_bins = len(np.unique(my_result))
        print(f"   Data Range: [{np.min(self.target_voxels):.2f}, {np.max(self.target_voxels):.2f}]")
        print(f"   Unique Bins Created: {unique_bins}")
        print("✅ SUCCESS: binWidth replication matches exactly.")

    def test_bin_count_exact_match(self):
        """Validates Fixed Bin Count (Histogram+Digitize) with Gradient Data"""
        bin_count = 64
        print(f"Testing Fixed Bin Count (Count={bin_count}) on Gradient+Noise data...")
        
        # A. Actual PyRadiomics
        pyrad_result, _ = imageoperations.binImage(
            self.target_voxels, 
            parameterMatrixCoordinates=None, 
            binCount=bin_count
        )
        
        # B. Replication
        my_result = _discretize(self.target_voxels, binCount=bin_count)
        
        # C. Compare
        np.testing.assert_array_equal(my_result, pyrad_result)
        
        # D. Verify we actually filled the bins (due to gradient)
        #    Ideally, with a perfect gradient, we use all bins 1..N
        generated_bins = np.unique(my_result)
        print(f"   Bins generated: {len(generated_bins)} / {bin_count}")
        
        # Because we added noise and edge cases, we might have slightly different bounds,
        # but the assertion above proves we MATCH pyradiomics exactly.
        print("✅ SUCCESS: binCount replication matches exactly.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)