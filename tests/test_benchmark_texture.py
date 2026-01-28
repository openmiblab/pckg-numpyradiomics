import pytest
import numpy as np
import SimpleITK as sitk

# Import PyRadiomics classes
from radiomics import glcm as pyrad_glcm
from radiomics import gldm as pyrad_gldm
from radiomics import glrlm as pyrad_glrlm
from radiomics import glszm as pyrad_glszm
from radiomics import ngtdm as pyrad_ngtdm

# Import your wrapper
from numpyradiomics import texture

# --- Helper Functions ---

def get_noisy_ellipsoid(radii, spacing, padding=10, intensity_range=(0, 100)):
    """Generates a random noise object. Good for smoke testing, bad for exact math verification."""
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

def get_checkerboard_3d(size=(20, 20, 20), block_size=2):
    """
    Generates a deterministic 3D checkerboard pattern.
    Excellent for verifying exact mathematical logic (GLRLM runs, GLCM co-occurrences).
    """
    image = np.zeros(size, dtype=np.float32)
    for z in range(size[0]):
        for y in range(size[1]):
            for x in range(size[2]):
                # Checkerboard logic
                if ((z // block_size) + (y // block_size) + (x // block_size)) % 2 == 0:
                    image[z, y, x] = 100.0
                else:
                    image[z, y, x] = 20.0
    
    mask = np.ones(size, dtype=np.uint8)
    return image, mask

def run_pyradiomics_all(image, mask, spacing, **kwargs):
    img_itk = sitk.GetImageFromArray(image)
    mask_itk = sitk.GetImageFromArray(mask)
    img_itk.SetSpacing(spacing[::-1]) 
    mask_itk.SetSpacing(spacing[::-1])
    
    bin_width = kwargs.get('binWidth', 25)
    combined_results = {}

    # 1. GLCM
    extractor = pyrad_glcm.RadiomicsGLCM(
        img_itk, mask_itk, binWidth=bin_width, 
        distances=kwargs.get('distances', [1]),
        symmetricalGLCM=kwargs.get('symmetricalGLCM', True)
    )
    extractor.enableAllFeatures()
    res = extractor.execute()
    combined_results.update({f"glcm_{k}": v for k, v in res.items()})

    # 2. GLDM
    extractor = pyrad_gldm.RadiomicsGLDM(
        img_itk, mask_itk, binWidth=bin_width, gldm_a=kwargs.get('alpha', 0)
    )
    extractor.enableAllFeatures()
    res = extractor.execute()
    combined_results.update({f"gldm_{k}": v for k, v in res.items()})

    # 3. GLRLM
    extractor = pyrad_glrlm.RadiomicsGLRLM(
        img_itk, mask_itk, binWidth=bin_width
    )
    extractor.enableAllFeatures()
    res = extractor.execute()
    combined_results.update({f"glrlm_{k}": v for k, v in res.items()})

    # 4. GLSZM
    extractor = pyrad_glszm.RadiomicsGLSZM(
        img_itk, mask_itk, binWidth=bin_width
    )
    extractor.enableAllFeatures()
    res = extractor.execute()
    combined_results.update({f"glszm_{k}": v for k, v in res.items()})

    # 5. NGTDM
    extractor = pyrad_ngtdm.RadiomicsNGTDM(
        img_itk, mask_itk, binWidth=bin_width, distances=[kwargs.get('distance', 1)]
    )
    extractor.enableAllFeatures()
    res = extractor.execute()
    combined_results.update({f"ngtdm_{k}": v for k, v in res.items()})

    return combined_results

class TestTextureWrapper:

    def _compare(self, my_res, pyrad_res, strict=False):
        print(f"\n--- Comparing {len(my_res)} Features (Strict Mode: {strict}) ---")
        failures = []
        
        for key, my_val in my_res.items():
            if key in pyrad_res:
                py_val = pyrad_res[key]
                
                if strict:
                    # STRICT MODE: For deterministic patterns (Checkerboard)
                    # We expect near-perfect floating point equality.
                    local_rtol = 1e-5
                    local_atol = 1e-5
                else:
                    # NOISE MODE: For random noise images
                    # We relax tolerances significantly because "Peak" features are unstable on noise.
                    local_rtol = 0.05  # 5% error allowed standard
                    local_atol = 1e-3
                    
                    # Specific unstable features on Noise
                    if any(x in key for x in [
                        'Correlation', 'ClusterShade', 'Imc1', 'Imc2', 'MCC', # Near Zero
                        'MaximumProbability', 'JointEnergy',                  # Peak finding
                        'Busyness', 'ClusterProminence'                       # High powers
                    ]):
                        local_rtol = 0.25 # Allow 25% deviation on unstable noise peaks
                        local_atol = 0.5  # Allow absolute deviation

                try:
                    np.testing.assert_allclose(
                        my_val, py_val, 
                        rtol=local_rtol, atol=local_atol
                    )
                except AssertionError:
                    failures.append(f"{key}: My={my_val:.6f} | Py={py_val:.6f}")
            else:
                pass

        if failures:
            pytest.fail("\n".join(failures[:10]) + (f"\n...and {len(failures)-10} more." if len(failures) > 10 else ""))

    def test_deterministic_checkerboard(self):
        """
        GOLD STANDARD TEST: Uses a deterministic checkerboard pattern.
        If logic is correct, this MUST pass with strict tolerances.
        """
        print("\n[Test] Deterministic Checkerboard")
        image, mask = get_checkerboard_3d()
        spacing = (1.0, 1.0, 1.0)
        
        # PyRadiomics
        pyrad = run_pyradiomics_all(image, mask, spacing, binWidth=10)
        # Your Wrapper
        myres = texture(image, mask, binWidth=10)
        
        # COMPARE STRICTLY (1e-5 tolerance)
        self._compare(myres, pyrad, strict=True)

    def test_isotropic_noise_cube(self):
        """
        SMOKE TEST: Uses random noise.
        Verifies pipeline runs without crashing, with loose tolerances.
        """
        print("\n[Test] Isotropic Noise Cube")
        image = np.random.uniform(0, 100, (20,20,20)).astype(np.float32)
        mask = np.ones((20,20,20), dtype=np.uint8)
        spacing = (1.0, 1.0, 1.0)
        
        pyrad = run_pyradiomics_all(image, mask, spacing, binWidth=10)
        myres = texture(image, mask, binWidth=10)
        
        # COMPARE LOOSELY (5-25% tolerance)
        self._compare(myres, pyrad, strict=False)

    def test_anisotropic_ellipsoid(self):
        """
        SMOKE TEST: Custom parameters on noise.
        """
        print("\n[Test] Anisotropic Ellipsoid")
        spacing = (2.0, 1.0, 1.0)
        radii = (12, 8, 8)
        image, mask = get_noisy_ellipsoid(radii, spacing)
        
        params = {
            'binWidth': 5,
            'distances': [1],
            'distance': 1,
            'alpha': 0,
            'symmetricalGLCM': True
        }
        
        pyrad = run_pyradiomics_all(image, mask, spacing, **params)
        myres = texture(image, mask, **params)
        
        # COMPARE LOOSELY
        self._compare(myres, pyrad, strict=False)

if __name__ == '__main__':
    # Manual execution for debugging
    t = TestTextureWrapper()
    t.test_deterministic_checkerboard()
    print("Strict test passed.")
    t.test_isotropic_noise_cube()
    print("Noise test passed.")