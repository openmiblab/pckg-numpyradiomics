import pytest
import numpy as np
from scipy.ndimage import center_of_mass
from numpyradiomics import dro

class TestDRO:

    def _get_bbox_physical_size(self, mask, spacing):
        """Helper to measure the physical size of the mask's bounding box."""
        coords = np.argwhere(mask)
        min_idx = coords.min(axis=0)
        max_idx = coords.max(axis=0)
        
        # Extent in pixels (include the endpoint pixel)
        extent_px = max_idx - min_idx + 1
        
        # Extent in mm
        extent_mm = extent_px * np.array(spacing)
        return extent_mm

    def test_cuboid_dimensions(self):
        """Verify the cuboid mask matches the requested physical size."""
        radii = (20, 10, 5) # mm
        spacing = (2.0, 1.0, 0.5) # Anisotropic spacing
        
        image, mask = dro.cuboid(radii_mm=radii, spacing=spacing, padding_mm=10)
        
        # 1. Check Intensity
        assert np.all(image[mask == 1] == 100.0), "Cuboid interior should be 100.0"
        assert np.all(image[mask == 0] == 0.0), "Background should be 0.0"

        # 2. Check Bounding Box Size
        # Expected total size = 2 * radii
        expected_size = np.array(radii) * 2
        measured_size = self._get_bbox_physical_size(mask, spacing)
        
        # Tolerance: The measured size can vary by up to 1 voxel dimension due to discretization
        tolerance = np.array(spacing) 
        
        np.testing.assert_allclose(
            measured_size, expected_size, 
            atol=np.max(tolerance), 
            err_msg="Cuboid physical dimensions incorrect"
        )
        
        # 3. Check Volume (Analytic vs Voxel count)
        # For a cuboid aligned with grid, this should be very precise
        # Volume = (2*rx) * (2*ry) * (2*rz)
        analytic_vol = (radii[0]*2) * (radii[1]*2) * (radii[2]*2)
        voxel_vol = np.prod(spacing)
        calculated_vol = np.sum(mask) * voxel_vol
        
        # Allow small deviation for edge inclusion logic
        assert abs(calculated_vol - analytic_vol) / analytic_vol < 0.05

    def test_ellipsoid_volume_and_center(self):
        """Verify ellipsoid volume and centering."""
        radii = (10, 10, 10) # Sphere
        spacing = (1.0, 1.0, 1.0)
        
        image, mask = dro.ellipsoid(radii_mm=radii, spacing=spacing)

        # 1. Centering Check
        # The center of mass of the mask should be at the center of the array
        center_mask = np.array(center_of_mass(mask))
        center_array = np.array(mask.shape) / 2.0 - 0.5 # 0-based index center
        
        np.testing.assert_allclose(
            center_mask, center_array, 
            atol=0.5, 
            err_msg="Ellipsoid is not centered in the array"
        )

        # 2. Volume Check
        # Analytic Volume = 4/3 * pi * r1 * r2 * r3
        analytic_vol = (4/3) * np.pi * np.prod(radii)
        
        voxel_vol = np.prod(spacing)
        calculated_vol = np.sum(mask) * voxel_vol
        
        # Discretization error for spheres is usually < 2-3%
        error_margin = abs(calculated_vol - analytic_vol) / analytic_vol
        print(f"\nEllipsoid Volume Error: {error_margin:.4%}")
        assert error_margin < 0.03, f"Volume deviation {error_margin} too high"

    def test_noisy_ellipsoid_statistics(self):
        """Verify noise generation works inside the ROI."""
        radii = (15, 10, 5)
        spacing = (1.0, 1.0, 1.0)
        intensity_range = (20, 80)
        
        image, mask = dro.noisy_ellipsoid(
            radii_mm=radii, 
            spacing=spacing, 
            intensity_range=intensity_range
        )
        
        # 1. Check Masking
        # Outside mask should be strictly 0
        assert np.all(image[mask == 0] == 0), "Noise leaked outside the mask"
        
        # 2. Check Range
        # Inside mask should be within range
        roi_pixels = image[mask == 1]
        assert roi_pixels.min() >= intensity_range[0]
        assert roi_pixels.max() <= intensity_range[1]
        
        # 3. Check it's actually noisy (not constant)
        assert np.std(roi_pixels) > 0, "Image inside ROI is constant (no noise)"
        
        # Mean should be roughly the middle of the range (uniform dist)
        expected_mean = np.mean(intensity_range)
        assert abs(np.mean(roi_pixels) - expected_mean) < 2.0

    def test_grid_resolution_invariance(self):
        """
        Critical Test: Ensure physical size stays consistent 
        even when we change voxel resolution (spacing).
        """
        radii = (30, 10, 5)
        
        # Case A: Coarse Resolution (2mm)
        _, mask_coarse = dro.cuboid(radii_mm=radii, spacing=(2.0, 2.0, 2.0))
        vol_coarse = np.sum(mask_coarse) * (2.0**3)
        
        # Case B: Fine Resolution (0.5mm)
        _, mask_fine = dro.cuboid(radii_mm=radii, spacing=(0.5, 0.5, 0.5))
        vol_fine = np.sum(mask_fine) * (0.5**3)
        
        # The physical volumes should be nearly identical
        # (Difference solely due to surface discretization)
        diff = abs(vol_coarse - vol_fine)
        relative_diff = diff / vol_fine
        
        print(f"\nResolution Check - Coarse Vol: {vol_coarse}, Fine Vol: {vol_fine}")
        assert relative_diff < 0.05, "Physical volume changed significantly with resolution"