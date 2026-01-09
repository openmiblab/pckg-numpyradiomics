
import numpy as np
import SimpleITK as sitk

# Pyradiomics imports
from radiomics import firstorder, glcm, glszm, glrlm, gldm, ngtdm, shape

# Custom implementations
from numpyradiomics import first_order_features, glcm_features, glszm_features, glrlm_features, gldm_features, ngtdm_features, shape_features_3d, shape_features_2d

# Tolerance for numerical comparison
RTOL = 1e-5



# ------------------- TEST TEXTURE FEATURES ------------------- #
def test_first_order_features():
    img = np.array([[0, 1], [2, 3]], dtype=np.float64)
    mask = np.ones_like(img)

    itk_img = sitk.GetImageFromArray(img.astype(np.float64))
    itk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))

    # Explicit PyRadiomics settings
    settings = {
        "binWidth": 1.0
    }

    # PyRadiomics
    result_py = firstorder.RadiomicsFirstOrder(
        itk_img,
        itk_mask,
        **settings
    ).execute()

    # Custom
    result_custom = first_order_features(
        img,
        mask,
        **settings
    )

    for key in result_py:
        assert np.isclose(
            result_py[key],
            result_custom[key],
            rtol=RTOL
        ), f"Mismatch in {key}"



def test_glcm_features():
    img = np.array([[0, 1], [2, 3]], dtype=np.float64)
    mask = np.ones_like(img)
    
    itk_img = sitk.GetImageFromArray(img.astype(np.float64))
    itk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    
    result_py = glcm.RadiomicsGLCM(itk_img, itk_mask).execute()
    result_custom = glcm_features(img, mask)
    
    for key in result_py:
        assert np.isclose(result_py[key], result_custom[key], rtol=RTOL), f"Mismatch in {key}"


def test_glszm_features():
    img = np.array([[0, 1], [2, 3]], dtype=np.float64)
    mask = np.ones_like(img)

    itk_img = sitk.GetImageFromArray(img.astype(np.float64))
    itk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))

    result_py = glszm.RadiomicsGLSZM(itk_img, itk_mask).execute()
    result_custom = glszm_features(img, mask)

    for key in result_py:
        assert np.isclose(result_py[key], result_custom[key], rtol=1e-6), f"Mismatch in {key}"



def test_glrlm_features():
    img = np.array([[0, 1], [2, 3]], dtype=np.float64)
    mask = np.ones_like(img)
    
    itk_img = sitk.GetImageFromArray(img.astype(np.float64))
    itk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    
    result_py = glrlm.RadiomicsGLRLM(itk_img, itk_mask).execute()
    result_custom = glrlm_features(img, mask, binWidth=1)
    
    for key in result_py:
        assert np.isclose(result_py[key], result_custom[key], rtol=RTOL), f"Mismatch in {key}"


def test_gldm_features():
    img = np.array([[0, 1], [2, 3]], dtype=np.float64)
    mask = np.ones_like(img)
    
    itk_img = sitk.GetImageFromArray(img.astype(np.float64))
    itk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    
    result_py = gldm.RadiomicsGLDM(itk_img, itk_mask).execute()
    result_custom = gldm_features(img, mask, binWidth=1)
    
    for key in result_py:
        assert np.isclose(result_py[key], result_custom[key], rtol=RTOL), f"Mismatch in {key}"


def test_ngtdm_features():
    img = np.array([[0, 1], [2, 3]], dtype=np.float64)
    mask = np.ones_like(img)
    
    itk_img = sitk.GetImageFromArray(img.astype(np.float64))
    itk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    
    result_py = ngtdm.RadiomicsNGTDM(itk_img, itk_mask).execute()
    result_custom = ngtdm_features(img, mask, binWidth=1)
    
    for key in result_py:
        assert np.isclose(result_py[key], result_custom[key], rtol=RTOL), f"Mismatch in {key}"


# ------------------- TEST SHAPE FEATURES ------------------- #
def test_shape_features_2d():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 3:7] = 1
    
    itk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    result_py = shape.RadiomicsShape2D(itk_mask).execute()
    result_custom = shape_features_2d(mask)
    
    for key in result_custom:
        assert np.isclose(result_py[key], result_custom[key], rtol=RTOL), f"Mismatch in {key}"


def test_shape_features_3d():
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:8, 3:7, 1:9] = 1
    
    itk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    voxel_spacing = (1.0, 1.0, 1.0)
    
    result_py = shape.RadiomicsShape3D(itk_mask).execute()
    result_custom = shape_features_3d(mask, voxel_spacing)
    
    for key in result_custom:
        assert np.isclose(result_py[key], result_custom[key], rtol=RTOL), f"Mismatch in {key}"


if __name__=='__main__':
    # test_shape_features_3d()
    # test_shape_features_2d()
    # test_ngtdm_features()
    # test_gldm_features()
    # test_glrlm_features()
    test_glszm_features()
    # test_glcm_features()
    # test_first_order_features()
