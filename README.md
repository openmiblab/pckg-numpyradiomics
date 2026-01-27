
# NumpyRadiomics

A lightweight, pure-Python implementation of radiomic shape features, fully compatible with **PyRadiomics** definitions. Built on top of `numpy`, `scipy`, and `skimage` to minimize heavy dependencies while maintaining accuracy.

## Features

- **Standard Compliance**: Implements the official PyRadiomics definitions for 2D and 3D shape features.
- **Lightweight**: No heavy C++ extensions or SimpleITK dependencies required—just standard scientific Python stacks.


## Installation

```bash
pip install numpyradiomics
```

## Usage

### 1. Basic Shape Extraction

The `shape()` function automatically dispatches to 2D or 3D logic based on the input mask dimensions.

```python
import numpy as np
from numpyradiomics import shape, dro

# 1. Create a synthetic 3D mask (e.g., a 20x10x5 mm cuboid)
spacing = (1.0, 1.0, 1.0)
mask = dro.cuboid(radii_mm=(10.0, 5.0, 2.5), spacing=spacing)

# 2. Extract features
# extend=True (default) adds advanced metrics like Solidity and ConvexHullVolume
features = shape(mask, spacing, extend=True)

print(f"Volume:     {features['VoxelVolume']} mm^3")
print(f"Sphericity: {features['Sphericity']:.4f}")
print(f"Solidity:   {features['Solidity']:.4f}")

```

### 2. Working with Units

You can retrieve a dictionary mapping every feature to its physical unit (e.g., `mm`, `mm^2`, `mm^3` or dimensionless).

```python
from numpyradiomics import shape_units

# Get units for 3D features in centimeters
units = shape_units(dim=3, base_unit='cm')

print(f"MeshVolume unit: {units['MeshVolume']}")    # Output: cm^3
print(f"Elongation unit: {units['Elongation']}")    # Output: (empty string for dimensionless)

```

## Supported Features

### 3D Shape (Standard)

* **Volume**: Mesh Volume, Voxel Volume
* **Surface**: Surface Area, Surface-to-Volume Ratio
* **Dimensions**: Max 3D Diameter, Max 2D Diameter (Slice, Column, Row)
* **Shape Descriptors**: Sphericity, Elongation, Flatness
* **Axes**: Major, Minor, and Least Axis Lengths

### 3D Shape (Extended)

* **Advanced Volume**: Convex Hull Volume, Bounding Box Volume
* **Ratios**: Solidity, Extent
* **Inertia**: Moments of Inertia, Fractional Anisotropy
* **Other**: Maximum Depth (Chebyshev Radius), Longest Caliper Diameter

### 2D Shape

* **Area**: Mesh Surface, Pixel Surface
* **Perimeter**: Perimeter, Perimeter-Surface Ratio
* **Shape Descriptors**: Sphericity, Spherical Disproportion, Elongation
* **Axes**: Major Axis, Minor Axis, Max Diameter

### Texture

All standard texture metrics are available:

* **GLCM**: Gray Level Co-occurrence Matrix
* **GLRLM**: Gray Level Run Length Matrix
* **GLSZM**: Gray Level Size Zone Matrix
* **GLDM**: Gray Level Dependence Matrix
* **NGTDM**: Neighbouring Gray Tone Difference Matrix


## License

Apache 2.0 License
