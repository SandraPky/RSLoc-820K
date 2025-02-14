# RSLoc-82K Dataset
**For Large-Scale Geo-Localization of Remote Sensing Images**

Welcome to the **RSLoc-82K Dataset**, a large-scale benchmark designed for advancing research in remote sensing image geo-localization. This dataset provides a comprehensive collection of high-resolution remote sensing images, covering diverse terrains and challenging scenarios to support the development and evaluation of robust geo-localization algorithms.
## Dataset Highlights
### Key Features

### Design Goals

## Dataset Structure
RSLoc-82K/
├── train/ # Training set (60,000 images)
│ ├── images/ # Image files (JPEG format)
│ └── metadata.csv # Metadata (coordinates, time, sensor, etc.)
├── test/ # Test set (22,000 images)
│ ├── queries/ # Query images (complex scenes)
│ └── references/ # Reference database (continuous coverage)
└── evaluation_scripts/ # Evaluation toolkit
├── topk_accuracy.py # Top-K accuracy calculation
└── geospatial_utils/ # Geospatial analysis tools


---

## Download and Usage
### Step 1: Download the Dataset
```bash
# Using Git LFS (recommended)
git clone https://github.com/yourusername/RSLoc-82K-dataset.git
cd RSLoc-82K-dataset
git lfs pull

### Step 2: Quick Validation
import pandas as pd
from PIL import Image

# Load metadata
metadata = pd.read_csv("RSLoc-82K/train/metadata.csv")
sample = metadata.iloc[0]

# Visualize an example
image = Image.open(f"RSLoc-82K/train/images/{sample['image_id']}.jpg")
print(f"Coordinates: {sample['latitude']}, {sample['longitude']}")
image.show()

Contact
For questions or collaboration opportunities, please contact:
📧 your.email@university.edu
🐛 Open a GitHub Issue

