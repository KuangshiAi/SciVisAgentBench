# BioImage Archive Dataset Downloader

A Python script for automatically downloading datasets and metadata from the [EMBL BioImage Archive](https://www.ebi.ac.uk/bioimage-archive/).

## Overview

This tool provides automated downloading of scientific imaging datasets from EMBL's BioImage Archive, with comprehensive metadata extraction and anonymization capabilities. It's designed for researchers who need to systematically collect and organize bioimaging datasets.

## Features

- **Automated Dataset Download**: Download complete datasets or specific images by ID
- **Metadata Extraction**: Comprehensive extraction of study information, image dimensions, and technical details
- **Preview Image Download**: Automatically downloads thumbnail previews when available
- **Anonymization**: Rename files and folders to generic identifiers for privacy
- **Flexible Download Modes**: Download files or extract metadata only
- **YAML Metadata**: Structured metadata output in YAML format

## Installation

```bash
pip install requests beautifulsoup4 PyYAML lxml
```

## Quick Start

### Basic Usage

```python
from download_biarchive_dataset import BioImageArchiveDownloader

# Initialize downloader
downloader = BioImageArchiveDownloader()

# Download a specific image
dataset_folder, metadata_file = downloader.download_dataset(
    dataset_url="https://www.ebi.ac.uk/bioimage-archive/galleries/S-BIAD7.html",
    image_id="IM1",  # Specific image ID
    download_files=True
)
```

### List Available Images

```python
# See what images are available for download
available_ids = downloader.list_available_images(dataset_url)
print(f"Available images: {available_ids}")
```

### Metadata Only Mode

```python
# Extract metadata without downloading files
dataset_folder, metadata_file = downloader.download_dataset(
    dataset_url=dataset_url,
    image_id="IM1",
    download_files=False  # Metadata only
)
```

## Dataset Structure

The downloader creates organized folder structures:

```
data/
└── dataset_001/
    ├── dataset_001.yaml          # Metadata file
    ├── dataset_001.tiff          # Anonymized image file
    ├── preview_IM1.jpg           # Preview thumbnail
    └── [other files...]
```

## Metadata Output

The generated YAML metadata includes:

- **Study Information**: Organism, imaging type, license, author
- **Image Details**: Dimensions, filenames, download URLs
- **Content Info**: Total number of images available
- **Download Tracking**: Which files were downloaded, local paths
- **Anonymization Info**: Original vs. anonymized filenames

## Limitations

- **Preview Images Only**: Only images with preview thumbnails are directly accessible
- **Dynamic Loading**: The full list of images (e.g., 1170 images in S-BIAD7) is loaded dynamically via JavaScript and not accessible through simple HTTP requests
- **Available Images**: Typically only 2-10 images per dataset have direct download access

## Example: S-BIAD7 Dataset

The S-BIAD7 dataset contains 1170 images, but only 2 are directly accessible:

- **IM1**: `ExperimentA/Rep1_DIV3_Cortical_r02c04f01.tiff`
- **IM76**: `ExperimentA/Rep1_DIV3_Hippocampal_r02c07f01.tiff`

```python
# Download IM1 from S-BIAD7
downloader.download_dataset(
    "https://www.ebi.ac.uk/bioimage-archive/galleries/S-BIAD7.html",
    image_id="IM1",
    download_files=True
)
```

## Anonymization

The anonymization feature renames files and folders to generic identifiers:

- **Original**: `ExperimentA/Rep1_DIV3_Cortical_r02c04f01.tiff`
- **Anonymized**: `dataset_001.tiff`

This helps protect sensitive information while maintaining dataset organization.

## Error Handling

The script provides clear error messages when:
- Image ID not found
- Network connection issues
- Invalid dataset URLs
- Missing preview images

## About EMBL BioImage Archive

The [EMBL BioImage Archive](https://www.ebi.ac.uk/bioimage-archive/) is a public repository for biological imaging data, providing:

- **Open Access**: CC0 licensed datasets
- **Scientific Data**: High-quality microscopy and imaging data
- **Metadata Rich**: Comprehensive technical and experimental information
- **Research Ready**: Datasets suitable for machine learning and analysis

## Citation

When using datasets from the EMBL BioImage Archive, please cite the original publications and acknowledge the archive:

```
EMBL BioImage Archive: https://www.ebi.ac.uk/bioimage-archive/
```

## Support

For issues with the downloader script, please check:
1. Network connectivity
2. Valid dataset URLs
3. Available image IDs using `list_available_images()`

For questions about the EMBL BioImage Archive, visit: https://www.ebi.ac.uk/bioimage-archive/help/
