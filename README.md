# img_text_removal_py

A project for removing text from images and detecting shapes and lines.

## Description

This project includes two scripts:

1. `preprocess_image.py`: This script uses EasyOCR to detect and remove text from an image.
2. `process_image.py`: This script processes the preprocessed image to detect shapes and lines and outputs the results to an SVG file.

**Note:** This is just a fun project to experiment with and practice using image processing libraries.

## Installation

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Required Libraries

Install the required libraries using pip:

```sh
pip install easyocr opencv-python-headless numpy pillow
```

## Example Workflow

```sh
python preprocess_image.py input_image.png preprocessed_image.png output_dir
python process_image.py preprocessed_image.png output_dir
```
