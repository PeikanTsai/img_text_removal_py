import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import easyocr

def preprocess_image(input_image_path, output_image_path, output_dir):
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    image = Image.open(input_image_path)

    # Enhance image sharpness and contrast
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)  # Increase sharpness

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)  # Increase contrast

    # Save the enhanced image for debugging
    enhanced_image_path = os.path.join(output_dir, 'enhanced_image.png')
    image.save(enhanced_image_path)

    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Use EasyOCR for text detection
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(np.array(image))

    # Create a mask for inpainting
    mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)

    # Draw the bounding boxes on the mask
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(mask, top_left, bottom_right, 255, -1)

    # Save the mask for debugging
    mask_image_path = os.path.join(output_dir, 'mask.png')
    cv2.imwrite(mask_image_path, mask)

    # Inpaint the original image using the mask
    inpainted_image = cv2.inpaint(image_cv, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Save the inpainted image for debugging
    inpainted_image_path = os.path.join(output_dir, 'inpainted_image.png')
    cv2.imwrite(inpainted_image_path, inpainted_image)

    # Convert the processed OpenCV image back to PIL format
    processed_image = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))

    # Save the processed image
    processed_image.save(output_image_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python preprocess_image.py <input_image> <output_image> <output_dir>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    output_dir = sys.argv[3]

    preprocess_image(input_image_path, output_image_path, output_dir)

