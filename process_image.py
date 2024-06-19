import sys
import os
import cv2
import numpy as np
from PIL import Image

def detect_shapes_and_lines(input_image_path, output_dir):
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Save the edges image for debugging
    edges_image_path = os.path.join(output_dir, 'edges.png')
    cv2.imwrite(edges_image_path, edges)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    # Draw lines on the image
    line_image = image.copy()
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save the line image for debugging
    line_image_path = os.path.join(output_dir, 'lines.png')
    cv2.imwrite(line_image_path, line_image)

    # Detect contours for shapes
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the image
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    # Save the contour image for debugging
    contour_image_path = os.path.join(output_dir, 'contours.png')
    cv2.imwrite(contour_image_path, contour_image)

    # Save results to SVG
    svg_output_path = os.path.join(output_dir, 'output.svg')
    with open(svg_output_path, 'w') as svg_file:
        svg_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        svg_file.write('<svg height="{}" width="{}" xmlns="http://www.w3.org/2000/svg">\n'.format(image.shape[0], image.shape[1]))
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    svg_file.write('<line x1="{}" y1="{}" x2="{}" y2="{}" style="stroke:rgb(0,255,0);stroke-width:2" />\n'.format(x1, y1, x2, y2))
        for contour in contours:
            points = " ".join(["{},{}".format(point[0][0], point[0][1]) for point in contour])
            svg_file.write('<polygon points="{}" style="fill:none;stroke:rgb(0,255,0);stroke-width:2" />\n'.format(points))
        svg_file.write('</svg>')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_image.py <input_image> <output_dir>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_dir = sys.argv[2]

    detect_shapes_and_lines(input_image_path, output_dir)

