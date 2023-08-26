# Import necessary libraries
import cv2
import numpy as np
import sys

# Function to find the longest line in an image
def find_longest_line(image_path, output_image_path, apertureSize):
    # Load the image in grayscale
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image loading was successful
    if img_gray is None:
        print(f"Failed to load image at {image_path}", file=sys.stderr)
        return 0

    # Invert the image if the average pixel value is high
    if np.mean(img_gray) > 127:
        img_gray = cv2.bitwise_not(img_gray)

    # Apply Canny edge detection to detect edges
    edges = cv2.Canny(img_gray, 50, 100, apertureSize)

    # Apply dilation only in the y-direction to enhance vertical features
    kernel = np.array([[0, 1, 0]] * 3, np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=10)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were detected
    if not contours:
        print("No edges were detected in the image.", file=sys.stderr)
        return 0

    # Find the contour with the longest horizontal span
    largest_contour = max(contours, key=lambda contour: max(point[0][0] for point in contour) - min(point[0][0] for point in contour))

    # Compute the convex hull of the largest contour
    hull = cv2.convexHull(largest_contour)

    # Find the longest horizontal distance
    max_distance = 0
    max_points = None

    # Organize points by their y-coordinate for easier analysis
    y_values = {}
    for point in hull:
        x, y = point[0]
        y_values.setdefault(y, []).append(x)

    # Iterate through y-coordinates and find the longest span
    for y, x_values in y_values.items():
        min_x = min(x_values)
        max_x = max(x_values)
        distance = max_x - min_x
        if distance > max_distance:
            max_distance = distance
            max_points = ((min_x, y), (max_x, y))

    # Load the original image
    img = cv2.imread(image_path)

    if max_points is not None:
        # Draw the longest line on the image
        cv2.line(img, max_points[0], max_points[1], (0, 0, 255), 2)
    else:
        print("No horizontal line found in the image.", file=sys.stderr)

    # Save the image with the drawn line
    cv2.imwrite(output_image_path, img)

    return max_distance  # Return the length of the longest line

# Main function to handle command-line arguments
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python measure_scale_in_pixels.py <scale_image_path> <output_image_path> [apertureSize]")
        sys.exit(1)

    # Get command-line arguments
    image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    apertureSize = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    # Call the function to find the longest line and measure its length
    distance = find_longest_line(image_path, output_image_path, apertureSize)

    # Print the measured length of the longest line
    print(f"The scale has a length of {distance} pixels.")
