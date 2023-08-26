# Import necessary libraries
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import measure
from scipy.ndimage import convolve
import argparse  # Importing argparse module for command-line arguments

# Define the dilation kernel for morphological operations
DILATION_KERNEL = np.ones((5, 5), np.uint8)

# Function to recognize dark areas in the grayscale image
def recognize_dark_areas(img_gray, threshold=100):
    _, binary_mask = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV)  # Binary thresholding
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    dark_areas_mask = np.zeros_like(binary_mask)  # Initialize a mask for dark areas
    for contour in contours:
        area = cv2.contourArea(contour)  # Compute the area of the contour
        if area > 50:  # If the area is larger than a threshold
            cv2.drawContours(dark_areas_mask, [contour], -1, 255, thickness=cv2.FILLED)  # Fill the area in the mask
    return dark_areas_mask  # Return the mask

# Function to recognize bright areas in the grayscale image
def recognize_bright_areas(img_gray, threshold=155):
    _, binary_mask = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)  # Binary thresholding
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    bright_areas_mask = np.zeros_like(binary_mask)  # Initialize a mask for bright areas
    for contour in contours:
        area = cv2.contourArea(contour)  # Compute the area of the contour
        if area > 50:  # If the area is larger than a threshold
            cv2.drawContours(bright_areas_mask, [contour], -1, 255, thickness=cv2.FILLED)  # Fill the area in the mask
    return bright_areas_mask  # Return the mask

# Function to check if a point is within the bright areas mask
def is_point_in_bright_areas_mask(bright_areas_mask, x, y):
    return bright_areas_mask[y, x] != 0

# Function to check if a point is within the dark areas mask
def is_point_in_dark_areas_mask(dark_areas_mask, x, y):
    return dark_areas_mask[y, x] != 0

# Function to remove small objects (noise) from the skeleton
def remove_small_objects(skel, min_size):
    labels = measure.label(skel)  # Label connected components in the skeleton
    component_sizes = np.bincount(labels.ravel())  # Count the sizes of components
    too_small = component_sizes < min_size  # Identify components that are too small
    too_small_mask = too_small[labels]  # Create a mask for small components
    skel[too_small_mask] = 0  # Remove small components from the skeleton
    return skel

# Function to close small loops in the skeleton using morphological operations
def close_small_loops(skeleton, kernel_size=3):
    skeleton_8bit = (skeleton.astype(np.uint8) * 255)  # Convert boolean skeleton to 8-bit image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))  # Create a rectangular kernel
    closed = cv2.morphologyEx(skeleton_8bit, cv2.MORPH_CLOSE, kernel)  # Perform morphological closing
    return closed

# Function to remove junctions from the skeleton
def remove_junctions(skeleton):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)  # Define a kernel for convolution
    neighbors_count = convolve(skeleton.astype(np.uint8), kernel)  # Convolve the skeleton with the kernel
    junctions = (skeleton & (neighbors_count > 2))  # Identify junction points
    skeleton_cleaned = skeleton & ~junctions  # Remove junction points from the skeleton
    return skeleton_cleaned

# Function to measure gyral size in pixels
def measure_gyral_size_in_pixels(input_image_path, output_image_path, canny_lower, canny_upper, sulci_detection_threshold, bridge_detection_threshold, pruning_size=30):
    img = cv2.imread(input_image_path, 0)  # Read the input image in grayscale
    if img is None:
        raise FileNotFoundError(f"Failed to load image from {input_image_path}")  # Raise an error if image loading fails

    # Recognize dark and bright areas
    dark_areas_mask = recognize_dark_areas(img, sulci_detection_threshold)
    bright_areas_mask = recognize_bright_areas(img, bridge_detection_threshold)

    edges = cv2.Canny(img, canny_lower, canny_upper)  # Apply Canny edge detection
    edges = cv2.dilate(edges, DILATION_KERNEL, iterations=2)  # Dilate the edges

    edges[bright_areas_mask == 255] = 0  # Remove bright areas from the edges

    edges = edges.astype(bool)  # Convert edges to boolean

    skeleton = skeletonize(edges)  # Perform skeletonization
    skeleton = close_small_loops(skeleton, pruning_size)  # Close small loops
    skeleton = skeletonize(skeleton)  # Skeletonize again
    skeleton = remove_junctions(skeleton)  # Remove junctions
    skeleton = remove_small_objects(skeleton, pruning_size)  # Remove small objects

    skeleton = (skeleton * 255).astype(np.uint8)  # Convert skeleton to 8-bit image
    dist_transform = cv2.distanceTransform(cv2.bitwise_not(skeleton), cv2.DIST_L2, 3)  # Compute distance transform

    # Define regions of interest for measuring gyral size
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2
    range_x = img.shape[1] // 3
    range_y = img.shape[0] // 3
    x_coords = np.linspace(center_x - range_x, center_x + range_x, 10)
    y_coords = np.linspace(center_y - range_y, center_y + range_y, 10)
    points = [(int(x), int(y)) for x in x_coords for y in y_coords]

    distances = []
    color_skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)  # Convert skeleton to color image
    color_skeleton[skeleton == 255] = [0, 255, 0]  # Colorize the skeleton

    # Measure distances and mark points on the skeleton
    for point in points:
        if is_point_in_dark_areas_mask(dark_areas_mask, point[0], point[1]):
            continue
        if skeleton[point[1], point[0]] == 255:
            continue
        distance = dist_transform[point[1], point[0]]
        distances.append(distance)
        cv2.circle(color_skeleton, point, int(distance), (25, 25, 0), 1)  # Draw circles indicating distance
        cv2.circle(color_skeleton, point, 1, (55, 0, 55), -1)  # Mark the point

    dark_areas_mask_color = cv2.cvtColor(dark_areas_mask, cv2.COLOR_GRAY2BGR)  # Convert dark areas mask to color
    dark_areas_mask_color[dark_areas_mask == 255] = [0, 50, 0]  # Colorize dark areas

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert input image to color
    blended = cv2.addWeighted(img_color, 0.7, color_skeleton, 0.3, 0)  # Blend input image with skeleton
    blended = cv2.addWeighted(blended, 0.7, dark_areas_mask_color, 0.3, 0)  # Blend with dark areas mask
    cv2.imwrite(output_image_path, blended)  # Save the blended image

    average_distance = sum(distances) / len(distances) if distances else 0  # Compute average distance
    gyral_size = 4 * average_distance  # Compute gyral size

    return gyral_size  # Return the computed gyral size

# Main function to handle command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure gyral size in pixels.")
    parser.add_argument("input_image_path", help="Path to the input brain sample image.")
    parser.add_argument("output_image_path", help="Path to save the output image.")
    parser.add_argument("--canny_lower", type=int, default=45, help="Lower threshold for Canny edge detection. Default: 45")
    parser.add_argument("--canny_upper", type=int, default=55, help="Upper threshold for Canny edge detection. Default: 55 (lower + 10)")
    parser.add_argument("--sulci_detection_threshold", type=int, default=100, help="Threshold for sulci detection. Default: 100")
    parser.add_argument("--bridge_detection_threshold", type=int, default=210, help="Threshold for bridge detection. Default: 210")
    parser.add_argument("--pruning_size", type=int, default=30, help="Pruning size for skeletonization. Default: 30")

    args = parser.parse_args()  # Parse the command-line arguments

    try:
        # Call the gyral size measurement function with provided arguments
        gyral_size = measure_gyral_size_in_pixels(args.input_image_path, args.output_image_path,
                                                   args.canny_lower, args.canny_upper,
                                                   args.sulci_detection_threshold,
                                                   args.bridge_detection_threshold,
                                                   args.pruning_size)
        print(f"The gyral size in pixels: {gyral_size}")  # Print the computed gyral size
    except FileNotFoundError as e:
        print(str(e))  # Print error message if image loading fails
