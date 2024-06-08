import cv2
import numpy as np

def pix_change(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if image is successfully loaded
    if image is None:
        print(f"Error: Unable to read image from {image_path}")
    else:

        # Threshold value to set pixels to 0
        threshold_value = 800

        # Create a mask where pixels > threshold_value are set to 0
        mask = image > threshold_value

        # Apply the mask to the image
        image[mask] = 0

        # Display modified image shape and type
        print(f"Modified image shape: {image.shape}")
        print(f"Modified image type: {image.dtype}")

        # Save the modified image
        cv2.imwrite(image_path, image)
