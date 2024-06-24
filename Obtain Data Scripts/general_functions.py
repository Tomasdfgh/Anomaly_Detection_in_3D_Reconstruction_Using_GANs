import numpy as np
import os
from PIL import Image
import view_3d as v3
import open3d as o3d
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import os

def iterate_files_filter(depth_path, rgb_path, save_path):
    for root, dirs, files in os.walk(depth_path):
        for file in files:

            jpg_file = file[:-4] + '.jpg'

            file_path = os.path.join(root, file)
            rgb_image_path = os.path.join(rgb_path, jpg_file)

            # Open the depth image
            depth_image = Image.open(file_path)

            # Convert the depth image to a numpy array
            depth_pixel_values = np.array(depth_image)

            # Find the coordinates of all non-zero pixel values in the depth image
            non_zero_coords = np.argwhere(depth_pixel_values == 0)

            # Open the RGB image
            rgb_image = Image.open(rgb_image_path)

            # Convert the RGB image to a numpy array
            rgb_pixel_values = np.array(rgb_image)

            # Zero out the corresponding pixel values in the RGB image
            for coord in non_zero_coords:
                y, x = coord
                rgb_pixel_values[y, x] = [0, 0, 0]


            modified_rgb_image = Image.fromarray(rgb_pixel_values)
            modified_rgb_image.save(save_path + '\\' + jpg_file)


def iterate_files_resize(depth_path, rgb_path, depth_save, rgb_save):

    for root, dirs, files in os.walk(depth_path):
        for file in files:

            jpg_file = file[:-4] + '.jpg'

            #Path Directory
            depth_image_path = str(os.path.join(root, file))
            rgb_image_path = str(os.path.join(rgb_path, jpg_file))

            #Destination Directory
            depth_save_dir = str(os.path.join(depth_save, file))
            rgb_save_dir = str(os.path.join(rgb_save, jpg_file))



            normalize_to_8bit(depth_image_path, depth_save_dir)
            normalize_to_8bit(rgb_image_path, rgb_save_dir)


def display_random(depth_path, rbg_path):

    # List all files in the folder
    files = os.listdir(rbg_path)
    
    # Filter out directories (if any)
    files = [f for f in files if os.path.isfile(os.path.join(rbg_path, f))]
    
    # Check if there are any files in the folder
    if not files:
        print(f"No files found in the folder: {rbg_path}")
        return None
    
    # Select a random file
    random_file = random.choice(files)

    print(random_file)

    #random_file = '29.jpg'

    #Build the file paths
    depth_random = os.path.join(depth_path, random_file[:-4] + '.png')
    rgb_random = os.path.join(rbg_path, random_file[:-4] + '.jpg')

    # Set camera parameters
    lookat = np.array([0, 0, 0])
    up = np.array([0, -1, 0])
    front = np.array([0, 0, -1])
    zoom = 0.5

    # Show the first model with specified camera parameters
    pcd = v3.load_and_transform_model(rgb_random, depth_random)
    o3d.visualization.draw_geometries(
        [pcd], 
        lookat=lookat, 
        up=up, 
        front=front, 
        zoom=zoom
    )


def resize_image(input_image_path, output_image_path, size=(192, 108)):
    try:
        with Image.open(input_image_path) as img:
            img = img.resize(size, Image.ANTIALIAS)
            img.save(output_image_path, "JPEG")
    except IOError:
        print(f"Unable to resize image: {input_image_path}")

def find_non_zero_coords_and_color_red(image_path):
    # Open the image
    image = Image.open(image_path)
    image_array = np.array(image)

    # Find the coordinates of all non-zero pixel values
    non_zero_coords = np.argwhere(np.any(image_array != [0, 0, 0], axis=-1))

    return non_zero_coords

def compare_images_and_find_diff(coords, image_path1, image_path2):
    # Open the images
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Convert the images to numpy arrays
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    # Initialize an empty list to store the coordinates with more than 80% difference
    significant_diff_coords = []

    # Iterate through the given coordinates
    for coord in coords:
        y, x = coord

        # Get the pixel values at the given coordinates for both images
        pixel_value1 = image1_array[y, x]
        pixel_value2 = image2_array[y, x]

        diff = abs(pixel_value1 - pixel_value2)

        # Check if the difference is more than 80%
        if diff < 0.1 * max(pixel_value1, pixel_value2):
            significant_diff_coords.append((y, x))

    return significant_diff_coords

def zero_out_rgb_values(coords, image_path):
    # Open the RGB image
    rgb_image = Image.open(image_path)

    # Convert the RGB image to a numpy array
    rgb_pixel_values = np.array(rgb_image)

    # Iterate through the given coordinates and set the RGB values to zero
    for coord in coords:
        y, x = coord
        rgb_pixel_values[y, x] = [0, 0, 0]

    # Convert the modified array back to an image
    modified_rgb_image = Image.fromarray(rgb_pixel_values)

    return modified_rgb_image

def zero_out_depth_values(coords, image_path):
    # Open the RGB image
    rgb_image = Image.open(image_path)

    # Convert the RGB image to a numpy array
    rgb_pixel_values = np.array(rgb_image)

    # Iterate through the given coordinates and set the RGB values to zero
    for coord in coords:
        y, x = coord
        rgb_pixel_values[y, x] = 0

    # Convert the modified array back to an image
    modified_rgb_image = Image.fromarray(rgb_pixel_values)

    return modified_rgb_image


def pix_change(image, pix_value):

    image = np.array(image)
    mask = image  <  pix_value

    # Apply the mask to the image
    image[mask] = 0

    # Save the modified image
    return Image.fromarray(image)


def normalize_to_8bit(image, size=(192, 108)):
    try:

        if image is None:
            raise ValueError("Invalid image provided")

        # Convert PIL image to numpy array
        image_np = np.array(image)

        # Normalize to 8-bit range (0-255)
        img_8bit = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_8bit = cv2.resize(img_8bit, size)
        normalized_image = Image.fromarray(img_8bit)

        return normalized_image

    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def finalize_im(image_depth_path, image_rgb_path, rgb_save_path, depth_save_path):

    #Zero out all background pixel values

    #Table's path. Used to remove the table in every data points
    table_rgb_path = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\rectangle_data\Table\RGB_Filtered\996.jpg"
    table_depth_path = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\rectangle_data\Table\Depth_Images\996.png"

    #Grab the coordinates of images to look at, and then get the coordinates to zero out
    non_zero_coords = find_non_zero_coords_and_color_red(table_rgb_path)
    diff_coords = compare_images_and_find_diff(non_zero_coords, table_depth_path, image_depth_path)

    #Zero out table pixel value
    final_rgb = zero_out_rgb_values(diff_coords, image_rgb_path)
    final_depth = zero_out_depth_values(diff_coords, image_depth_path)

    #Resize and normalize image
    rgb_im = normalize_to_8bit(final_rgb)
    depth_im = normalize_to_8bit(final_depth)

    #Remove noise by removing all pixel values before actual object
    depth_im = pix_change(depth_im, 110)

    #Save Images
    rgb_name = os.path.basename(image_rgb_path)
    depth_name = os.path.basename(image_depth_path)

    rgb_im.save(rgb_save_path + r"//" + rgb_name)
    depth_im.save(depth_save_path + r"//" + depth_name)