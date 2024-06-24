#General Imports
import subprocess
import time
import signal
import os
import random
import shutil
import numpy as np
import open3d as o3d

#Local Imports
import view_3d as v3
import pixel_change as pc
import general_functions as gf


def run_cmd_commands(commands, working_dir=None):
    for command in commands:
        print(f"Running Command: {command}")
        process = subprocess.Popen(command, shell=True, cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        try:
            out, err = process.communicate()
        except subprocess.TimeoutExpired:
            out, err = process.communicate()
        
        print(f"Output: {out.decode()}")
        if err:
            print(f"Error: {err.decode()}")

def list_files_in_folder(folder_path):
    try:
        # Check if the folder path exists
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

        # Get all files in the folder
        files_list = os.listdir(folder_path)

        # Filter out only the files (excluding directories)
        files = [file for file in files_list if os.path.isfile(os.path.join(folder_path, file))]

        return files

    except Exception as e:
        print(f"Error occurred while listing files in '{folder_path}': {e}")
        return []


def select_random_items(input_list, num_items=6):
    try:
        # Check if the number of items to select is greater than the list length
        if num_items > len(input_list):
            raise ValueError("Number of items to select exceeds the length of the input list.")

        # Use random.sample to select num_items items from input_list
        selected_items = random.sample(input_list, num_items)
        
        return selected_items
    
    except ValueError as ve:
        print(f"Error: {ve}")
        return []

def move_files_based_on_list(file_names, source_dir, destination_dir):
    try:
        # Check if source and destination directories exist
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory '{source_dir}' not found.")
        
        if not os.path.exists(destination_dir):
            raise FileNotFoundError(f"Destination directory '{destination_dir}' not found.")

        # Iterate through each file name in the list
        for file_name in file_names:
            source_file_path = os.path.join(source_dir, file_name)
            destination_file_path = os.path.join(destination_dir, file_name)
            
            # Check if the file exists in the source directory
            if os.path.exists(source_file_path):
                # Move the file to the destination directory
                shutil.move(source_file_path, destination_file_path)
                print(f"Moved '{file_name}' from '{source_dir}' to '{destination_dir}'.")
            else:
                print(f"File '{file_name}' not found in '{source_dir}'. Skipping.")

    except Exception as e:
        print(f"Error occurred: {e}")

def count_files_in_directory(directory):
    try:
        # Check if the directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory '{directory}' not found.")

        # Initialize a counter for files
        file_count = 0

        # Iterate through all items in the directory
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)

            # Check if the item is a file (not a directory)
            if os.path.isfile(item_path):
                file_count += 1

    except Exception as e:
        print(f"Error occurred: {e}")

    return file_count


def rename_files_in_folder(folder_path, start_number):
    try:
        # Check if the folder path exists
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder '{folder_path}' not found.")

        # Get a list of all files in the folder
        files = os.listdir(folder_path)

        print(files)

        # Initialize a counter for numbering files
        count = start_number

        # Iterate through all files in the folder
        for file_name in files:
            # Construct current file path
            old_file_path = os.path.join(folder_path, file_name)

            # Generate new file name with incremented number
            new_file_name = f"{count}{os.path.splitext(file_name)[1]}"  # Keeps original extension

            # Construct new file path
            new_file_path = os.path.join(folder_path, new_file_name)

            # Rename the file
            os.rename(old_file_path, new_file_path)

            # Increment the counter
            count += 1

            print(f"Renamed: {file_name} -> {new_file_name}")

        print(f"Renaming completed for {len(files)} files.")

    except Exception as e:
        print(f"Error occurred: {e}")

def delete_folder(folder_path):
    try:
        # Check if the folder path exists
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder '{folder_path}' not found.")

        # Use shutil.rmtree to delete the folder and all its contents recursively
        shutil.rmtree(folder_path)

        print(f"Folder '{folder_path}' successfully deleted.")

    except Exception as e:
        print(f"Error occurred: {e}")


def list_filepath_in_folder(folder_path):
    # Check if the provided path is a valid directory
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided path '{folder_path}' is not a valid directory.")
    
    # List all files in the directory
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    
    return file_paths


def remove_png_files(folder_path):

    # Check if the provided path is a valid directory
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided path '{folder_path}' is not a valid directory.")
    
    # List all files in the directory
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")



if __name__ == "__main__":

    #----Recording----#

    # Working directory where k4arecorder.exe is located
    working_dir = r"C:\Program Files\Azure Kinect SDK v1.4.1\tools"

    # List of commands to run
    commands = [
        r'k4arecorder.exe --record-length 1 "C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\log.mkv"'
    ]

    run_cmd_commands(commands, working_dir)

    #----Decode Video into Images----#

    working_dir = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Obtain Data Scripts"

    commands = [
        r'python azure_kinect_mkv_reader.py --input "C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\log.mkv" --output "C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\pre_data"'
    ]

    run_cmd_commands(commands, working_dir)

    #----Generate a Sample of 3D Models----#

    # Example usage:
    folder_path = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\pre_data\color"
    files = list_files_in_folder(folder_path)

    #Select Files at random
    selected_items_jpg = select_random_items(files, 3)
    selected_items_png = selected_items_jpg.copy()

    #
    for i in range(len(selected_items_jpg)):
        selected_items_png[i] = selected_items_jpg[i][:-4] + '.png'

    #Move the .jpg files into the temporary folder
    source_jpg = r'C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\pre_data\color'
    destination_jpg = r'C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\decide_data\RGB_Images'
    move_files_based_on_list(selected_items_jpg, source_jpg, destination_jpg)

    #Move the .png files into the temporary folder
    source_png = r'C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\pre_data\depth'
    destination_png = r'C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\decide_data\Depth_Images'
    move_files_based_on_list(selected_items_png, source_png, destination_png)

    #Count the number of files in the final dataset
    count = count_files_in_directory(r'C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\rectangle_data\Depth_Images')
    print(count)

    #Renaming Files in Depth
    folder_path = r'C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\decide_data\Depth_Images'
    rename_files_in_folder(folder_path, count)

    #Renaming Files in RGB
    folder_path = r'C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\decide_data\RGB_Images'
    rename_files_in_folder(folder_path, count)

    #Delete Original Image Directory
    folder_path = r'C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\pre_data'
    delete_folder(folder_path)

    #Obtaining list of images for displaying
    png_ims = list_filepath_in_folder(destination_png)
    jpg_ims = list_filepath_in_folder(destination_jpg)

    for im in png_ims:
        pc.pix_change(im)

    #Filter out data's background
    gf.iterate_files_filter(r'C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\decide_data\Depth_Images',r'C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\decide_data\RGB_Images', r'C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\decide_data\RGB_Images')

    #Render those 3d models as a point cloud in a subplot
    #v3.render_3d_models(jpg_ims, png_ims)

    remove_png_files(r'C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Obtain Data Scripts')

    for i in range(len(png_ims)):
        gf.finalize_im(png_ims[i], jpg_ims[i], r'C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\decide_data\RGB_Table_Removed', r'C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\decide_data\Depth_Table_Removed')

    # Set camera parameters
    lookat = np.array([0, 0, 0])
    up = np.array([0, -1, 0])
    front = np.array([0, 0, -1])
    zoom = 0.5

    # Show the first model with specified camera parameters
    pcd = v3.load_and_transform_model(jpg_ims[0], png_ims[0])
    o3d.visualization.draw_geometries(
        [pcd], 
        lookat=lookat, 
        up=up, 
        front=front, 
        zoom=zoom
    )