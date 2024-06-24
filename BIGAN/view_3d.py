import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

def pil_to_open3d_image(pil_image):
    # Convert PIL image to numpy array
    np_array = np.array(pil_image)

    # Convert numpy array to Open3D Image
    open3d_image = o3d.geometry.Image(np_array)

    return open3d_image


def load_and_transform_model(color_image, depth_image):
    # Create RGBD image
    color_image = pil_to_open3d_image(color_image)
    depth_image = pil_to_open3d_image(depth_image)

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image,
        depth_image,
        depth_scale=1000.0,  # Adjust depth scale if necessary
        depth_trunc=3.0,     # Adjust depth truncation if necessary
        convert_rgb_to_intensity=True
    )
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    )

    # Transform point cloud if necessary
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])


    # Set camera parameters
    lookat = np.array([0, 0, 0])
    up = np.array([0, -1, 0])
    front = np.array([0, 0, -1])
    zoom = 0.5

    o3d.visualization.draw_geometries(
        [pcd], 
        lookat=lookat, 
        up=up, 
        front=front, 
        zoom=zoom
    )