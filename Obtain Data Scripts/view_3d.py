import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

def load_and_transform_model(file_jpg, file_png):
    # Read color and depth images
    color_raw = o3d.io.read_image(file_jpg)
    depth_raw = o3d.io.read_image(file_png)

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

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

    return pcd

def render_3d_models(files_jpg, files_png):
    fig, axes = plt.subplots(1, len(files_jpg), figsize=(15, 5))

    for i, (file_jpg, file_png) in enumerate(zip(files_jpg, files_png)):
        # Load and transform model
        pcd = load_and_transform_model(file_jpg, file_png)

        # Create visualizer instance
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add geometry to visualizer
        vis.add_geometry(pcd)

        # Set view control parameters
        vis.get_render_option().point_size = 1.0  # Adjust point size if needed

        # Capture rendered image
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f"temp_render_{i}.png")  # Capture the rendered image
        image = plt.imread(f"temp_render_{i}.png")

        # Display rendered image in Matplotlib
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(f'Model {i + 1}')

        vis.destroy_window()  # Close the visualizer window

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Set camera parameters
    lookat = np.array([0, 0, 0])
    up = np.array([0, -1, 0])
    front = np.array([0, 0, -1])
    zoom = 0.5

    # Show the first model with specified camera parameters
    pcd = load_and_transform_model(r"C:\Users\tomng\Desktop\3D_Detection_Using_GANs\decide_data\RGB_Images\5.jpg", r"C:\Users\tomng\Desktop\3D_Detection_Using_GANs\decide_data\Depth_Images\5.png")
    o3d.visualization.draw_geometries(
        [pcd], 
        lookat=lookat, 
        up=up, 
        front=front, 
        zoom=zoom
    )