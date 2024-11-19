# Obtain Data Scripts
This directory contain the scripts that will use the RGBD camera to capture and analyze the data points. The process of which will be broken down in this read me file.

## Data Acquisition
To acquire the data, an RGBD camera was employed to capture images of the object from a single angle. This approach simplifies the model's application by requiring it to identify anomalies based on a single snapshot rather than a complete view of the object. The RGBD camera captures both an RGB image, with three color channels, and a depth image of the object. These images are then used to reconstruct a 3D model of the object by converting the data into a point cloud.

For each data point, background information is removed by discarding any pixels with a depth value above 700, ensuring the focus remains on the object itself. Additionally, the surface on which the object rests is filtered out by eliminating pixels where the depth value difference between the empty surface and the surface with the object exceeds 90\%. This threshold indicates the presence of the object, effectively isolating it from the background and underlying surface. Figure 1 shows what the data sample looks like after each data processing step.

It is important to note that while the input data inherently represents a 3D model, the approach leverages RGB and Depth images to simplify the analysis. By focusing on these two-dimensional representations, the complex task of 3D anomaly detection is effectively reduced to an image anomaly detection problem. This method allows for the application of well-established techniques in image processing and machine learning, streamlining the detection of anomalies and making the overall system more efficient and accessible. Moreover, despite this simplification, the approach retains the capacity for 3D applications, as the depth information still encapsulates critical spatial details that can be used to reconstruct or analyze the 3D structure. This dual capability ensures that the system benefits from the robustness and clarity of 2D techniques while preserving the essential 3D characteristics of the data, making it versatile and powerful for various anomaly detection tasks.


<p align="center">
  <img src="https://github.com/user-attachments/assets/186456f2-4674-4ca9-b540-ea072939ad76" alt="Data Acquisition Diagram (2)" width="85%" />
  <br/>
  <i>Figure 1: Example of data acquisition setup for 3D modeling. Starting with the original image with the surface and background, followed by background removed then surface removed. Each of the RGB images shown on the first row has a corresponding Depth image. This process is repeated for the entire dataset to achieve the final 3D model for each data point.</i>
</p>
