
import os
import general_functions as gf

if __name__ == "__main__":

	#rgb_path = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Spherical Data\RGB_Final"
	#depth_path = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Spherical Data\Depth_Final"
	rgb_path = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Spherical Data\RGB_Reduced"
	depth_path = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Spherical Data\Depth_Reduced"
	for root, dirs, files in os.walk(depth_path):
		for file in files:


			jpg_file = file[:-4] + '.jpg'

			depth_image_path = os.path.join(root, file)
			rgb_image_path = os.path.join(rgb_path, jpg_file)

			print("Processing Image: " + str(jpg_file))

			
			gf.reduce_im(depth_image_path, rgb_image_path, r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Spherical Data\RGB_Final" , r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Spherical Data\Depth_Final")