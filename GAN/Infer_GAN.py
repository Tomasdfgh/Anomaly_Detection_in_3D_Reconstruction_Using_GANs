import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os

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


if __name__ == "__main__":

	#Transformers
	denormalize = transforms.Normalize(
	mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
	std=[1 / 0.5, 1/ 0.5, 1/0.5]
	)
	denormalize_depth = transforms.Normalize(
	mean=[0.5 / 0.5],
	std=[1 / 0.5]
	)
	to_pil = transforms.ToPILImage()

	#Device and latent vector Dimension
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	z_dim = 100

	#Load up the Generator Model
	gen = torch.jit.load(r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\GAN\Models\Spherical\Spherical_WGAN_Traced")

	#Get latent vector passing it into model for inference
	z = torch.randn(50,z_dim, 1, 1).to(device)
	data = gen(z)
	depth = data[:, 3:, :, :]
	rgb = data[:, :3, : , :]



	#Plotting
	plot = False
	if plot:

		# Plot the images in a 2x3 grid
		fig, axes = plt.subplots(3, 6, figsize=(14, 11))

		for i in range(9):

			# Plot RGB image
			ax_rgb = axes[i // 3, (i % 3) * 2]
			ax_rgb.imshow(to_pil(denormalize(rgb[i])))
			ax_rgb.axis('off')  # Hide the axis

			# Plot depth image
			ax_depth = axes[i // 3, (i % 3) * 2 + 1]
			ax_depth.imshow(to_pil(denormalize_depth(depth[i])), cmap='gray')
			ax_depth.axis('off')  # Hide the axis

		plt.tight_layout()
		plt.show()

	#Save
	save = True
	if save:

		rec_rgb = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Generated Samples\Rectangle WGAN\RGB"
		rec_d = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Generated Samples\Rectangle WGAN\Depth"

		cyl_rgb = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Generated Samples\Cylinder WGAN\RGB"
		cyl_d = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Generated Samples\Cylinder WGAN\Depth"

		sph_rgb = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Generated Samples\Sphere WGAN\RGB"
		sph_d = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Generated Samples\Sphere WGAN\Depth"

		save_dir = sph_rgb
		depth_save_dir = sph_d

		count = count_files_in_directory(save_dir)
		for i in range(len(rgb)):
			image = to_pil(denormalize(rgb[i]))
			save_path = os.path.join(save_dir, f"{i + count}.png")
			image.save(save_path)

			depth_im = to_pil(denormalize_depth(depth[i]))
			save_depth = os.path.join(depth_save_dir, f"{i + count}.jpg")
			depth_im.save(save_depth)