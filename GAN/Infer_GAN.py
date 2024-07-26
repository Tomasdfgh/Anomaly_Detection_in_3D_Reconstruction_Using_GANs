import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

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
	gen = torch.jit.load(r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\GAN\Models\Cylindrical\Cylinder_WGAN_Traced")

	#Get latent vector passing it into model for inference
	z = torch.randn(9,z_dim, 1, 1).to(device)
	data = gen(z)
	depth = data[:, 3:, :, :]
	rgb = data[:, :3, : , :]

	# # Plot the images in a 2x3 grid
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