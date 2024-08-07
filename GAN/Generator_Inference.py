#General Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import os
import threading
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image

#Local Files
import loadData as ld
import training as tr
import view_3d as v3
import DC_model as dcm


def show_image(gen, z_dim, denormalize, denormalize_depth, to_pil):

	#Latent Vectors
	z_vector = torch.randn(1, z_dim).to(next(gen.parameters()).device)

	#Generate Data
	with torch.no_grad():
		fake_gen = gen(z_vector).cpu().view(4, 108, 192)


	rgb_tensor = fake_gen[:3, :, :]
	rgb_im = to_pil(denormalize(rgb_tensor))
	rgb_im.show()

	depth_tensor = fake_gen[3:, :, :]
	depth_im = to_pil(denormalize_depth(depth_tensor))
	depth_im.show()

	return rgb_im, depth_im

def show_image_depth(gen, z_dim, denormalize, denormalize_depth, to_pil):

	#Latent Vectors
	z_vector = torch.randn(1, z_dim).to(next(gen.parameters()).device)

	#Generate Data
	with torch.no_grad():
		fake_gen = gen(z_vector).cpu().view(1, 108, 192)

	depth_im = to_pil(denormalize_depth(fake_gen))
	depth_im.show()

	return depth_im


if __name__ == "__main__":

	#Transformers
	to_pil = transforms.ToPILImage()
	denormalize = transforms.Normalize(
		mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
		std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
	)
	denormalize_depth = transforms.Normalize(
	mean=[-0.5 / 0.5],
	std=[1 / 0.5]
	)

	#Hyperparameters
	z_dim = 64

	#Load Model
	Generative_filepath = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\GAN\Rectangle_WGAN2_Traced"
	gen = dcm.Generator(z_dim)
	gen.load_state_dict(torch.load(Generative_filepath, map_location=torch.device('cpu')))

	#rgb_im, depth_im = show_image(gen, z_dim, denormalize, denormalize_depth, to_pil)

	depth_im = show_image_depth(gen, z_dim, denormalize, denormalize_depth, to_pil)

	# Generate image
	red_image = np.zeros((108, 192, 3), dtype=np.uint8)
	red_image[:, :, 0] = 255  # Set the red channel to 255

	v3.load_and_transform_model(red_image, depth_im)

	show_many = False
	if show_many:
		for i in range(4):
			print(i)
			rgb_im, depth_im = show_image(gen, z_dim, denormalize, denormalize_depth, to_pil)
	#v3.load_and_transform_model(rgb_im, depth_im)