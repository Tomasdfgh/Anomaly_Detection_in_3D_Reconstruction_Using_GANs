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
import model as md
import training as tr
import view_3d as v3


def show_image(gen, z_dim, denormalize, denormalize_depth, to_pil):

	#Latent Vectors
	z_vector = torch.randn(5, z_dim).to(next(gen.parameters()).device)

	#Generate Data
	with torch.no_grad():
		fake_gen = gen(z_vector).cpu()[0].view(4, 108, 192)


	rgb_tensor = fake_gen[:3, :, :]
	rgb_im = to_pil(denormalize(rgb_tensor))
	rgb_im.show()

	depth_tensor = fake_gen[3:, :, :]
	depth_im = to_pil(denormalize_depth(depth_tensor))
	depth_im.show()

	return rgb_im, depth_im

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
	Generative_filepath = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\BIGAN\BIGAN_Gen.pth"
	gen = md.Generator(z_dim)
	gen.load_state_dict(torch.load(Generative_filepath))

	rgb_im, depth_im = show_image(gen, z_dim, denormalize, denormalize_depth, to_pil)
	#v3.load_and_transform_model(rgb_im, depth_im)