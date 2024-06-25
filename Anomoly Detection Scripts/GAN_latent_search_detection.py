#General Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
import open3d as o3d
import random
import copy

#Local Files
import GAN_model as gm
import view_3d as v3
import Generator_Inference as gi
import loadData as ld

def latent_search(lr_z, lr_G, n_seed, k, z_dim, data, gen, batch_size, Generative_filepath):

	#Get the initial model weights
	init_weight = copy.deepcopy(gen.state_dict())

	#Get the Array of latent Vectors
	z_list = [torch.randn(batch_size, z_dim, requires_grad = True) for _ in range(n_seed)]

	#initialize a hashmap to store model's final weights
	mod_weights = {}

	for j in range(n_seed):

		#Load up the model
		gen.load_state_dict(init_weight)

		#Define the optimizers
		opt_gen = optim.Adam(gen.parameters(), lr = lr_G)
		opt_z = optim.Adam([z_list[j]], lr = lr_z)

		for t in range(k):

			#Obtaining G(z)
			gen_z = gen(z_list[j])[torch.randint(0, batch_size, (1,)).item()]
			loss = F.mse_loss(gen_z.view(4,108,192), data[0])

			#zeroing out gradients for both z and gen
			opt_gen.zero_grad()
			opt_z.zero_grad()

			#Perform Loss backprop
			loss.backward(retain_graph = True)

			#Update the gen and z
			opt_gen.step()
			opt_z.step()

		mod_weights[j] = copy.deepcopy(gen.state_dict())

	#Calculate the Final Loss value from all the different seeds
	total_loss = 0
	for j in range(n_seed):

		#Create model with final weights
		gen.load_state_dict(mod_weights[j])

		gen_z = gen(z_list[j])[torch.randint(0, batch_size, (1,)).item()]
		loss = F.mse_loss(gen_z.view(4, 108,192), data[0])

		total_loss += loss.detach().item()

	return (1/n_seed) * total_loss



if __name__ == "__main__":

	#Link to Data
	rgb_link = r'C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\rectangle_data\RGB_Final'
	depth_link = r'C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\rectangle_data\Depth_Final'
	
	#Hyperparameters
	lr_z = 1e-4
	lr_G = 3e-4
	n_seed = 20
	k = 20
	z_dim = 64
	batch_size = 32

	#Transformer
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])

	#Setting up the Data
	dataset = ld.load_data(rgb_link, depth_link, [])
	ImageSet = list(ld.ConvertData(dataset, transform = transform))
	data = random.sample(ImageSet, 1)

	#Load up the Generators
	gen = gm.Generator(z_dim)

	#Model Filepath
	Generative_filepath = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\GAN\GAN_Generative_Dense.pth"

	#Load up the model
	gen.load_state_dict(torch.load(Generative_filepath))

	loss = latent_search(lr_z, lr_G, n_seed, k, z_dim, data, gen, batch_size, Generative_filepath)

	print(loss)