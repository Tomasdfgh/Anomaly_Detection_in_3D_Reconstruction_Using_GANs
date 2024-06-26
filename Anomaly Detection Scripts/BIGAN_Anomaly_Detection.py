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
import BIGAN_model as bm
import view_3d as v3
import Generator_Inference as gi
import loadData as ld


def reconstruction_loss(enc, gen, data, batch_size):

	#Convert data into batch of the exact same data points
	data_use = copy.deepcopy(data[0].view(-1).unsqueeze(0).repeat(batch_size,1))

	#Get latent vector of the data from the encoder
	z_prime = enc(data_use)

	#Get G(E(x_prime))
	Gz_prime = gen(z_prime)[0]

	loss = F.mse_loss(Gz_prime.view(4,108,192), data[0])

	return loss


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

	#Load up the Generator and Encoder
	gen = bm.Generator(z_dim)
	enc = bm.Encoder(z_dim)

	#Model Filepath
	Generative_filepath = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\BIGAN\BIGAN_Gen.pth"
	Encoder_filepath = r"C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\BIGAN\BIGAN_Enc.pth"

	#Load up the model
	gen.load_state_dict(torch.load(Generative_filepath))
	enc.load_state_dict(torch.load(Encoder_filepath))

	loss = reconstruction_loss(enc, gen, data, batch_size)

	print(loss.item())