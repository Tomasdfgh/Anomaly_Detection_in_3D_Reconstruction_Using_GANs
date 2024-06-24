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

#Local Files
import loadData as ld
import model as md
import training as tr
import view_3d as v3


np.set_printoptions(threshold=np.inf, linewidth=np.inf)


if __name__ == "__main__":

	#Link to Data
	rgb_link = r'C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\rectangle_data\RGB_Final'
	depth_link = r'C:\Users\tomng\Desktop\Git Uploads\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\rectangle_data\Depth_Final'

	#Hyperparameters
	lr = 3e-4
	batch_size = 32
	num_epochs = 50
	z_dim = 64

	#Transformers
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])
	to_pil = transforms.ToPILImage()
	denormalize = transforms.Normalize(
		mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
		std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
	)
	denormalize_depth = transforms.Normalize(
	mean=[-0.5 / 0.5],
	std=[1 / 0.5]
	)

	#Setting up Model
	disc = md.Discriminator(z_dim)
	gen = md.Generator(z_dim)
	enc = md.Encoder(z_dim)

	#Setting up the Data
	dataset = ld.load_data(rgb_link, depth_link, [])
	ImageSet = ld.ConvertData(dataset, transform = transform)

	#Converting Data to Dataloader
	train_loader = torch.utils.data.DataLoader(ImageSet, batch_size = batch_size, shuffle = True)

	#Optimizers and Criterion
	opt_disc = optim.Adam(disc.parameters(), lr = lr, betas = (0.5, 0.999), weight_decay = 1e-5)
	opt_enc = optim.Adam(enc.parameters(), lr = lr, betas = (0.5, 0.999), weight_decay = 1e-5)
	opt_gen = optim.Adam(gen.parameters(), lr = lr, betas = (0.5, 0.999), weight_decay = 1e-5)

	#Criterion
	criterion = nn.BCELoss()

	#Begin Training
	tr.training(gen, enc, disc, batch_size, num_epochs, z_dim, opt_disc, opt_gen, opt_enc, train_loader, criterion)