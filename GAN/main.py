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

#Local Files
import loadData as ld
import model as md
import training as tr


np.set_printoptions(threshold=np.inf, linewidth=np.inf)


if __name__ == "__main__":
	rgb_link = r'C:\Users\tomng\Desktop\3D_Detection_Using_GANs\rectangle_data\RGB_Final'
	depth_link = r'C:\Users\tomng\Desktop\3D_Detection_Using_GANs\rectangle_data\Depth_Final'

	#Setting up Transformations and Reverse Transformations
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])

	denormalize = transforms.Normalize(
		mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
		std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
	)

	to_pil = transforms.ToPILImage()

	#Hyperparameters
	lr = 3e-4
	batch_size = 32
	num_epochs = 50
	z_dim = 64

	#Setting up Model
	disc = md.Discriminator()
	gen = md.Generator(z_dim)

	#Setting up the Data
	dataset = ld.load_data(rgb_link, depth_link, [])
	ImageSet = ld.ConvertData(dataset, transform = transform)
	
	#Converting Data to Dataloader
	train_loader = torch.utils.data.DataLoader(ImageSet, batch_size = batch_size, shuffle = True)

	#Optimizers and Criterion
	opt_disc = optim.Adam(disc.parameters(), lr = lr)
	opt_gen = optim.Adam(gen.parameters(), lr = lr)
	criterion = nn.BCELoss()

	#Begin Training
	tr.training(disc, gen, lr, batch_size, num_epochs, z_dim, opt_disc, opt_gen, criterion, train_loader)