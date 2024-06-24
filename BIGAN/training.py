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

torch.autograd.set_detect_anomaly(True)

def disc_loss(DG, DE, eps = 1e-6):
	loss = torch.log(DE + eps) + torch.log(1 - DG + eps)
	return - torch.mean(loss)


def gen_loss(DG, eps = 1e-6):
	loss = torch.log(DG + eps)
	return -torch.mean(loss)

def enc_loss(DE, eps = 1e-6):
	loss = torch.log(DE + eps)
	return -torch.mean(loss)


def training(gen, enc, disc, batch_size, num_epochs, z_dim, opt_disc, opt_gen, opt_enc, train_set, criterion):

	for epoch in range(num_epochs):
		
		for idx, real in enumerate(train_set):

			images = real.reshape(real.size(0),-1)

			#Getting Noise
			z = torch.randn(batch_size, z_dim)

			#Get G(z) and E(x)
			Gz = gen(z)
			Ex = enc(images)

			#Get D(G(z), z) and D(x, E(x))
			DG = disc(Gz, z)
			DE = disc(images, Ex)

			#Calculate Losses
			loss_D = disc_loss(DG, DE)
			loss_E = enc_loss(DE)
			loss_G = gen_loss(DG)

			#Encoder Training
			opt_enc.zero_grad()
			loss_E.backward(retain_graph = True)
			opt_enc.step()

			#Discriminator Training
			opt_disc.zero_grad()
			loss_D.backward(retain_graph = True)
			opt_disc.step()

			#Generator Training
			opt_gen.zero_grad()
			loss_G.backward()
			opt_gen.step()

			break
		break
