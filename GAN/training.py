#General Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image

to_pil = transforms.ToPILImage()

denormalize = transforms.Normalize(
	mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
	std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
)

#to_pil(denormalize(real[:3, :, :])).show()



def training(disc, gen, lr, batch_size, num_epochs, z_dim, opt_disc, opt_gen, criterion, train_set, valid_set, test_set):

	for epoch in range(num_epochs):

		for idx, real in enumerate(train_set):

			#-------------Train Discriminator: max log(D(x)) + log(1 - D(G(z)))-------------#
			noise = torch.rand(batch_size, z_dim)
			fake_gen = gen(noise)
			fake_gen = fake_gen.view(batch_size, 4, 108, 192)
			
			disc_real = disc(real).view(-1)
			disc_loss_real = criterion(disc_real, torch.ones_like(disc_real))

			disc_fake = disc(fake_gen).view(-1)
			disc_loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

			disc_loss = (disc_loss_real + disc_loss_fake) / 2

			disc.zero_grad()
			disc_loss.backward(retain_graph = True)
			opt_disc.step()

			#---------Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))---------#
			output = disc(fake_gen).view(-1)
			gen_loss = criterion(output, torch.ones_like(output))
			gen.zero_grad()
			gen_loss.backward()
			opt_gen.step()

			if idx % 5 == 0:
				print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {idx + 1}/{len(train_set)} Loss D: {disc_loss:.4f}, loss G: {gen_loss:.4f}")