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
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image

def show_sample_from_generator(gen, z_dim, batch_size):

	to_pil = transforms.ToPILImage()

	denormalize = transforms.Normalize(
		mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
		std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
	)

	noise = torch.rand(batch_size, z_dim).to(next(gen.parameters()).device)
	with torch.no_grad():
		fake_gen = gen(noise).cpu()
	fake_gen = fake_gen.view(4, 108, 192)
	print(fake_gen.shape)
	rgb_tensor = fake_gen[:3, :, :]
	data_new = to_pil(denormalize(rgb_tensor))
	data_new.show()


def training(disc, gen, lr, batch_size, num_epochs, z_dim, opt_disc, opt_gen, criterion, train_set):

	D_loss = []
	G_loss = []
	batch_num = []

	plt.ion()
	fig, ax = plt.subplots()

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

			#Adding Losses to array for plotting
			D_loss.append(disc_loss.detach())
			G_loss.append(gen_loss.detach())
			batch_num.append(len(batch_num) + 1)

			if idx % 5 == 0:
				print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {idx}/{len(train_set)} Loss D: {disc_loss:.4f}, loss G: {gen_loss:.4f}")

				clear_output(wait = True)

				ax.clear()

				ax.plot(batch_num, D_loss, label = 'Discriminator Loss', color = 'red')
				ax.plot(batch_num, G_loss, label = 'Generator Loss', color = 'blue')

				ax.set_xlabel('Batch Number')
				ax.set_ylabel('Loss')
				ax.legend()
				plt.show()
				plt.pause(0.001)

		#-------------Show a sample of the Generative Model-------------#
		show_sample_from_generator(gen, z_dim, 1)

	plt.ioff()
	plt.show()