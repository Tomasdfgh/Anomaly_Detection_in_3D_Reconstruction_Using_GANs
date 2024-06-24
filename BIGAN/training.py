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

def training(gen, enc, disc, batch_size, num_epochs, z_dim, opt_disc, opt_gen, opt_enc, train_set, criterion):

	d_loss_graph = []
	g_loss_graph = []
	e_loss_graph = []
	batch_num = []

	for epoch in range(num_epochs):

		disc.train()
		gen.train()
		enc.train()
		
		for idx, real in enumerate(train_set):

			batch_num.append(len(batch_num) + 1)
			real = real.view(batch_size, -1)

			#Train Discriminator
			z_real = enc(real)
			z_noise = torch.randn(batch_size, z_dim)
			fake_images = gen(z_noise)

			real_pairs = disc(real, z_real)
			fake_pairs = disc(fake_images.detach(), z_noise)

			d_loss_real = criterion(real_pairs, torch.ones(real_pairs.shape[0], 1))
			d_loss_fake = criterion(fake_pairs, torch.zeros(fake_pairs.shape[0], 1))
			d_loss = d_loss_real + d_loss_fake
			d_loss_graph.append(d_loss.detach())

			opt_disc.zero_grad()
			d_loss.backward()
			opt_disc.step()

			#Train Generator
			fake_pairs = disc(fake_images, z_noise)
			g_loss = criterion(fake_pairs, torch.ones(fake_pairs.shape[0], 1))
			g_loss_graph.append(g_loss.detach())

			opt_gen.zero_grad()
			g_loss.backward()
			opt_gen.step()

			#Train Encoder
			encoded_real = enc(real)
			recon_loss = criterion(disc(real,encoded_real), torch.ones(encoded_real.shape[0], 1))
			e_loss_graph.append(recon_loss.detach())

			opt_enc.zero_grad()
			recon_loss.backward()
			opt_enc.step()

			if idx % 5 == 0:
				print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {idx}/{len(train_set)} Loss D: {d_loss:.4f}, loss G: {g_loss:.4f} Loss E: {recon_loss:.4f}")

				clear_output(wait = True)

				plt.cla()
				plt.plot(batch_num, d_loss_graph, label = 'Discriminator Loss', color = 'red')
				plt.plot(batch_num, g_loss_graph, label = 'Generator Loss', color = 'blue')
				plt.plot(batch_num, e_loss_graph, label = 'Encoder Loss', color = 'yellow')

				plt.title('Generator, Discriminator, and Encoder Loss')
				plt.xlabel('Batch Number')
				plt.ylabel('Loss')
				plt.legend()
				plt.pause(0.001)

		disc.eval()
		gen.eval()
		enc.eval()

	plt.ioff()
	plt.show()