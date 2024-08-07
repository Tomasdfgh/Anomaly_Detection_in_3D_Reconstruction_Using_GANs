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

def training(disc, gen, batch_size, num_epochs, z_dim, opt_disc, opt_gen, criterion, train_set, Generative_filepath, Disc_filepath, plt_dir, device):

        D_loss = []
        G_loss = []
        batch_num = []

        for epoch in range(num_epochs):
			
                disc.train()
                gen.train()

                for idx, real in enumerate(train_set):

                        #-------------Train Discriminator: max log(D(x)) + log(1 - D(G(z)))-------------#
                        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
                        fake_gen = gen(noise)

                        fake_gen = fake_gen.view(batch_size,4, 64,64)
    
                        disc_real = disc(real.to(device)).view(-1)
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

                        if idx % 100 == 0:
                            print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {idx}/{len(train_set)} Loss D: {disc_loss:.4f}, loss G: {gen_loss:.4f}")

                print('\n')

                disc.eval()
                gen.eval()

                #Save Model
                torch.save(gen.state_dict(), Generative_filepath)
                torch.save(disc.state_dict(), Disc_filepath)

        plt.plot(torch.tensor(batch_num).cpu().numpy(), torch.tensor(D_loss).cpu().numpy(), label = 'Disc Loss')
        plt.plot(torch.tensor(batch_num).cpu().numpy(), torch.tensor(G_loss).cpu().numpy(), label = 'Gen Loss')
        plt.xlabel('Batch Number')
        plt.ylabel('Loss')
        plt.title('Discriminator and Generator Loss')
        plt.savefig(os.path.join(plt_dir, 'loss_graph.png'))
