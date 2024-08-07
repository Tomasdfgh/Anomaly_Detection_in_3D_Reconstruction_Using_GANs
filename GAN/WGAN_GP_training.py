import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
import torch

def gradient_penalty(critic, real, fake, device):
    
    batch_size, C, H, W = real.shape
    eps = torch.rand((real.shape[0], 1,1,1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * eps + fake * (1 - eps)
    
    #Calculate Critic Scores
    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
            inputs = interpolated_images,
            outputs = mixed_scores,
            grad_outputs = torch.ones_like(mixed_scores),
            create_graph = True,
            retain_graph = True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim = 1)
    gradient_penalty = torch.mean((gradient_norm -1) ** 2)
    return gradient_penalty

def training(disc, gen, batch_size, num_epochs, z_dim, opt_disc, opt_gen, train_set, Generative_filepath, Disc_filepath, plt_dir, device, critic_iterations, LAMBDA_GP, WEIGHT_CLIP):
    
    D_loss = []
    G_loss = []
    batch_num = []

    for epoch in range(num_epochs):

        disc.train()
        gen.train()

        for idx, real in enumerate(train_set):

            real = real.to(device)
            crit_loss_add = 0

            #---------Train Discriminator---------#
            for _ in range(critic_iterations):
                #z = torch.randn(real.shape[0], z_dim).to(device)
                z = torch.randn(real.shape[0],z_dim, 1, 1).to(device)
                fake = gen(z)
                critic_real = disc(real).reshape(-1)
                critic_fake = disc(fake).reshape(-1)
                gp = gradient_penalty(disc, real, fake, device)
                loss_critic = (
                        -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP*gp
                        )
                crit_loss_add += loss_critic.detach()/critic_iterations
                disc.zero_grad()
                loss_critic.backward(retain_graph = True)
                opt_disc.step()

                #for p in disc.parameters():
                #    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            #---------Train Generator: min -E[disc(gen_fake)]---------#
            output = disc(fake).reshape(-1)
            gen_loss = -torch.mean(output)
            gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            #Adding Losses to array for plotting
            D_loss.append(crit_loss_add.detach())
            G_loss.append(gen_loss.detach())
            batch_num.append(len(batch_num) + 1)

            if idx % 5 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {idx}/{len(train_set)} Loss D: {crit_loss_add:.4f}, loss G: {gen_loss:.4f}")

        print('\n')

        disc.eval()
        gen.eval()

        #Save Model
        torch.save(gen.state_dict(), Generative_filepath)
        torch.save(disc.state_dict(), Disc_filepath)
        
    #Plot and Save Graph
    plt.plot(torch.tensor(batch_num).cpu().numpy(), torch.tensor(D_loss).cpu().numpy(), label = 'Disc Loss')
    plt.plot(torch.tensor(batch_num).cpu().numpy(), torch.tensor(G_loss).cpu().numpy(), label = 'Gen Loss')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Discriminator and Generator Loss')
    plt.savefig(os.path.join(plt_dir, 'loss_graph_GAN.png'))
