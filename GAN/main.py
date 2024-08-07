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
#import open3d as o3d

#Local Files
import loadData as ld
import DC_model as dcm
import GAN_training as tr
import WGAN_GP_training as wtr
import view_3d as v3

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

if __name__ == "__main__":

    #Link to Data
    rgb_link = r'C:\Users\sens\Desktop\Tom\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Data\rectangle_data\RGB_Final_FR_Now'
    depth_link = r'C:\Users\sens\Desktop\Tom\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Data\rectangle_data\Depth_Final_FR_Now'

    #Model Filepath
    Generative_filepath = r"C:\Users\sens\desktop\Cylinder_GEN_DC_WGAN"
    Disc_filepath = r"C:\Users\sens\desktop\Cylinder_DISC_DC_WGAN"

    #Graph Filepath
    plt_dir = r"C:\Users\sens\desktop"
    
    #Setting up GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using Device: {device}')

    #Transformers
    transform = transforms.Compose([
	transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    to_pil = transforms.ToPILImage()

    #Setting up the Data
    dataset = ld.load_data(rgb_link, depth_link, [])
    ImageSet = ld.ConvertData(dataset, transform_rgb = transform)

    #Hyperparameters
    batch_size = 128
    z_dim = 100
    num_epochs = 1
    lr = 1e-4
    features_gen = 64
    features_disc = 64
    critic_iterations = 5
    lambda_GP = 10
    channels_img = 4
    weight_clip = 0.01


    #Convert Data to Dataloader
    train_loader = torch.utils.data.DataLoader(ImageSet, batch_size = batch_size, shuffle = True)

    #Setting up Model
    gen = dcm.Generator(z_dim, channels_img, features_gen).to(device)
    disc = dcm.Discriminator(channels_img, features_disc).to(device)
    dcm.initialize_weights(gen)
    dcm.initialize_weights(disc)

    #Optimizers
    opt_gen = optim.Adam(gen.parameters(), lr = lr, betas = (0.0, 0.9))
    opt_disc = optim.Adam(disc.parameters(), lr = lr, betas = (0.0, 0.9))
    
    #GAN Training
    tr.training(disc, gen, batch_size, num_epochs, z_dim, opt_disc, opt_gen, nn.BCELoss(), train_loader, Generative_filepath, Disc_filepath, plt_dir, device) 

    #WGAN Training
    #wtr.training(disc, gen, batch_size, num_epochs, z_dim, opt_disc, opt_gen, train_loader, Generative_filepath, Disc_filepath, plt_dir, device, critic_iterations,lambda_GP, weight_clip)
