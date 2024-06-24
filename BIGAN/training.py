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

def disc_loss(DG, DE, eps = 1e-6):
	loss = torch.log(DE + eps) + torch.log(1 - DG + eps)
	return - torch.mean(loss)


def enc_gen_loss(DG, DE, eps = 1e-6):
	loss = torch.log(DG + eps) + torch.log(1 - DE + eps)
	return -torch.mean(loss)


def training(gen, enc, disc, num_epochs):

	for epoch in range(num_epochs):

		for i, 
