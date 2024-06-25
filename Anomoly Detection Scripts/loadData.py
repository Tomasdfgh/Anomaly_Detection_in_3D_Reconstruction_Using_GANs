import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset


class ConvertData(Dataset):
	def __init__(self, dataset, transform = None):
		self.dataset = dataset
		self.transform = transform

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		depth_image, rgb_image = self.dataset[idx]

		rgb_image = self.transform(rgb_image)

		depth_image = transforms.ToTensor()(depth_image)
		depth_image = transforms.Normalize((0.5,),(0.5,))(depth_image)

		combined_image = torch.cat((rgb_image, depth_image), dim = 0)

		return combined_image


def load_data(rgb_link, depth_link, dataset):


	for filename in os.listdir(rgb_link):

		rgb_path = os.path.join(rgb_link, filename)
		depth_path = os.path.join(depth_link, filename[:-4] + '.png')

		depth_image = Image.open(depth_path)
		rgb_image = Image.open(rgb_path)

		dataset.append((depth_image, rgb_image))

	return dataset


