import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()

		self.pool = nn.MaxPool2d(2,2)
		self.sigmoid = nn.Sigmoid()

		#Convolutional Layers
		self.convLayer1 = nn.Conv2d(4,16,3)
		self.convLayer2 = nn.Conv2d(16,120,3)

		#Fully Connected Layers
		self.FulCon1 = nn.Linear(120 * 25 * 46, 256)
		self.FulCon2 = nn.Linear(256, 120)
		self.FulCon3 = nn.Linear(120,1)


	
	def forward(self, x):

		#Convolutional Layers
		x = self.pool(F.leaky_relu(self.convLayer1(x), negative_slope = 0.01))
		x = self.pool(F.leaky_relu(self.convLayer2(x), negative_slope = 0.01))

		#Fully Connected Layers
		x = x.view(-1, 120 * 25 * 46)
		x = F.leaky_relu(self.FulCon1(x), negative_slope = 0.01)
		x = F.leaky_relu(self.FulCon2(x), negative_slope = 0.01)
		x = self.sigmoid(self.FulCon3(x))

		return x


class Generator(nn.Module):
	def __init__(self, z_dim):
		super().__init__()

		self.FulCon1 = nn.Linear(z_dim, 256)
		self.FulCon2 = nn.Linear(256, 4 * 108 * 192)

		self.tanh = nn.Tanh()

	def forward(self, x):

		x = F.leaky_relu(self.FulCon1(x), negative_slope = 0.01)
		x = self.tanh(self.FulCon2(x))

		return x