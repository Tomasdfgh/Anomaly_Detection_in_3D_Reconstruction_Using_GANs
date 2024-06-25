import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
	def __init__(self, z_dim):
		super().__init__()

		#Fully Connected Layers
		self.FulCon1 = nn.Linear(4 * 108 * 192 + z_dim, 1024)
		self.FulCon2 = nn.Linear(1024, 1024)
		self.FulCon3 = nn.Linear(1024, 1)

		#Batch Normalizations
		self.BatchNorm1 = nn.BatchNorm1d(1024)

		#Activation Function
		self.Sigmoid = nn.Sigmoid()

	
	def forward(self, im, z):

		im = im.view(-1, 4 * 108 * 192)
		x = torch.cat([im, z], dim = 1)

		x = F.leaky_relu(self.FulCon1(x), negative_slope = 0.01)
		x = F.leaky_relu(self.BatchNorm1(self.FulCon2(x)), negative_slope = 0.01)
		x = self.Sigmoid(self.FulCon3(x))

		return x

class Encoder(nn.Module):
	def __init__(self, z_dim):
		super().__init__()
		
		#Fully Connected Layers
		self.FulCon1 = nn.Linear(4 * 108 * 192, 1024)
		self.FulCon2 = nn.Linear(1024, 1024)
		self.FulCon3 = nn.Linear(1024, 120)
		self.FulCon4 = nn.Linear(120, z_dim)

		#Batch Normalization
		self.BatchNorm1 = nn.BatchNorm1d(1024)

	def forward(self, x):

		x = x.view(-1, 4 * 108 * 192)

		x = F.leaky_relu(self.FulCon1(x), negative_slope = 0.01)
		x = F.leaky_relu(self.BatchNorm1(self.FulCon2(x)), negative_slope = 0.01)
		x = F.leaky_relu(self.FulCon3(x), negative_slope = 0.01)
		x = self.FulCon4(x)

		return x


class Generator(nn.Module):
	def __init__(self, z_dim):
		super().__init__()

		#Fully Connected Layers
		self.FulCon1 = nn.Linear(z_dim, 1024)
		self.FulCon2 = nn.Linear(1024, 1024)
		self.FulCon3 = nn.Linear(1024, 520)
		self.FulCon4 = nn.Linear(520, 4 * 108 * 192)

		#Batch Normalization
		self.BatchNorm1 = nn.BatchNorm1d(1024)

		#Activation Functions
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):

		x = F.leaky_relu(self.FulCon1(x), negative_slope = 0.01)
		x = F.leaky_relu(self.BatchNorm1(self.FulCon2(x)), negative_slope = 0.01)
		x = F.leaky_relu(self.FulCon3(x), negative_slope = 0.01)
		x = self.sigmoid(self.FulCon4(x))

		return x