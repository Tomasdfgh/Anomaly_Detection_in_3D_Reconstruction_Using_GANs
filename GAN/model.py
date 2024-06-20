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

		self.FulCon1 = nn.Linear(z_dim, 512 * 27 * 48)
		self.BatchNorm1 = nn.BatchNorm1d(512 * 27 * 48)
		self.unFlatten1 = nn.Unflatten(1, (512, 27, 48))

		self.convTrans1 = nn.ConvTranspose2d(512, 256, kernel_size = 3, stride = 2, padding = 1)
		self.BatchNorm2 = nn.BatchNorm2d(256)

		self.convTrans2 = nn.ConvTranspose2d(256, 128, kernel_size = 3, stride = 2, padding = 1)
		self.BatchNorm3 = nn.BatchNorm2d(128)

		self.convTrans3 = nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 1, padding = 1)
		self.BatchNorm4 = nn.BatchNorm2d(64)

		self.convTrans4 = nn.ConvTranspose2d(64, 4, kernel_size = 3, stride = 1, padding = 1)
		self.convTrans5 = nn.ConvTranspose2d(4, 4, kernel_size=3, stride=1, padding=0)

		self.tanh = nn.Tanh()

	def forward(self, x):

		if x.size(0) > 1:
			x = self.unFlatten1(self.BatchNorm1(F.leaky_relu(self.FulCon1(x), negative_slope = 0.01)))

		else:
			x = self.unFlatten1(F.leaky_relu(self.FulCon1(x), negative_slope = 0.01))

		x = self.BatchNorm2(F.leaky_relu(self.convTrans1(x), negative_slope = 0.01))
		x = self.BatchNorm3(F.leaky_relu(self.convTrans2(x), negative_slope = 0.01))
		x = self.BatchNorm4(F.leaky_relu(self.convTrans3(x), negative_slope = 0.01))
		x = F.leaky_relu(self.convTrans4(x), negative_slope=0.01)
		x = self.tanh(self.convTrans5(x))
		
		return x