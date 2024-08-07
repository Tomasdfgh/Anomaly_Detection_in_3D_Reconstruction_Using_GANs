#General Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
import random
import copy

#Local Files
import loadData as ld

to_pil = transforms.ToPILImage()
denormalize = transforms.Normalize(
	mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
	std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
)
denormalize_depth = transforms.Normalize(
mean=[-0.5 / 0.5],
std=[1 / 0.5]
)

def latent_search(lr_z, lr_G, n_seed, k, z_dim, data, gen):

	#Get the initial model weights
	init_weight = copy.deepcopy(gen.state_dict())

	#Get the Array of latent Vectors
	z_list = [torch.randn(1, z_dim, 1, 1, requires_grad = True) for _ in range(n_seed)]

	z_list_copy = copy.deepcopy(z_list[0])

	#initialize a hashmap to store model's final weights
	mod_weights = {}

	for j in range(n_seed):

		#Load up the model
		gen.load_state_dict(init_weight)

		#Define the optimizers
		opt_gen = optim.Adam(gen.parameters(), lr = lr_G)
		opt_z = optim.Adam([z_list[j]], lr = lr_z)

		for t in range(k):

			#Obtaining G(z)
			gen_z = gen(z_list[j])[0]
			loss = F.mse_loss(gen_z.view(4,64,64), data)

			#zeroing out gradients for both z and gen
			opt_gen.zero_grad()
			opt_z.zero_grad()

			#Perform Loss backprop
			loss.backward(retain_graph = True)

			#Update the gen and z
			opt_gen.step()
			opt_z.step()

		mod_weights[j] = copy.deepcopy(gen.state_dict())

	#Calculate the Final Loss value from all the different seeds
	total_loss = 0
	for j in range(n_seed):

		#Create model with final weights
		gen.load_state_dict(mod_weights[j])
		gen_z = gen(z_list[j])[0]
		loss = F.mse_loss(gen_z.view(4, 64,64), data)

		total_loss += loss.detach().item()

	return (1/n_seed) * total_loss

def get_loss_result(gen, lr_z, lr_G, n_seed, k, z_dim, total_data):

	correct = 0
	correct_pos = 0
	result = []
	loss_avg = {1: 0, 2: 0, 3: 0}
	avg_count = {1: 1e-6, 2: 1e-6, 3: 1e-6}

	for idx, i in enumerate(total_data):


		data, label = i
		loss = latent_search(lr_z, lr_G, n_seed, k, z_dim, data, gen)
		loss_avg[label] += loss
		avg_count[label] += 1

		print("data: " + str(idx) + ", label: " + str(label))
		print("Loss: " + str(loss))
		print("Rec Avg: " + str(loss_avg[1]/avg_count[1]) + ", Cyl Avg: " + str(loss_avg[2]/avg_count[2]) + ", Sph Avg: " + str(loss_avg[3]/avg_count[3]))
		print('\n')

		result.append((loss, label, data))

	return result

def return_acc(threshold, shape_code, result):

	#Find true positives and false positives

	correct = 0
	correct_pos = 0
	correct_neg = 0

	right_label = 0
	wrong_label = 0

	fp_ = 0
	fp_count = 0

	for i in result:
		loss, label = i[0],i[1]

		if label == shape_code:
			right_label += 1
		else:
			wrong_label += 1

		if loss <= threshold and label == shape_code:
			correct += 1
			correct_pos += 1

		elif loss > threshold and label != shape_code:
			correct += 1
			correct_neg += 1
			fp_count += 1

		if loss <= threshold and label != shape_code:
			fp_ += 1
			fp_count += 1

	overall_acc = correct / len(result) if len(result) > 0 else 0
	true_pos = correct_pos / right_label if right_label > 0 else 0
	true_neg = correct_neg / wrong_label if wrong_label > 0 else 0
	false_pos = fp_ / fp_count if fp_count > 0 else 0

	return overall_acc, true_pos, true_neg, false_pos

def graphing(group_code, gan_code, use_result):

	mpl.rcParams['font.family'] = 'serif'
	mpl.rcParams['font.serif'] = ['Times New Roman']

	#Setting Up Naming Code
	name_dict = {1: 'Rectangle', 2: 'Cylinder', 3: 'Sphere'}
	gan_type = {1: 'GAN', 2: 'WGAN'}



	#-------Plot the Graph for accuracy against threshold-------#
	
	max_acc = 0 				#Maximum accuracy
	max_th = 0 					#Right Threshold
	
	best_min_tp_tn = np.inf 	#Right threshold for the min distance between tp and tn
	best_min_th = np.inf
	best_min_acc = np.inf

	x_count = []
	acc_plot = []
	tp_plot = []
	tn_plot = []
	
	for i in np.arange(0.00, 0.5, 0.0001):

		acc, tp_, tn_, _ = return_acc(i, group_code, use_result)
		min_tp_tn = abs(tp_ - tn_)

		x_count.append(i)
		acc_plot.append(100 * acc)
		tp_plot.append(100 *tp_)
		tn_plot.append(100 * tn_)

		#Find the location of the best accuracy
		if acc > max_acc:

			max_acc = acc
			tp  = tp_
			max_th = i
			tn = tn_

		#Find the location of the most optimal accuracy
		if min_tp_tn < best_min_tp_tn:
			
			best_min_tp_tn = min_tp_tn
			best_min_th = i
			best_min_acc = acc



	#Plot the Graph
	plt.title('Performance Metrics vs. Threshold for ' + str(name_dict[group_code]) + " " + str(gan_type[gan_code]))
	plt.xlabel('Threshold')
	plt.ylabel('Accuracy (%)')
	plt.scatter([best_min_th], [(100 * best_min_acc)], color = 'red', marker = 'o', s = 35, label = 'Optimal Accuracy', zorder = 2)
	plt.scatter([max_th], [(100 * max_acc)], color = 'brown', marker = 'o', s = 35, label = 'Max Accuracy', zorder = 2)
	plt.plot(x_count, acc_plot, label = 'Accuracy', zorder = 1)
	plt.plot(x_count, tp_plot, label = 'True Positive', zorder = 1)
	plt.plot(x_count, tn_plot, label = 'True Negative', zorder = 1)
	plt.plot(x_count, len(x_count) * [50], linestyle = '--', color = 'red')
	plt.text(x_count[:int(len(x_count) * 0.85)][-1], 50, '50% Margin', fontsize = 9, ha='left', va='bottom')
	plt.text(max_th, 100 * max_acc + 3, str(round(100 * max_acc, 1)) + '%', fontsize = 9, ha='center', va='center')
	plt.text(best_min_th + 0.015, 100 * best_min_acc, str(round(100 * best_min_acc, 1)) + '%', fontsize = 9, ha='left', va='center')
	plt.legend(loc='lower left', prop={'size': 9})
	plt.show()


	#-------Plot the Graph for Average Bar Graph-------#
	#Find the average
	rec_avg = 0
	cyl_avg = 0
	sph_avg = 0

	rec_avg_count = 0
	cyl_avg_count = 0
	sph_avg_count = 0

	for i in use_result:
		if i[1] == 1:
			rec_avg += i[0]
			rec_avg_count += 1
		if i[1] == 2:
			cyl_avg += i[0]
			cyl_avg_count += 1
		if i[1] == 3:
			sph_avg += i[0]
			sph_avg_count += 1

	# Calculate average values
	rec_avg /= rec_avg_count
	cyl_avg /= cyl_avg_count
	sph_avg /= sph_avg_count

	rec_color = {1: 'red', 2: 'blue', 3: 'blue'}
	cyl_color = {1: 'blue', 2: 'red', 3: 'blue'}
	sph_color = {1: 'blue', 2: 'blue', 3: 'red'}

	# Plot the Graph
	bars = plt.bar(['Rectangle', 'Cylinder', 'Sphere'], [rec_avg, cyl_avg, sph_avg], color= [rec_color[group_code], cyl_color[group_code], sph_color[group_code]], width=0.5)

	# Add value annotations
	for bar in bars:
		yval = bar.get_height()
		plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=12)

	plt.xlabel('Categories')
	plt.ylabel('Loss')
	plt.title('Category-wise Average Loss for ' + str(name_dict[group_code]) + " " + str(gan_type[gan_code]))
	plt.show()


def concatenate_images(image_list, cols, rows):
	# Ensure the number of images matches the specified grid
	assert len(image_list) == rows * cols, "Number of images must match rows * cols"
	
	images = [to_pil(denormalize(i[:3, :, :])) for i in image_list]
	
	# Calculate the width and height for each cell
	max_width = max(image.width for image in images)
	max_height = max(image.height for image in images)
	
	# Create a new blank image with the calculated width and height
	concatenated_image = Image.new('RGB', (cols * max_width, rows * max_height))
	
	# Paste each image into the new image
	for row in range(rows):
		for col in range(cols):
			idx = row * cols + col
			image = images[idx]
			x_offset = col * max_width
			y_offset = row * max_height
			concatenated_image.paste(image, (x_offset, y_offset))
	
	return concatenated_image



if __name__ == "__main__":

	#Links to the Model and Dataset
	rgb_link = r"C:\Users\sens\Desktop\Tom's\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Data\rectangle_data\RGB_Final_FR_Now"
	depth_link = r"C:\Users\sens\Desktop\Tom's\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Data\rectangle_data\Depth_Final_FR_Now"

	rgb_link_cyl = r"C:\Users\sens\Desktop\Tom's\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Data\cylinder_data\RGB_Final"
	depth_link_cyl = r"C:\Users\sens\Desktop\Tom's\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Data\cylinder_data\Depth_Final"

	rgb_link_sphere = r"C:\Users\sens\Desktop\Tom's\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Data\spherical Data\Spherical_RGB_Final"
	depth_link_sphere = r"C:\Users\sens\Desktop\Tom's\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\Data\spherical Data\Spherical_Depth_Final"

	model_link = {
	1: {
	#Rectangle GAN:	
	1: r"C:\Users\sens\Desktop\Tom's\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\GAN\Models\Rectangular\Rectangle_GAN_Traced",
	
	#Rectangle WGAN:
	2: r"C:\Users\sens\Desktop\Tom's\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\GAN\Models\Rectangular\Rectangle_WGAN_traced"
	},
	
	2: {
	#Cylinder GAN:
	1: r"C:\Users\sens\Desktop\Tom's\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\GAN\Models\Cylindrical\Cylinder_GAN_Traced",

	#Cylinder WGAN:
	2: r"C:\Users\sens\Desktop\Tom's\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\GAN\Models\Cylindrical\Cylinder_WGAN_Traced"
	},

	3: {
	#Spherical GAN:
	1: r"C:\Users\sens\Desktop\Tom's\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\GAN\Models\Spherical\Spherical_GAN_Traced",
	
	#Spherical WGAN:
	2: r"C:\Users\sens\Desktop\Tom's\Anomaly_Detection_in_3D_Reconstruction_Using_GANs\GAN\Models\Spherical\Spherical_WGAN_Traced"
	}
	}

	#Testing Codes. Used to determine the model type used and which shape to determine the accuracy
	group_code = 3 		#Legend: 1 for rec, 2 for cyl, and 3 for sph
	gan_code = 2 		#Legend: 1 for gan and 2 for wgan

	#'''
	
	#Hyperparameters
	lr_z = 5e-2
	lr_G = 6e-9
	n_seed = 1
	k = 19
	z_dim = 100
	channels_img = 4

	#Transformer
	transform_RGB = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.5 for _ in range(channels_img - 1)], [0.5 for _ in range(channels_img - 1)]),
	])

	#Setting up rectangular Data
	dataset = ld.load_data(rgb_link, depth_link, [])
	ImageSet = list(ld.ConvertData(dataset, transform = transform_RGB))
	rectangle_use = random.sample(ImageSet, 25)

	#Setting up Cylindrical Data
	dataset_cyl = ld.load_data(rgb_link_cyl, depth_link_cyl, [])
	ImageSet_cyl = list(ld.ConvertData(dataset_cyl, transform = transform_RGB))
	cylindrical_use = random.sample(ImageSet_cyl, 25)

	#Setting up Spherical Data
	dataset_sph = ld.load_data(rgb_link_sphere, depth_link_sphere, [])
	ImageSet_sph = list(ld.ConvertData(dataset_sph, transform = transform_RGB))
	spherical_use = random.sample(ImageSet_sph, 25)

	#Setting up total Dataset
	total_data = []
	for i in rectangle_use:
		total_data.append((i, 1))
	for i in cylindrical_use:
		total_data.append((i ,2))
	for i in spherical_use:
		total_data.append((i, 3))
	random.shuffle(total_data)

	#Load up the model
	gen = torch.jit.load(model_link[group_code][gan_code])

	#Assign Anomaly score to the sampled data
	result = get_loss_result(gen, lr_z, lr_G, n_seed, k, z_dim, total_data)
	#'''

	#The First set of Experiments:
	first = True
	if first:
		#Get Loss Result
		graphing(3, 1, result)

	second = False
	if second:
		row = 6
		col = 2

		result = sorted(result, key=lambda x: x[0])

		top_im = concatenate_images([i[2] for i in result[:row * col]], row, col)
		top_im.show()

		bottom_im = concatenate_images([i[2] for i in result[-(row * col):]], row, col)
		bottom_im.show()



