import loadData as ld
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch


np.set_printoptions(threshold=np.inf, linewidth=np.inf)


if __name__ == "__main__":
	rgb_link = r'C:\Users\tomng\Desktop\3D_Detection_Using_GANs\rectangle_data\RGB_Final'
	depth_link = r'C:\Users\tomng\Desktop\3D_Detection_Using_GANs\rectangle_data\Depth_Final'

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])

	dataset = ld.load_data(rgb_link, depth_link, [])

	ImageSet = ld.ConvertData(dataset, transform = transform)

	first_item, r, d = ImageSet[0]

	rgb_array = first_item[:3, :, :]
	depth_array = first_item[3:,:,:]

	d.show()

	#d_t = np.transpose(d.numpy(), (1,2,0))
	# image = Image.fromarray(d.numpy(), 'RGB')
	# image.show()