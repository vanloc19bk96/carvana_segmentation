import numpy as np

def image2onehot(image, num_classes):
	shape = image.shape[:2] + (num_classes, )
	encoded = np.zeros(shape, dtype=np.int8)
	indexes = np.where(np.all(image[:, :] == [255, 255, 255], axis=2))
	indexes = zip(indexes[0], indexes[1])
	for index in indexes:
		encoded[index[0], index[1]] = 1
	return encoded

def onehot2image(onehot):
	num_chanels = 3
	output = np.zeros(onehot.shape[:2] + (num_chanels,))
	indexes = np.where(np.all(onehot[:, :] > 0.5, axis=2))
	indexes = zip(indexes[0], indexes[1])
	for index in indexes:
		output[index[0], index[1]] = [255, 255, 255]
	return output