import tensorflow as tf
import keras
from keras import layers
import numpy as np
import cv2
import glob
import os
from sklearn.preprocessing import MinMaxScaler

def load_craters_images(N_train = 100, N_val= 20, img_sz = 28, data_path = "./data/", seed = 25):
	'''
	Loads the craters WAC images dataset

	Input:
	N_train - number of training samples
	N_val - number of validation samples
	img_sz - the image size
	data_path - path to the data directory
	seed - random seed (int)

	Output:
	xtrain - training samples as a tensor with size (N_train, image_width, image_height, 1)
	xval - validation samples as a tensor with size (N_val, image_width, image_height, 1)
	'''
	img_list = list()

	# Names of samples (images) in training directory
	np.random.seed(seed)

	# Randmoly choose path to load
	imgs_to_load = np.random.choice(glob.glob(data_path + "*.png"), N_train + N_val, replace=False)

	# Get image indexes from paths
	index_list = [np.int32(os.path.basename(img_path).split('.')[0]) for img_path in imgs_to_load]
	index_list = np.array(index_list) - 1 # subtract 1 due to Matlab's 1-indexing system
	
	for img_path in imgs_to_load:
		# Load image:
		img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
		# Resize image:
		img = cv2.resize(img, (img_sz, img_sz))

		img_list.append(img)

	img_list = np.array(img_list)

	val_imgs = img_list[:N_val]
	val_idxs = index_list[:N_val]
	train_imgs = img_list[N_val:]
	train_idxs = index_list[N_val:]

	return (train_imgs, train_idxs, val_imgs, val_idxs)

def load_crater_by_idx(idx, data_path = "./data/", img_sz = 28):
	'''
	Load as specific crater by its index (filename)
	'''
	img_path = data_path + str(idx) + ".png"

	# Load image:
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	# Resize image:
	img = cv2.resize(img, (img_sz, img_sz))

	scaler = MinMaxScaler()
	img = scaler.fit_transform(img)

	return img