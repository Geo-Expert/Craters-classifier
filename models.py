import tensorflow as tf
import keras
from keras import layers

def convolutional_ae(input_img):
	x = layers.Conv2D(16, (3, 3), activation = 'relu', padding='same')(input_img)
	x = layers.MaxPooling2D((2, 2), padding='same')(x)
	x = layers.Conv2D(8, (3, 3), activation = 'relu', padding='same')(x)
	x = layers.MaxPooling2D((2, 2), padding='same')(x)
	x = layers.Conv2D(8, (3, 3), activation = 'relu', padding='same')(x)
	x = layers.MaxPooling2D((2, 2), padding='same')(x)

	x = layers.Flatten()(x)
	x = layers.Dense(1152, activation='relu')(x)
	# x = layers.Dense(500, activation='relu')(x)
	x = layers.Dense(250, activation='relu')(x)
	x = layers.Dense(32, activation='linear')(x)
	encoder_layer = layers.Dense(2, activation='linear')(x)
	x = layers.Dense(32, activation='relu')(encoder_layer)
	x = layers.Dense(250, activation='relu')(x)
	# x = layers.Dense(500, activation='relu')(x)
	x = layers.Dense(1152, activation='relu')(x)
	x = layers.Reshape((12,12,8))(x)

	x = layers.Conv2D(8, (3, 3), activation = 'relu', padding='same')(x)
	x = layers.UpSampling2D((2, 2))(x)
	x = layers.Conv2D(8, (3, 3), activation = 'relu', padding='same')(x)
	x = layers.UpSampling2D((2, 2))(x)
	x = layers.Conv2D(16, (3, 3), activation = 'relu', padding='same')(x)
	x = layers.UpSampling2D((2, 2))(x)
	x = layers.Conv2D(8, (3, 3), activation = 'relu', padding='same')(x)

	decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

	autoencoder = keras.Model(input_img, decoded)
	autoencoder.compile(optimizer='adam', loss='mse')

	return autoencoder, encoder_layer


def convolutional_ae_2(input_img, alpha):
	x = layers.Conv2D(32, (3, 3), strides = (1, 1), padding = 'same')(input_img)
	x = layers.LeakyReLU(alpha=alpha)(x)
	x = layers.MaxPooling2D(pool_size = (5, 5), strides = (1, 1),  padding = 'same')(x)
	x = layers.Conv2D(32, (3, 3), strides = (1, 1), padding = 'same')(x)
	x = layers.LeakyReLU(alpha=alpha)(x)
	x = layers.Conv2D(64, (3, 3), strides = (1, 1), padding = 'same')(x)
	x = layers.LeakyReLU(alpha=alpha)(x)
	x = layers.Conv2D(128, (3, 3), strides = (1, 1), padding = 'same')(x)

	x = layers.Flatten()(x)
	encoder_layer = layers.Dense(2, activation='linear')(x)
	
	x = layers.Dense(524288, activation='relu')(encoder_layer)
	x = layers.Reshape((64, 64, 128))(x)
	x = layers.Conv2DTranspose(64, (3, 3), strides = (1, 1), padding = 'same')(x)
	x = layers.LeakyReLU(alpha=alpha)(x)
	x = layers.Conv2DTranspose(32, (3, 3), strides = (1, 1), padding = 'same')(x)
	x = layers.LeakyReLU(alpha=alpha)(x)
	x = layers.Conv2DTranspose(32, (3, 3), strides = (1, 1), padding = 'same')(x)
	x = layers.LeakyReLU(alpha=alpha)(x)
	x = layers.MaxPooling2D(pool_size = (5, 5), strides = (1, 1),  padding = 'same')(x)
	decoder_layer = layers.Conv2DTranspose(1, (3, 3), strides = (1, 1), padding = 'same', activation = 'sigmoid')(x)

	autoencoder = keras.Model(input_img, decoder_layer)
	autoencoder.compile(optimizer='adam', loss='mse')

	return autoencoder, encoder_layer


def seq_autoencoder(input_img, alpha):
	x = layers.Flatten()(input_img)
	x = layers.Dense(1000)(x)
	x = layers.LeakyReLU(alpha=alpha)(x)
	x = layers.Dense(500)(x)
	x = layers.LeakyReLU(alpha=alpha)(x)
	x = layers.Dense(250)(x)
	x = layers.LeakyReLU(alpha=alpha)(x)
	x = layers.Dense(32)(x)
	x = layers.LeakyReLU(alpha=alpha)(x)

	encoder_layer = layers.Dense(3,  activation='linear')(x)
	
	x = layers.Dense(32)(encoder_layer)
	x = layers.LeakyReLU(alpha=alpha)(x)
	x = layers.Dense(250)(x)
	x = layers.LeakyReLU(alpha=alpha)(x)
	x = layers.Dense(500)(x)
	x = layers.LeakyReLU(alpha=alpha)(x)
	x = layers.Dense(1000)(x)
	x = layers.LeakyReLU(alpha=alpha)(x)
	decoded = layers.Dense(input_img.shape[1],  activation='sigmoid')(x)

	autoencoder = keras.Model(input_img, decoded)

	autoencoder.compile(loss='mse',optimizer='adam')
	    
	return autoencoder, encoder_layer
