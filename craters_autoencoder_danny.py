import scipy
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import cv2
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA



def get_triple(inputs, encoder, autoencoder):
    latent_repr = encoder.predict(inputs)
    print(latent_repr.shape)
    outputs = autoencoder.predict(inputs)
    latent_repr = latent_repr.reshape((latent_repr.shape[0], latent_repr.shape[1], 1))

    return inputs, latent_repr, outputs


def show_encodings(inputs, latent_repr, outputs):
    n = len(inputs)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 5))
    for i in range(n):
        axes[1, i].set_title('({0:.2f}, {1:.2f})'.format(float(latent_repr[i, 0]), float(latent_repr[i, 1])))
        axes[0, i].imshow(inputs[i].reshape(img_height, img_widht), cmap='gray')
        axes[1, i].imshow(outputs[i].reshape(img_height, img_widht), cmap='gray')
    for ax in axes.flatten():
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


def plot_latent(mode, count, technique, x_test, encoder):
    idx = np.random.choice(len(x_test), count)
    inputs = x_test[idx]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(technique)
    if technique == 'autoencoder':
        coords = encoder.predict(inputs)
    elif technique == 'pca':
        coords = PCA(n_components=2).fit_transform(inputs.reshape(count, -1))
    elif technique == 'tsne':
        coords = TSNE(n_components=2).fit_transform(inputs.reshape(count, -1))

    if mode == 'imgs':
        for image, (x, y) in zip(inputs, coords):
            im = OffsetImage(image.reshape(img_height, img_widht), zoom=1, cmap='gray')
            ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
            ax.add_artist(ab)
        ax.update_datalim(coords)
        ax.autoscale()
    elif mode == 'dots':
        classes = [int(random.randint(1, 1)) for _ in range(len(idx))]
        plt.scatter(coords[:, 0], coords[:, 1], c=classes)
        plt.colorbar()
    plt.show()


if __name__ == '__main__':

    base_dir = "/Users/danny/Library/CloudStorage/OneDrive-Technion/Second_degree/Courses/lior's course/moon_craters"
    crater_dataset_path = os.path.join(base_dir, "craters_dataset.npy")
    craters_array = np.load(crater_dataset_path)

    x_train, x_test = train_test_split(craters_array, test_size=0.2)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    x_train = x_train.reshape(x_train.shape[0], img_height, img_widht, 1)
    x_test = x_test.reshape(x_test.shape[0], img_height, img_widht, 1)

    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(x_train[i], cmap='gray')
    plt.show()

    autoencoder, encoder = convolutional_ae_2(x_train, x_test, img_height, img_widht)

    show_encodings(*get_triple(x_test[:10]))
    inputs = np.random.random(size=(10, 10, 10, 1))
    inputs = scipy.ndimage.zoom(inputs, (1, 10, 10, 1))
    show_encodings(*get_triple(inputs, autoencoder))

    plot_latent('dots', 10000, 'autoencoder')
    plot_latent('dots', 10000, 'pca')
    plot_latent('dots', 2000, 'tsne')

    plot_latent('imgs', 300, 'autoencoder')
    plot_latent('imgs', 300, 'pca')
    plot_latent('imgs', 300, 'tsne')