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
from autoencoder_danny import convolutional_ae_2


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
    use_all_data = True
    sample_size = 10000
    craters_images_dataset_folder = "/Users/danny/Library/CloudStorage/OneDrive-Technion/Second_degree/Courses/lior's course/moon_craters/craters_dataset"

    craters_img_paths = [os.path.join(craters_images_dataset_folder, f) for f in os.listdir(craters_images_dataset_folder)]
    print(len(craters_img_paths))
    first_img = plt.imread(craters_img_paths[0])
    img_widht, img_height = first_img.shape[1], first_img.shape[0]

    craters_list = []
    if not use_all_data:
        random_numbers = [random.randint(1, len(craters_img_paths)) for _ in range(sample_size)]
        for i in range(sample_size):
            craters_list.append(cv2.imread(craters_img_paths[random_numbers[i]], cv2.IMREAD_GRAYSCALE))
    else:
        for img_path in craters_img_paths:
            craters_list.append(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
    cratters_array = np.array(craters_list) / 255.0
    cratters_array = cratters_array.reshape(len(craters_list), img_height * img_widht)
    print(cratters_array.shape)
    np.save("/Users/danny/Library/CloudStorage/OneDrive-Technion/Second_degree/Courses/lior's course/moon_craters/craters_array_all", cratters_array)

    # x_train, x_test = train_test_split(cratters_array, test_size=0.2)
    #
    # scaler = MinMaxScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.fit_transform(x_test)
    #
    # x_train = x_train.reshape(x_train.shape[0], img_height, img_widht, 1)
    # x_test = x_test.reshape(x_test.shape[0], img_height, img_widht, 1)
    #
    # for i in range(10):
    #     plt.subplot(1, 10, i+1)
    #     plt.imshow(x_train[i], cmap='gray')
    # plt.show()
    #
    # autoencoder, encoder = convolutional_ae_2(x_train, x_test, img_height, img_widht)
    #
    # show_encodings(*get_triple(x_test[:10]))
    # inputs = np.random.random(size=(10, 10, 10, 1))
    # inputs = scipy.ndimage.zoom(inputs, (1, 10, 10, 1))
    # show_encodings(*get_triple(inputs, autoencoder))
    #
    # plot_latent('dots', 10000, 'autoencoder')
    # plot_latent('dots', 10000, 'pca')
    # plot_latent('dots', 2000, 'tsne')
    #
    # plot_latent('imgs', 300, 'autoencoder')
    # plot_latent('imgs', 300, 'pca')
    # plot_latent('imgs', 300, 'tsne')