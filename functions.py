from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from rasterio.windows import from_bounds, transform
import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy import cos, radians


def crop_and_save_crater(map_ref, crater_id, lat, lon, diameter, offset, output_dir, transformer, dst_h, dst_w):

    if lon > 180:
        lon -= 360
    # Convert latitude and longitude to map's coordinate system
    x, y = transformer.transform(lon, lat)
    # Define bounding box in projected coordinates
    radius = (diameter / 2) * 1000  # Convert km to meters
    radius_with_offset_x = (radius + radius * offset) / cos(radians(lat))
    radius_with_offset_y = radius + radius * offset
    min_x, min_y = x - radius_with_offset_x, y - radius_with_offset_y
    max_x, max_y = x + radius_with_offset_x, y + radius_with_offset_y

    # Get the window for cropping
    window = from_bounds(min_x, min_y, max_x, max_y, transform=map_ref.transform)

    # Read and crop the data
    cropped_image = map_ref.read(window=window)

    cropped_image = cropped_image.reshape((cropped_image.shape[1], cropped_image.shape[2]))

    projected_height = int(cropped_image.shape[0] / cos(radians(abs(lat))))
    if projected_height > cropped_image.shape[0]:
        cropped_image_projected = cv2.resize(cropped_image, (cropped_image.shape[1], projected_height))
    else:
        cropped_image_projected = cropped_image

    resized_image = cv2.resize(cropped_image_projected, (dst_w, dst_h))

    flipped_image = flip_crater(resized_image)


    plt.subplot(1, 4, 1)
    plt.imshow(cropped_image, cmap='gray')
    plt.title(f'cylindrical')
    plt.subplot(1, 4, 2)
    plt.imshow(cropped_image_projected, cmap='gray')
    plt.title(f'conformal')
    plt.subplot(1, 4, 3)
    plt.imshow(resized_image, cmap='gray')
    plt.title(f'resized')
    plt.subplot(1, 4, 4)
    plt.imshow(flipped_image, cmap='gray')
    plt.title(f'shadow flipped')
    plt.suptitle(f'diamitter:{round(diameter, 0)}, lat:{round(lat, 0)}')
    plt.show()

    filename = f"{output_dir}/{crater_id}.jpeg"
    plt.imsave(filename, flipped_image, cmap='gray')


def flip_crater(img):
    '''
    Flips crater s.t. the shadow will always be on the r.h.s
    '''
    qtr_img_width = np.int16(img.shape[1] / 4)
    half_img_width = np.int16(img.shape[1] / 2)

    left_crater_side = img[:, qtr_img_width:half_img_width]
    right_crater_side = img[:, half_img_width:-qtr_img_width]

    if left_crater_side.mean() > right_crater_side.mean():
        pass
    else:
        img = np.fliplr(img)

    return img


def train_autoencoder(autoencoder, train_loader, val_loader, epochs=10, learning_rate=0.001):
    # Move model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder.to(device)

    # Loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # Training phase
        autoencoder.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = autoencoder(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        autoencoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = autoencoder(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # Log epoch statistics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return autoencoder



def show_encodings(inputs, random_inputs, encoder, autoencoder):
    latent_repr = encoder(inputs).detach().numpy()
    latent_repr_random = encoder(random_inputs).detach().numpy()
    outputs = autoencoder(inputs).detach().numpy()
    outputs_random = autoencoder(random_inputs).detach().numpy()
    latent_repr = latent_repr.reshape((latent_repr.shape[0], latent_repr.shape[1], 1))
    latent_repr_random = latent_repr_random.reshape((latent_repr_random.shape[0], latent_repr_random.shape[1], 1))
    n = len(inputs)
    fig, axes = plt.subplots(4, n, figsize=(4*n, 5))
    for i in range(n):
        coords = []
        coords_random = []
        for j in range(len(latent_repr[i])):
            coords.append(round(latent_repr[i][j][0]))
            coords_random.append(round(latent_repr_random[i][j][0]))
        axes[0, i].set_title(str(coords))
        axes[0, i].imshow(inputs[i].reshape(100, 100), cmap='gray')
        axes[1, i].imshow(outputs[i].reshape(100, 100), cmap='gray')
        axes[2, i].set_title(str(coords_random))
        axes[2, i].imshow(random_inputs[i].reshape(100, 100), cmap='gray')
        axes[3, i].imshow(outputs_random[i].reshape(100, 100), cmap='gray')
    for ax in axes.flatten():
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def plot_latent(inputs, mode, technique, encoder=None, classes=None, label_dict=None):

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(technique)
    if technique == 'autoencoder 2' or technique == 'autoencoder 6':
        coords = (encoder(inputs).detach().numpy())
        if coords.shape[1] > 2:
          coords = (PCA(n_components=2).fit_transform(coords.reshape(len(inputs), -1)))
    elif technique == 'pca':
        coords = (PCA(n_components=2).fit_transform(inputs.reshape(len(inputs), -1)))
    elif technique == 'tsne':
        coords =(TSNE(n_components=2).fit_transform(inputs.reshape(len(inputs), -1)))

    if mode == 'imgs':
        for image, (x, y) in zip(inputs, coords):
            im = OffsetImage(image.reshape(100, 100), zoom=0.2, cmap='gray')
            ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
            ax.add_artist(ab)
        ax.update_datalim(coords)
        ax.autoscale()
    elif mode == 'dots':
        if classes is None:
          plt.scatter(coords[:, 0], coords[:, 1])
        else:
          for i in np.unique(classes):
            plt.scatter(coords[classes == i, 0], coords[classes == i, 1], label=label_dict[i])
            plt.legend()
          # for i in np.unique(classes):
          #   class_center = np.mean(coords[classes == i], axis=0)
          #   text = TextArea('{} ({})'.format(label_dict[i], i))
          #   ab = AnnotationBbox(text, class_center, xycoords='data', frameon=True)
          #   ax.add_artist(ab)
    plt.show()