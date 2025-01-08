import scipy
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from functions import *
from autoencoder import ConvAutoencoder


if __name__ == '__main__':

    base_dir = "D:/craters_classifier"
    crater_dataset_path = os.path.join(base_dir, "craters_dataset_10000.npy")
    mode = 'load'   # load/train
    craters_array = np.load(crater_dataset_path)

    x_train, x_test = train_test_split(craters_array, test_size=0.2)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    x_train = x_train.reshape(x_train.shape[0], 100, 100, 1)
    x_test = x_test.reshape(x_test.shape[0], 100, 100, 1)

    # for i in range(10):
    #     plt.subplot(1, 10, i+1)
    #     plt.imshow(x_train[i], cmap='gray')
    # plt.show()

    "auto encoder training"
    #Training Loop
    x_train_tensor = torch.from_numpy(x_train.astype(np.float32).reshape(x_train.shape[0], 1, 100, 100))
    x_test_tensor = torch.from_numpy(x_test.astype(np.float32).reshape(x_test.shape[0], 1, 100, 100))
    train_dataset = TensorDataset(x_train_tensor, x_train_tensor)  # Input and target are the same for autoencoder
    val_dataset = TensorDataset(x_test_tensor, x_test_tensor)
    # Create DataLoader for batching and shuffling
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    autoencoder_2 = ConvAutoencoder(bottleneck_size=2)
    if mode == 'train':
        autoencoder_2 = train_autoencoder(autoencoder_2, train_loader, val_loader, epochs=2, learning_rate=0.001)
        torch.save(autoencoder_2.state_dict(), "autoencoder_2.pth")
    else:
        state_dict = torch.load("autoencoder_2.pth", weights_only=True)
        autoencoder_2.load_state_dict(state_dict)

    autoencoder_6 = ConvAutoencoder(bottleneck_size=6)
    if mode == 'train':
        trained_autoencoder_6 = train_autoencoder(autoencoder_6, train_loader, val_loader, epochs=2, learning_rate=0.001)
        torch.save(autoencoder_6.state_dict(), "autoencoder_6.pth")
    else:
        state_dict = torch.load("autoencoder_6.pth", weights_only=True)
        autoencoder_6.load_state_dict(state_dict)

    "test results"

    random_inputs = np.random.random(size=(10, 10, 10, 1))
    random_inputs = scipy.ndimage.zoom(random_inputs, (1, 10, 10, 1))
    random_inputs_tensor = torch.from_numpy(random_inputs.astype(np.float32).reshape(random_inputs.shape[0], 1, 100, 100))

    show_encodings(x_test_tensor[:10], random_inputs_tensor, autoencoder_2.encoder, autoencoder_2)
    show_encodings(x_test_tensor[:10], random_inputs_tensor, autoencoder_6.encoder, autoencoder_6)

    plot_latent(x_test_tensor[:300], 'dots', 'autoencoder 2', autoencoder_2.encoder)
    plot_latent(x_test_tensor[:300], 'dots', 'autoencoder 6', autoencoder_6.encoder)
    plot_latent(x_test_tensor[:300], 'dots', 'pca')
    plot_latent(x_test_tensor[:300], 'dots', 'tsne')

    plot_latent(x_test_tensor[:300], 'imgs', 'autoencoder 2', autoencoder_2.encoder)
    plot_latent(x_test_tensor[:300], 'imgs', 'autoencoder 6', autoencoder_6.encoder)
    plot_latent(x_test_tensor[:300], 'imgs', 'pca')
    plot_latent(x_test_tensor[:300], 'imgs', 'tsne')

    tagged_craters_images_dataset_folder = base_dir + "/tagged_craters"
    tagged_craters_img_paths = [os.path.join(tagged_craters_images_dataset_folder, f) for f in os.listdir(tagged_craters_images_dataset_folder)]
    print(len(tagged_craters_img_paths))
    tagged_craters_list = []
    labels = []
    for i in range(len(tagged_craters_img_paths)):
        crater_img = cv2.imread(tagged_craters_img_paths[i], cv2.IMREAD_GRAYSCALE)
        crater_img = cv2.resize(crater_img, (100, 100))
        crater_img = flip_crater(crater_img)
        tagged_craters_list.append(crater_img)
        label = os.path.basename(tagged_craters_img_paths[i]).split('.')[0].split('_')[1]
        labels.append(int(label))

    label_dict = {
        1: 'new crater',
        2: 'semi new crater',
        3: 'semi old crater',
        4: 'old crater'
    }

    tagged_cratters_array = np.array(tagged_craters_list) / 255.0
    tagged_cratters_array = tagged_cratters_array.reshape(len(tagged_craters_list), 100 * 100)
    print(tagged_cratters_array.shape)

    x_test_tagged = scaler.fit_transform(tagged_cratters_array)

    x_test_tagged = x_test_tagged.reshape(x_test_tagged.shape[0], 100, 100, 1)

    fig, ax = plt.subplots(1, 10, figsize=(20, 5))
    for i in range(10):
        ax[i].imshow(x_test_tagged[i], cmap='gray')
        ax[i].set_title(label_dict[labels[i]])
        ax[i].axis('off')
    plt.show()

    x_test_tagged_tensor = torch.from_numpy(x_test_tagged.astype(np.float32).reshape(x_test_tagged.shape[0], 1, 100, 100))

    plot_latent(x_test_tagged_tensor, 'dots', 'autoencoder 2', autoencoder_2.encoder, labels, label_dict)
    plot_latent(x_test_tagged_tensor, 'imgs', 'autoencoder 2', autoencoder_2.encoder, labels, label_dict)

    plot_latent(x_test_tagged_tensor, 'dots', 'autoencoder 6', autoencoder_6.encoder, labels, label_dict)
    plot_latent(x_test_tagged_tensor, 'imgs', 'autoencoder 6', autoencoder_6.encoder, labels, label_dict)