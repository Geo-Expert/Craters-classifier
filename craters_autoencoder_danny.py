import scipy
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from functions import *
from autoencoder import ConvAutoencoder, ResnetAutoencoder
import pandas as pd


if __name__ == '__main__':

    base_dir = "D:/craters_classifier"
    crater_dataset_path = os.path.join(base_dir, "craters_dataset_all.npy")
    craters_info_path = os.path.join(base_dir, "craters_info.csv")
    autoencoder_type = 'conv'  # 'conv' / 'resnet'
    train = False  # if false, load the selected type autoencoder weights from drive
    save = False
    loss = 'mse'  # 'mse' / 'ssim'
    epochs = 5
    batch_size = 256

    print("Loading craters data")
    craters_array = np.load(crater_dataset_path)
    craters_info = pd.read_csv(craters_info_path)
    print(craters_array.shape)

    img_width, img_height = int(np.sqrt(craters_array.shape[1])), int(np.sqrt(craters_array.shape[1]))

    scaler = MinMaxScaler()
    craters_array = scaler.fit_transform(craters_array)
    craters_array = craters_array.reshape(craters_array.shape[0], img_height, img_width, 1)

    indices = np.arange(craters_array.shape[0])
    np.random.shuffle(indices)
    craters_array[:] = craters_array[indices]

    # fig, ax = plt.subplots(1, 10, figsize=(20, 2))
    # for i in range(10):
    #     ax[i].imshow(craters_array[i], cmap='gray')
    #     ax[i].axis('off')
    # plt.show()

    if autoencoder_type == 'conv':
        autoencoder_2 = ConvAutoencoder(bottleneck_size=2)
        autoencoder_6 = ConvAutoencoder(bottleneck_size=6)
    if autoencoder_type == 'resnet':
        autoencoder_2 = ResnetAutoencoder(bottleneck_size=2)
        autoencoder_6 = ResnetAutoencoder(bottleneck_size=6)

    if train:
        "auto encoder training"
        train_data, val_data = train_test_split(craters_array, test_size=0.2)
        train_data_tensor = torch.from_numpy(train_data.astype(np.float32).reshape(train_data.shape[0], 1, img_height, img_width))
        val_data_tensor = torch.from_numpy(val_data.astype(np.float32).reshape(val_data.shape[0], 1, img_height, img_width))
        train_dataset = TensorDataset(train_data_tensor, train_data_tensor)
        val_dataset = TensorDataset(val_data_tensor, val_data_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        autoencoder_2 = train_autoencoder(autoencoder_2, train_loader, val_loader, epochs=epochs, learning_rate=0.001, loss=loss)

        autoencoder_6 = train_autoencoder(autoencoder_6, train_loader, val_loader, epochs=epochs, learning_rate=0.001, loss=loss)

        if save:
            if autoencoder_type == 'conv':
                torch.save(autoencoder_2.state_dict(), "autoencoder_2.pth")
                torch.save(autoencoder_6.state_dict(), "autoencoder_6.pth")
            if autoencoder_type == 'resnet':
                torch.save(autoencoder_2.state_dict(), "autoencoder_2_resnet.pth")
                torch.save(autoencoder_6.state_dict(), "autoencoder_6_resnet.pth")

    else:
        if autoencoder_type == 'conv':
            state_dict = torch.load(os.path.join(base_dir, "autoencoder_2.pth"), weights_only=True)
            autoencoder_2.load_state_dict(state_dict)
            state_dict = torch.load(os.path.join(base_dir, "autoencoder_6.pth"), weights_only=True)
            autoencoder_6.load_state_dict(state_dict)
        if autoencoder_type == 'resnet':
            state_dict = torch.load(os.path.join(base_dir, "autoencoder_2_resnet.pth"), weights_only=True)
            autoencoder_2.load_state_dict(state_dict)
            state_dict = torch.load(os.path.join(base_dir, "autoencoder_6_resnet.pth"), weights_only=True)
            autoencoder_6.load_state_dict(state_dict)

    craters_array_tensor = torch.from_numpy(craters_array.astype(np.float32).reshape(craters_array.shape[0], 1, img_height, img_width))

    "test results"

    random_inputs = np.random.random(size=(10, 10, 10, 1))
    random_inputs = scipy.ndimage.zoom(random_inputs, (1, 10, 10, 1))
    random_inputs_tensor = torch.from_numpy(random_inputs.astype(np.float32).reshape(random_inputs.shape[0], 1, 100, 100))

    show_encodings(craters_array_tensor[:10], random_inputs_tensor, autoencoder_2.encoder, autoencoder_2)
    show_encodings(craters_array_tensor[:10], random_inputs_tensor, autoencoder_6.encoder, autoencoder_6)

    plot_latent(craters_array_tensor[:300], 'dots', 'autoencoder 2', autoencoder_2.encoder)
    plot_latent(craters_array_tensor[:300], 'dots', 'autoencoder 6', autoencoder_6.encoder)
    plot_latent(craters_array_tensor[:300], 'dots', 'pca')
    plot_latent(craters_array_tensor[:300], 'dots', 'tsne')

    plot_latent(craters_array_tensor[:300], 'imgs', 'autoencoder 2', autoencoder_2.encoder)
    plot_latent(craters_array_tensor[:300], 'imgs', 'autoencoder 6', autoencoder_6.encoder)
    plot_latent(craters_array_tensor[:300], 'imgs', 'pca')
    plot_latent(craters_array_tensor[:300], 'imgs', 'tsne')

    "Tagged craters"

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