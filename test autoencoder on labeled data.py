import os
from sklearn.preprocessing import MinMaxScaler
from functions import *
from autoencoder import ConvAutoencoder, ResnetAutoencoder
import pandas as pd


if __name__ == '__main__':

    base_dir = "D:/craters_classifier"

    autoencoder_type = 'resnet'  # 'conv' / 'resnet'
    bottle_neck_size = 6  # 2 / 6

    "Tagged craters"

    tagged_craters_images_dataset_folder = base_dir + "/tagged_craters"
    tagged_craters_img_paths = [os.path.join(tagged_craters_images_dataset_folder, f) for f in
                                os.listdir(tagged_craters_images_dataset_folder)]
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

    tagged_craters_array = np.array(tagged_craters_list) / 255.0
    tagged_craters_array = tagged_craters_array.reshape(len(tagged_craters_list), 100 * 100)
    print(tagged_craters_array.shape)

    scaler = MinMaxScaler()
    tagged_craters_array = scaler.fit_transform(tagged_craters_array)

    tagged_craters_array = tagged_craters_array.reshape(tagged_craters_array.shape[0], 100, 100, 1)

    tagged_craters_array_tensor = torch.from_numpy(
        tagged_craters_array.astype(np.float32).reshape(tagged_craters_array.shape[0], 1, 100, 100))

    if autoencoder_type == 'conv':
        if bottle_neck_size == 2:
            autoencoder = ConvAutoencoder(bottleneck_size=2)
            state_dict = torch.load(os.path.join(base_dir, "autoencoder_2.pth"), weights_only=True)
            autoencoder.load_state_dict(state_dict)
        if bottle_neck_size == 6:
            autoencoder = ConvAutoencoder(bottleneck_size=6)
            state_dict = torch.load(os.path.join(base_dir, "autoencoder_6.pth"), weights_only=True)
            autoencoder.load_state_dict(state_dict)
    if autoencoder_type == 'resnet':
        if bottle_neck_size == 2:
            autoencoder = ResnetAutoencoder(bottleneck_size=2)
            state_dict = torch.load(os.path.join(base_dir, "autoencoder_2_resnet.pth"), weights_only=True)
            autoencoder.load_state_dict(state_dict)
        if bottle_neck_size == 6:
            autoencoder = ResnetAutoencoder(bottleneck_size=6)
            state_dict = torch.load(os.path.join(base_dir, "autoencoder_6_resnet.pth"), weights_only=True)
            autoencoder.load_state_dict(state_dict)

    autoencoder = autoencoder.to('cpu')
    plot_latent(tagged_craters_array_tensor, 'dots', 'autoencoder', autoencoder.encoder, labels, label_dict)

