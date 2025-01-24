import os
from sklearn.preprocessing import MinMaxScaler
from functions import *
from autoencoder import ConvAutoencoder, ResnetAutoencoder
import pandas as pd


if __name__ == '__main__':

    base_dir = "D:/craters_classifier"
    crater_dataset_path = os.path.join(base_dir, "craters_dataset_all.npy")
    craters_info_path = os.path.join(base_dir, "craters_info.csv")

    autoencoder_type = 'resnet'  # 'conv' / 'resnet'
    bottle_neck_size = 6  # 2 / 6
    use_pca = False
    if bottle_neck_size>2:
        output_path = os.path.join(base_dir, f'craters_info_results_{autoencoder_type}{bottle_neck_size}_pca_{use_pca}.csv')
    else:
        output_path = os.path.join(base_dir, f'craters_info_results_{autoencoder_type}{bottle_neck_size}.csv')

    print("Loading craters data")
    craters_array = np.load(crater_dataset_path)
    craters_info = pd.read_csv(craters_info_path)
    print(craters_array.shape)

    img_width, img_height = int(np.sqrt(craters_array.shape[1])), int(np.sqrt(craters_array.shape[1]))

    scaler = MinMaxScaler()
    craters_array = scaler.fit_transform(craters_array)
    craters_array = craters_array.reshape(craters_array.shape[0], img_height, img_width, 1)

    if autoencoder_type == 'conv':
        if bottle_neck_size == 2:
            autoencoder = ConvAutoencoder(bottleneck_size=2)
            state_dict = torch.load("autoencoder_2.pth", weights_only=True)
            autoencoder.load_state_dict(state_dict)
        if bottle_neck_size == 6:
            autoencoder = ConvAutoencoder(bottleneck_size=6)
            state_dict = torch.load("autoencoder_6.pth", weights_only=True)
            autoencoder.load_state_dict(state_dict)
    if autoencoder_type == 'resnet':
        if bottle_neck_size == 2:
            autoencoder = ResnetAutoencoder(bottleneck_size=2)
            state_dict = torch.load("autoencoder_2_resnet.pth", weights_only=True)
            autoencoder.load_state_dict(state_dict)
        if bottle_neck_size == 6:
            autoencoder = ResnetAutoencoder(bottleneck_size=6)
            state_dict = torch.load("autoencoder_6_resnet.pth", weights_only=True)
            autoencoder.load_state_dict(state_dict)

    craters_array_tensor = torch.from_numpy(craters_array.astype(np.float32).reshape(craters_array.shape[0], 1, img_height, img_width))

    results = predict_craters(craters_array_tensor, autoencoder.encoder, use_pca=use_pca)
    for i in range(results.shape[1]):
        craters_info[f'result {i+1}'] = results[:, i]
    craters_info.to_csv(output_path, index=False)


