import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os


if __name__ == '__main__':

    use_all_data = True
    sample_size = 10000
    base_dir = "D:/craters_classifier"
    craters_images_dataset_folder = os.path.join(base_dir, "craters_dataset")
    output_path = base_dir + "/craters_dataset_all"

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
    craters_array = np.array(craters_list) / 255.0
    craters_array = craters_array.reshape(len(craters_list), img_height * img_widht)
    print(craters_array.shape)
    np.save(output_path, craters_array)