from matplotlib.ft2font import LOAD_NO_BITMAP
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os


def plot_synthetic_data(image_path, landmark_path, shape=(4, 4), figsize=(12, 16)):
    landmark_df = pd.read_csv(landmark_path)
    samples = landmark_df.sample(n=shape[0] * shape[1])
    fig, ax = plt.subplots(shape[0], shape[1], figsize=figsize)
    for idx, (i, row) in enumerate(samples.iterrows()):
        img = cv2.imread(os.path.join(image_path, row['im_name']))
        landmarks = row.iloc[1:].values.reshape(-1, 3)
        landmarks = landmarks[landmarks[:, 2] > 0]
        ax[idx // shape[0], idx % shape[1]].scatter(landmarks[:, 0], landmarks[:, 1], s=8, c='r')
        ax[idx // shape[0], idx % shape[1]].imshow(img)
        ax[idx // shape[0], idx % shape[1]].axis('off')
    plt.show()
    fig.savefig('images/synthetic_data.png')

if __name__ == '__main__':
    image_path = './data/synthetic_shapes/images/'
    landmark_path = './data/synthetic_shapes/syn_shape_labels.csv'
    plot_synthetic_data(image_path, landmark_path)