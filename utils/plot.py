import matplotlib.pyplot as plt
import numpy as np


def plot_imgs(imgs, label=None, titles=None, cmap='gray', ylabel='', normalize=False, ax=None, dpi=100):
    if not label is None:
        landmarks = np.asarray(label[:,0,:,:])
        landmarks = landmarks.astype('float')
    n = len(imgs)
    if not isinstance(cmap, list):
        cmap = [cmap]*n
    if ax is None:
        _, ax = plt.subplots(1, n, figsize=(6*n, 6), dpi=dpi)
        if n == 1:
            ax = [ax]
    else:
        if not isinstance(ax, list):
            ax = [ax]
        assert len(ax) == len(imgs)
    for i in range(n):
        if imgs[i].shape[-1] == 3:
            imgs[i] = imgs[i][..., ::-1]  # BGR to RGB
        #imgs[i] = cv2.GaussianBlur(imgs[i], (21, 21), 0)  #add this blur in train
        ax[i].imshow(
            imgs[i], cmap=plt.get_cmap(cmap[i]),
            vmin=None if normalize else 0,
            vmax=None if normalize else 1
        )
        if titles:
            ax[i].set_title(titles[i])
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if not label is None:
            ax[i].scatter(landmarks[i,:, 0][landmarks[i,:, 2]>0.01], landmarks[i,:, 1][landmarks[i,:, 2]>0.01], s=40, marker='.', c='r')
        
    ax[0].set_ylabel(ylabel)
    plt.tight_layout()