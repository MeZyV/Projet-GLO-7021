import os
import cv2
import numpy as np
from datasets.imagematching import ImageCovisibility
from torchvision import transforms as T
from matplotlib import pyplot as plt

# Download data from https://www.kaggle.com/competitions/image-matching-challenge-2022/data
# Extract content of the archive to ./data/ImageMatching
dataset = ImageCovisibility(path=os.path.join('data', 'ImageMatching', 'train'), verbose=False)

MIN_MATCH_COUNT = 10

idx = 5

img1 = dataset[idx]['image1']
img2 = dataset[idx]['image2']

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3)
plt.axis('off')
plt.savefig('SIFT.png')