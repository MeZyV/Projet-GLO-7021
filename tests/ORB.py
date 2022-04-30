import os
import cv2
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

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3)
plt.axis('off')
plt.savefig('ORB.png')