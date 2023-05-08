import joblib
import cv2
import LMTRP
import numpy as np

# Load classifier


# Load image and extract feature
img = cv2.imread("./ROI1/0001_0002.bmp")
img=cv2.resize(img,(64,64))
print(img.shape)
feature = LMTRP.LMTRP_process(img)

print(feature.shape)
# Predict


