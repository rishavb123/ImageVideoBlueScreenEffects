import cv2
import numpy as np

img = cv2.imread('./raw/belts.jpg')

img = cv2.resize(img, None, fx=0.2, fy=0.2)

cv2.imshow('Original', img)

# lower_bound = np.array([100, 0, 0])    
# upper_bound = np.array([255, 100, 120])
#                       B  G  R
print(img[0][0])
lower_bound = np.array([100, 0, 0])    
upper_bound = np.array([255, 200, 200])

mask = cv2.inRange(img, lower_bound, upper_bound)

cv2.imshow('Mask', mask)

masked_image = np.copy(img)
masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2BGRA)
masked_image[mask != 0] = [0, 0, 0, 0]

cv2.imwrite('./processed/belts.png', masked_image)

cv2.imshow('Masked Image', masked_image)

cv2.waitKey(0)