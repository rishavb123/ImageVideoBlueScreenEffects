import cv2
import numpy as np

img = cv2.imread('./raw/belts.jpg')

def display(name, img):
    cv2.imshow(name, cv2.resize(img, None, fx=0.2, fy=0.2))

# display('Original', img)

# lower_bound = np.array([100, 0, 0])    
# upper_bound = np.array([255, 100, 120])
#                       B  G  R

def create_mask(img, f):
    mask = np.zeros((img.shape[0], img.shape[1]))
    for i in range(len(img)):
        for j in range(len(img[i])):
            print(i, '/', len(img), j, '/', len(img[i]), ' ' * 10, end='\r')
            if f(img[i][j]):
                mask[i][j] = 1
    print()
    return mask

masked_image = np.copy(img)

lower_bound = np.array([100, 0, 0])    
upper_bound = np.array([255, 200, 200])

mask = cv2.inRange(img, lower_bound, upper_bound)
print(mask.shape, img.shape)

print(masked_image[0][0])

orig_pix = np.array([163, 144, 109])

threshold = 100

def f(pix):
    return np.linalg.norm(pix - orig_pix) < threshold

mask = create_mask(img, f)
print(mask.shape)

display('Mask', mask)

masked_image = np.copy(img)
# masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2BGRA)
masked_image[mask != 0] = [0, 0, 0]

display('Masked Image', masked_image)

cv2.waitKey(0)

cv2.imwrite('./processed/belts.png', masked_image)