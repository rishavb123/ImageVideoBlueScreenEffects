import cv2
import numpy as np

name = 'combined'

capture = cv2.VideoCapture('./processed/' + name + '.avi')

out = cv2.VideoWriter('./processed/' + name + '-postprocessed.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1280, 720))

def process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img[np.array_equal(img, [1, 0, 1, 1])] = [0, 0, 0, 0]
    return img

_, img = capture.read()

while img is not None:

    img = process(img)

    out.write(img)

    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
    _, img = capture.read()

capture.release()
out.release()
cv2.destroyAllWindows()
