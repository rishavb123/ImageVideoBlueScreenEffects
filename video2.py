import sys

import cv2
import numpy as np

name = sys.argv[1] if len(sys.argv) > 1 else 'combined'

dev = True
processed_folder = 'test' if dev else 'processed'

capture = cv2.VideoCapture('./raw/' + name + '.mp4')

def process(img):
    return img

def extra_process(img, last_img):
    return img

_, last_img = capture.read()
last_img = process(last_img)
out = cv2.VideoWriter('./' + processed_folder + '/' + name + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (last_img.shape[1], last_img.shape[0]))

cnts_list = []

retrace = 5
minArea = 200

first = True

while True:
    _, img = capture.read()
    if img is None: 
        break


    processed_img = process(img, last_img)
    end = extra_process(processed_img, last_img)

    cv2.imshow('img', end)
    last_img = img

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

capture.release()
out.release()
cv2.destroyAllWindows()
