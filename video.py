import sys

import cv2
import numpy as np

name = sys.argv[1] if len(sys.argv) > 1 else 'combined'

capture = cv2.VideoCapture('./raw/' + name + '.mp4')

dev = False
processed_folder = 'test' if dev else 'processed'

def process(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
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
        capture = cv2.VideoCapture('./raw/' + name + '.mp4')
        _, img = capture.read()
        first = False
    gray = process(img)

    delta = cv2.absdiff(last_img, gray)

    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts.sort(key=lambda c: -cv2.contourArea(c))
    cnts_list.append([c for c in cnts if cv2.contourArea(c) > minArea])

    delta = cv2.cvtColor(delta, cv2.COLOR_GRAY2BGR)
    black = np.zeros_like(delta)

    if len(cnts_list) > retrace:
        for i in range(retrace + 1):
            cv2.drawContours(black, cnts_list[-retrace + i - 1], -1, (100 * (1 - i / retrace), 255 * i / retrace, 0), 3)

    if first:
        out.write(black)
    
    cv2.imshow('img', black)
    last_img = gray

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

capture.release()
out.release()
cv2.destroyAllWindows()
