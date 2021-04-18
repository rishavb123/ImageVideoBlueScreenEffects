import sys

import cv2
import numpy as np

dev = True
default_name = 'setup'

use_factors = True
view_factors = (0.5, 0.5)
view_size = (800, 600)

max_frame_queue_length = 20

min_color = 50
max_color = 255

thresh_vals = (127, 255)
thresh_type = cv2.THRESH_TRUNC

def process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, min_color, max_color, cv2.THRESH_BINARY)
    weights = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    scaled = np.array(weights / 255, np.uint8)
    increased_black_point = img * scaled
    rgb_ret, rgb_thresh = cv2.threshold(increased_black_point, thresh_vals[0], thresh_vals[1], thresh_type)
    print(rgb_thresh.shape)
    return rgb_thresh

def extra_process(img, frame_queue):
    return img
    # delta = cv2.absdiff(img, frame_queue[0])
    # return delta

processed_folder = 'test' if dev else 'processed'

name = sys.argv[1] if len(sys.argv) > 1 else default_name
capture = cv2.VideoCapture('./raw/' + name + '.mp4')

_, last_img = capture.read()
frame_queue = [process(last_img)]
if use_factors:
    view_size = (int(last_img.shape[1] * view_factors[0]), int(last_img.shape[0] * view_factors[1]))
out = cv2.VideoWriter('./' + processed_folder + '/' + name + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (last_img.shape[1], last_img.shape[0]))

cnts_list = []

retrace = 5
minArea = 200

first = True

while True:
    _, img = capture.read()
    if img is None: 
        break

    processed_img = process(img)
    end = extra_process(processed_img, frame_queue)

    cv2.imshow('img', cv2.resize(end, view_size))
    frame_queue.append(img)
    if len(frame_queue) > max_frame_queue_length:
        frame_queue.pop(0)

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

capture.release()
out.release()
cv2.destroyAllWindows()
