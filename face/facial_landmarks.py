# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import sys
import imutils
import dlib
import cv2

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# taking the input
args = sys.argv
inputs = []
outputs = []
i = 0
while i < len(args):
    if args[i] == '-i':
        i = i+1
        while args[i] != '-o':
            inputs.append(args[i])
            i = i + 1
    if args[i] == '-o':
        i = i + 1
        while i < len(args):
            outputs.append(args[i])
            i = i + 1
    i = i+1

star = cv2.imread('star.png', -1)
moustache = cv2.imread('moustache.png', -1)

imgs = []
for i in inputs:
    image = cv2.imread(i)
    if image is not 'None':
        imgs.append(image)

def overlay(x1, y1, x2, y2, src, mask):
    dy = y2 - y1
    dx = x2 - x1
    angle = np.math.atan2(dy, dx) * 180 / np.math.pi
    mask = imutils.rotate_bound(mask, int(angle / 2))

    eyew = x2 - x1
    mask = cv2.resize(mask, (eyew + 10, eyew + 10))

    x_offset = x1 - 5
    y_offset = y1 - 3
    y1, y2 = y_offset, y_offset + mask.shape[0]
    x1, x2 = x_offset, x_offset + mask.shape[1]
    alpha_s = mask[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        src[y1:y2, x1:x2, c] = (alpha_s * mask[:, :, c] + alpha_l * src[y1:y2, x1:x2, c])
    return src


for (i, image) in enumerate(imgs):
    # resize the input image, and convert it to grayscale
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # put a star on eyes
        eyew=shape[40][0]-shape[37][0]
        star = cv2.resize(star, (eyew+10, eyew+10))

        x_offset = shape[37][0]-5
        y_offset = shape[37][1]-3
        y1, y2 = y_offset, y_offset + star.shape[0]
        x1, x2 = x_offset, x_offset + star.shape[1]
        alpha_s = star[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            image[y1:y2, x1:x2, c] = (alpha_s * star[:, :, c] +
                                      alpha_l * image[y1:y2, x1:x2, c])

        # adding stars over the eyes
        overlay(shape[37][0], shape[37][1], shape[40][0], shape[40][1], image, star)
        overlay(shape[43][0], shape[43][1], shape[46][0], shape[46][1], image, star)

        # put a moustache
        dy = shape[55][1] - shape[49][1]
        dx = shape[55][0] - shape[49][0]
        angle = np.math.atan2(dy, dx) * 180 / np.math.pi
        moustache = imutils.rotate_bound(moustache, int(angle/10))
        moustachew = shape[55][0] - shape[49][0]
        moustache = cv2.resize(moustache, (moustachew + 10, int((moustachew + 10)/2)))
        x_offset = shape[49][0] - 5
        y_offset = shape[49][1] - (shape[60][1]-shape[34][1])
        y1, y2 = y_offset, y_offset + moustache.shape[0]
        x1, x2 = x_offset, x_offset + moustache.shape[1]
        alpha_s = moustache[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            image[y1:y2, x1:x2, c] = (alpha_s * moustache[:, :, c] +
                                      alpha_l * image[y1:y2, x1:x2, c])

    # writing the image
    cv2.imwrite(outputs[i], image)
