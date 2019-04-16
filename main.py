import cv2
from imgaug import augmenters as iaa
import sys

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


def fog_augmentation1(img):
    seq = iaa.Sequential(iaa.Fog(random_state=9))
    img = seq.augment_image(img)
    seq = iaa.Sequential(iaa.MotionBlur())
    img = seq.augment_image(img)
    return img


imgs = []
for i in inputs:
    image = cv2.imread(i)
    if image is not 'None':
        imgs.append(image)


i = 0
while i < len(imgs):
    cv2.imwrite(outputs[i], fog_augmentation1(imgs[i]))
    i = i+1
