import numpy as np
import cv2

def im_to_gs(im):
    return np.dot(im, [0.299, 0.587, 0.114])

def darkness(im, threshold):
    gs = im_to_gs(im)
    gs = (gs > threshold) * 255
    gs = gs[100:, 20:140]
    return len(gs[gs==0])


def image_to_ascii(im, size):

    gs_im = im_to_gs(im)
    im = cv2.resize(gs_im, (size*2, size)).T
    asc = []
    chars = ["B","S","#","&","@","$","%","*","!",":","."]
    for j in range(im.shape[1]):
        line = []
        for i in range(im.shape[0]):
            line.append(chars[int(im[i, j]) // 25])
        asc.append("".join(line))

    for line in asc:
        print(line)

