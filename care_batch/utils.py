import numpy as np


def int_type(img):
    if img.max() > 255:
        dtype = np.uint16
    else:
        dtype = np.uint8
    return dtype
