import numpy as np


def int_type(img):
    if img.max() > 255:
        dtype = np.uint16
    else:
        dtype = np.uint8
    return dtype


def normalize(img, maxval, pmin=0, pmax=100):
    img = img.astype(np.float32)
    mn, mx = [np.percentile(img, p) for p in [pmin, pmax]]
    img = np.clip((img - mn) / (mx - mn), 0, 1) * maxval
    return img
