import os

import numpy as np
import pylab as plt
from csbdeep.utils import plot_some
from skimage import io


def plot_pairs(fn, folders, basepath, figsize=7, pmin=2, pmax=99.8):
    imgs = [io.imread(os.path.join(basepath, fld, fn)) for fld in folders]
    plt.figure(figsize=(figsize * len(imgs), figsize))
    plot_some(np.stack(imgs), title_list=[folders], pmin=pmin, pmax=pmax)
    plt.tight_layout()
