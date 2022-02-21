import os

import numpy as np
import pylab as plt
from csbdeep.io import load_training_data
from csbdeep.models import CARE
from csbdeep.utils import plot_some
from skimage import io


def plot_pairs(fn, folders, basepath, figsize=7, pmin=2, pmax=99.8):
    imgs = [io.imread(os.path.join(basepath, fld, fn)) for fld in folders]
    plt.figure(figsize=(figsize * len(imgs), figsize))
    plot_some(np.stack(imgs), title_list=[folders], pmin=pmin, pmax=pmax)
    plt.tight_layout()


def plot_patches(datafile, ncols=7, nrows=2, figsize=4, model_name=None, model_basedir=None):
    (X, Y), (_, _), axes = load_training_data(datafile, validation_split=0.2, verbose=True)
    title = 'Example patches:\n' \
            'first row: input (source),  ' \
            'second row: target (ground truth)'

    patches = [X, Y]

    if model_name is not None and model_basedir is not None:
        model = CARE(config=None, name=model_name, basedir=model_basedir)
        predicted = model.keras_model.predict(X[:ncols * nrows])
        patches.append(predicted)
        title += ',  third row: predicted from source'

    plt.figure(figsize=(ncols * figsize, len(patches) * figsize + 2))
    for i in range(nrows):
        sl = slice(ncols * i, ncols * (i + 1)), 0
        plot_some(*[pt[sl] for pt in patches], title_list=[np.arange(sl[0].start, sl[0].stop)])
        plt.suptitle(title)
    plt.tight_layout()
