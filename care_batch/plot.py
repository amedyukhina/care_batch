import os

import numpy as np
import pylab as plt
from csbdeep.io import load_training_data
from csbdeep.models import CARE
from csbdeep.utils import plot_some
from skimage import io
from .evaluate import nrmse
from skimage.metrics import structural_similarity


def plot_pairs(fn, folders, basepath, figsize=7, pmin=0, pmax=100, ind=None,
               plot_ssim=False, name_high='high', plot_profile=False, z=50, y=100):
    imgs = [io.imread(os.path.join(basepath, fld, fn)) for fld in folders]
    if ind is not None:
        imgs = [img[ind[0]:ind[1]] for img in imgs]
    title = ''
    ts = 0
    j = 0
    for i in range(len(folders)):
        if len(folders[i]) > 20:
            j += 1
            code = rf'model {j}'
            title += f"{code}: {folders[i]}\n"
            folders[i] = code
            ts += 1

    plt.figure(figsize=(figsize * len(imgs), figsize + ts))
    plot_some(np.stack(imgs), title_list=[folders], pmin=pmin, pmax=pmax)
    plt.suptitle(title)
    plt.tight_layout()

    if plot_ssim:
        ssim_maps = []
        titles = []
        gt = io.imread(os.path.join(basepath, name_high, fn))
        if ind is not None:
            gt = gt[ind[0]:ind[1]]
        for img in imgs:
            ssim_ind, ssim_map = structural_similarity(gt*1., img*1., full=True)
            ssim_maps.append(ssim_map)
            titles.append(rf'SSIM={round(ssim_ind, 2)}; NRSME={round(nrmse(gt, img), 4)}')

        plt.figure(figsize=(figsize * len(imgs), figsize))
        plot_some(np.stack(ssim_maps), title_list=[titles], pmin=0, pmax=100)
        plt.tight_layout()

    if plot_profile:
        plt.figure(figsize=(figsize*2, figsize))
        if z >= imgs[0].shape[0]:
            z = imgs[0].shape[0] - 1
        if y >= imgs[0].shape[1]:
            y = imgs[0].shape[1] - 1
        for i in range(len(imgs)):
            plt.plot(imgs[i][z, y], label=folders[i], lw=2)
        plt.legend()


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

    for i in range(nrows):
        plt.figure(figsize=(ncols * figsize, len(patches) * figsize + 2))
        sl = slice(ncols * i, ncols * (i + 1)), 0
        plot_some(*[pt[sl] for pt in patches], title_list=[np.arange(sl[0].start, sl[0].stop)])
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
