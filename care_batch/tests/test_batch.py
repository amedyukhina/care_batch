import os
import shutil
import unittest

import numpy as np
import pandas as pd
import pylab as plt
from am_utils.utils import walk_dir
from ddt import ddt
from skimage import io

from ..care_prep import care_prep
from ..datagen import datagen
from ..evaluate import evaluate, summarize_stats
from ..plot import plot_pairs, plot_patches
from ..restore import restore
from ..train import train


def generate_training_pair(imgsize=100, objsize=50, noise_std=30):
    img = np.zeros([imgsize] * 3)
    n = int((imgsize - objsize) / 2)
    img[n:n + objsize, n:n + objsize, n:n + objsize] = 255.
    img_noise = img + np.random.normal(loc=0, scale=noise_std, size=img.shape)
    return img, img_noise


@ddt
class TestBatch(unittest.TestCase):

    def test_care(self):
        path = os.getcwd() + '/tmp/'
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path + 'input/high')
        os.makedirs(path + 'input/low')

        for i in range(5):
            img, img_noise = generate_training_pair()
            io.imsave(path + rf'input/high/image{i}.tif', img)
            io.imsave(path + rf'input/low/image{i}.tif', img_noise)

        care_prep([path + 'input/high', path + 'input/low'],
                  path + 'data', normalize_image=True, test_fraction=0, validation_fraction=0.5)
        datagen(path + 'data/train', source_dir='low', target_dir='high',
                save_file=path + 'data.npz', axes='ZYX', patch_size=[8] * 3, n_patches_per_image=20)

        train(path + 'data.npz', model_name='care_model', model_basedir=path + 'models',
              train_epochs=3, train_steps_per_epoch=10)

        restore(path + 'data/validation/low', path + 'data/validation/low_restored',
                model_name='care_model', model_basedir=path + 'models', axes='ZYX')

        evaluate(path + 'data/validation/low_restored', path + 'data/validation/high',
                 path + 'accuracy/restored.csv', model_name='restored')
        evaluate(path + 'data/validation/low', path + 'data/validation/high',
                 path + 'accuracy/low.csv', model_name='low')
        evaluate(path + 'data/validation/high', path + 'data/validation/high',
                 path + 'accuracy/high.csv', model_name='high')
        summarize_stats(walk_dir(path + 'accuracy'), path + 'accuracy.csv')

        fns = os.listdir(path + 'data/validation/high')
        ind = 0
        folders = ['high', 'low', 'low_restored']
        plot_pairs(fns[ind], folders, path + 'data/validation')
        plt.savefig(path + 'plot.png')

        plot_patches(path + 'data.npz', model_name='care_model', model_basedir=path + 'models',
                     ncols=7, nrows=2)
        plt.savefig(path + 'patches.png')

        self.assertTrue(os.path.exists(path + 'accuracy.csv'))
        self.assertEqual(len(pd.read_csv(path + 'accuracy.csv')), 6)
        shutil.rmtree(path)


if __name__ == '__main__':
    unittest.main()
