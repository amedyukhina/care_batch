import os
import shutil
import unittest

import numpy as np
import pandas as pd
from am_utils.utils import walk_dir
from ddt import ddt
from skimage import io

from ..care_prep import care_prep
from ..datagen import datagen
from ..evaluate import evaluate, summarize_stats
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
        img, img_noise = generate_training_pair()
        os.makedirs(path + 'input/high')
        os.makedirs(path + 'input/low')
        io.imsave(path + 'input/high/image.tif', img)
        io.imsave(path + 'input/low/image.tif', img_noise)

        care_prep([path + 'input/high', path + 'input/low'],
                  path + 'data', normalize=True)
        datagen(path + 'data', source_dir='low', target_dir='high',
                save_file=path + 'data.npz', axes='ZYX', patch_size=[8] * 3, n_patches_per_image=4)

        train(path + 'data.npz', model_name='care_model', model_basedir=path + 'models',
              train_epochs=3, train_steps_per_epoch=10)

        restore(path + 'data/low', path + 'data/low_restored',
                model_name='care_model', model_basedir=path + 'models', axes='ZYX')

        evaluate(path + 'data/low_restored', path + 'data/high', path + 'accuracy/restored.csv', model_name='restored')
        evaluate(path + 'data/low', path + 'data/high', path + 'accuracy/low.csv', model_name='low')
        evaluate(path + 'data/high', path + 'data/high', path + 'accuracy/high.csv', model_name='high')
        summarize_stats(walk_dir(path + 'accuracy'), path + 'accuracy.csv')

        self.assertTrue(os.path.exists(path + 'accuracy.csv'))
        self.assertEqual(len(pd.read_csv(path + 'accuracy.csv')), 3)
        shutil.rmtree(path)


if __name__ == '__main__':
    unittest.main()
