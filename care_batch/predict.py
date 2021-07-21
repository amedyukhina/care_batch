from __future__ import print_function, unicode_literals, absolute_import, division

import argparse
import os

from am_utils.utils import walk_dir, imsave
from csbdeep.models import CARE
from csbdeep.utils.tf import limit_gpu_memory
from skimage import io
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_basedir', type=str,
                        help='Path to model folder (which stores configuration, weights, etc.)')
    parser.add_argument('--model_name', type=str,
                        help='Model name (relative to ``model_basedir``')
    parser.add_argument('--input_dir', type=str,
                        help='Folder name with images to restore')
    parser.add_argument('--output_dir', type=str,
                        help='Folder name to save the restored images')
    parser.add_argument('--limit_gpu', type=float, default=0.5,
                        help='Fraction of the GPU memory to use. Default: ``0.5``')
    parser.add_argument('--axes', type=str, default='ZYX',
                        help='Axes of the input image. Default: ``ZYX``')
    parser.add_argument("--n_tiles", type=str, default=None,
                        help='A tuple of the number of tiles for every image axis delimited by ``,``')

    for argname in ['normalizer', 'resizer']:
        parser.add_argument(rf"--{argname}", type=str, default='not_provided')

    args = vars(parser.parse_args())
    kwargs = dict()
    for key in args.keys():
        if args[key] != 'not_provided':
            kwargs[key] = args[key]
    kwargs['n_tiles'] = tuple([int(item) for item in kwargs['n_tiles'].split(',')])
    predict(**kwargs)


def predict(input_dir, output_dir, model_name, model_basedir, limit_gpu, **kwargs):
    """

    Parameters
    ----------
    input_dir : str
        Folder name with images to restore
    output_dir : str
        Folder name to save the restored images
    model_name : str
        Model name.
    model_basedir : str
        Path to model folder (which stores configuration, weights, etc.)
    limit_gpu : float
        Fraction of the GPU memory to use.
        Default: 0.5
    kwargs : dict
        Configuration attributes (see below).

    Attributes
    ----------
    axes : str
        Axes of the input ``img``.
    normalizer : :class:`csbdeep.data.Normalizer` or None
        Normalization of input image before prediction and (potentially) transformation back after prediction.
    resizer : :class:`csbdeep.data.Resizer` or None
        If necessary, input image is resized to enable neural network prediction and result is (possibly)
        resized to yield original image size.
    n_tiles : iterable or None
        Out of memory (OOM) errors can occur if the input image is too large.
        To avoid this problem, the input image is broken up into (overlapping) tiles
        that can then be processed independently and re-assembled to yield the restored image.
        This parameter denotes a tuple of the number of tiles for every image axis.
        Note that if the number of tiles is too low, it is adaptively increased until
        OOM errors are avoided, albeit at the expense of runtime.
        A value of ``None`` denotes that no tiling should initially be used.

    """
    limit_gpu_memory(fraction=limit_gpu)
    model = CARE(config=None, name=model_name, basedir=model_basedir)
    samples = walk_dir(input_dir)
    for sample in tqdm(samples):
        if sample.startswith('.'):
            continue
        output_fn = output_dir + sample[len(input_dir):]
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)

        x = io.imread(sample)
        restored = model.predict(x, **kwargs)
        imsave(output_fn, restored)


if __name__ == '__main__':
    main()
