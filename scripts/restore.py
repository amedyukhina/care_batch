from __future__ import print_function, unicode_literals, absolute_import, division

import argparse
from care_batch.restore import restore


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
    restore(**kwargs)


if __name__ == '__main__':
    main()
