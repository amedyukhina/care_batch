from __future__ import print_function, unicode_literals, absolute_import, division

import argparse
from care_batch.datagen import datagen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--basepath', type=str, help='Base folder that contains sub-folders with images')
    parser.add_argument('--source_dir', type=str,
                        help='Folder name relative to `basepath` that contain the source images (e.g., with low SNR)')
    parser.add_argument('--target_dir', type=str,
                        help='Folder name relative to `basepath` that contains the target images (e.g., with high SNR)')
    parser.add_argument("--n_patches_per_image", type=int,
                        help="Number of patches to be sampled/extracted from each raw image pair.")
    parser.add_argument("--patch_size", type=str, default='16,16,16',
                        help='list of patch sizes for all dimensions delimited by ``,``')
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Switch off the random shuffle all extracted patches.")
    parser.add_argument("--normalize-patches", action="store_true",
                        help="Normalize patches with the default CARE normalizer")

    for argname in ['axes', 'pattern', 'patch_axes', 'save_file']:
        parser.add_argument(rf"--{argname}", type=str, default='not_provided')

    args = vars(parser.parse_args())
    kwargs = dict()
    for key in args.keys():
        if args[key] != 'not_provided':
            kwargs[key] = args[key]
    kwargs['patch_size'] = tuple([int(item) for item in kwargs['patch_size'].split(',')])

    noshuffle = kwargs.pop('no_shuffle')
    if noshuffle:
        kwargs['shuffle'] = False
    else:
        kwargs['shuffle'] = True

    datagen(**kwargs)


if __name__ == '__main__':
    main()
