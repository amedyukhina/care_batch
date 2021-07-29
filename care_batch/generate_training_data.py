from __future__ import print_function, unicode_literals, absolute_import, division

import argparse
import inspect

from csbdeep.data import RawData, create_patches


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
    parser.add_argument("--no-normalization", action="store_true",
                        help="Switch off CARE's internal percentile normalizer")
        
    
    for argname in ['axes', 'pattern', 'patch_axes', 'save_file']:
        parser.add_argument(rf"--{argname}", type=str, default='not_provided')
        

    args = vars(parser.parse_args())
    kwargs = dict()
    for key in args.keys():
        if args[key] != 'not_provided':
            kwargs[key] = args[key]
    kwargs['patch_size'] = tuple([int(item) for item in kwargs['patch_size'].split(',')])
    
    nonormalize = kwargs.pop('no_normalization')
    if nonormalize:
        kwargs['normalization'] = None
        
    noshuffle = kwargs.pop('no_shuffle')
    if noshuffle:
        kwargs['shuffle'] = False
    else:
        kwargs['shuffle'] = True

    generate_training_data(**kwargs)


def generate_training_data(basepath, source_dir, target_dir, save_file, axes='CZYX', pattern='*.tif*', **kwargs):
    """
    Create normalized training data from pairs of corresponding TIFF images read from folders.

    Two images correspond to each other if they have the same file name, but are located in different folders.

    Parameters
    ----------
    basepath : str
        Base folder that contains sub-folders with images.
    source_dir : str
        Folder name relative to `basepath` that contain the source images (e.g., with low SNR).
    target_dir : str
        Folder name relative to `basepath` that contains the target images (e.g., with high SNR).
    axes : str
        Semantics of axes of loaded images (assumed to be the same for all images).
    pattern : str
        Glob-style pattern to match the desired TIFF images.

    Attributes
    ----------
    patch_size : tuple
        Shape of the patches to be extraced from raw images.
        Must be compatible with the number of dimensions and axes of the raw images.
        As a general rule, use a power of two along all XYZT axes, or at least divisible by 8.
    n_patches_per_image : int
        Number of patches to be sampled/extracted from each raw image pair (after transformations, see below).
    patch_axes : str or None
        Axes of the extracted patches. If ``None``, will assume to be equal to that of transformed raw data.
    save_file : str or None
        File name to save training data to disk in ``.npz`` format (see :func:`csbdeep.io.save_training_data`).
        If ``None``, data will not be saved.
    transforms : list or tuple, optional
        List of :class:`Transform` objects that apply additional transformations to the raw images.
        This can be used to augment the set of raw images (e.g., by including rotations).
        Set to ``None`` to disable. Default: ``None``.
    patch_filter : function, optional
        Function to determine for each image pair which patches are eligible to be extracted
        (default: :func:`no_background_patches`). Set to ``None`` to disable.
    normalization : function, optional
        Function that takes arguments `(patches_x, patches_y, x, y, mask, channel)`, whose purpose is to
        normalize the patches (`patches_x`, `patches_y`) extracted from the associated raw images
        (`x`, `y`, with `mask`; see :class:`RawData`). Default: :func:`norm_percentiles`.
    shuffle : bool, optional
        Randomly shuffle all extracted patches.

    """
    raw_data = RawData.from_folder(
        basepath=basepath,
        source_dirs=[source_dir],
        target_dir=target_dir,
        axes=axes,
        pattern=pattern
    )

    create_patches(raw_data, save_file=save_file, **kwargs)


if __name__ == '__main__':
    main()
