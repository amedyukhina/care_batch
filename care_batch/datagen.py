
from csbdeep.data import RawData, create_patches


def datagen(basepath, source_dir, target_dir, save_file, axes='CZYX', pattern='*.tif*',
            normalize_patches=False, **kwargs):
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
    normalize_patches : bool
        If True, extracted patches will be normalized by the default CARE normalizer (:func:`norm_percentiles`).
        If False, no normalization will be done.
        Default is False.

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
    if not normalize_patches:
        kwargs['normalization'] = None

    create_patches(raw_data, save_file=save_file, **kwargs)
