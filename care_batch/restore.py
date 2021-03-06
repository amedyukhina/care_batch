import os

from am_utils.utils import walk_dir, imsave
from csbdeep.models import CARE
from csbdeep.utils.tf import limit_gpu_memory
from skimage import io
from tqdm import tqdm

from .utils import int_type, normalize


def restore(input_dir, output_dir, model_name, model_basedir, limit_gpu=0.5,
            normalize_image=True, maxval=255, pmin=0, pmax=100, **kwargs):
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
    normalize_image : bool
        If True, the entire image will be normalized before restoration and no CARE patch normalization will be done.
        If False, the default CARE patch normalization will be done.
        Set to True, if the images were normalized before training data generation and no patch normalization was used.
        Default is True.
    maxval : int, optional
        Maximum value for the normalized image.
        Should be the same as in `care_prep`
        Default is 255.
    pmin : scalar, optional
        Lower percentile for normalization.
        Default is 0 (minimum value).
    pmax : scalar, optional
        Upper percentile for normalization.
        Default is 100 (maximum value).
    kwargs : key value
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
        if normalize_image:
            x = normalize(x, maxval, pmin=pmin, pmax=pmax)
            kwargs['normalizer'] = None
        restored = model.predict(x, **kwargs)
        imsave(output_fn, restored.astype(int_type(restored)))
