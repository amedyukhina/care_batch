import os

from am_utils.utils import walk_dir
from skimage import io
from tqdm import tqdm

from .utils import int_type, normalize


def __copy_and_normalize(fn_in, fn_out, **kwargs):
    img = io.imread(fn_in)
    img = normalize(img, **kwargs)
    io.imsave(fn_out, img.astype(int_type(img)))


def care_prep(input_pair, output_dir, name_high='high', name_low='low', normalize=True, **norm_kwargs):
    input_high, input_low = input_pair
    for dir_in, dir_out in zip([input_high, input_low],
                               [name_high, name_low]):
        os.makedirs(os.path.join(output_dir, dir_out), exist_ok=True)
        for fn in tqdm(walk_dir(dir_in)):
            fn_out = os.path.join(output_dir, dir_out, fn[len(dir_in.rstrip('/')) + 1:].replace('/', '_'))
            if os.path.exists(fn_out):
                os.remove(fn_out)
            if normalize:
                __copy_and_normalize(fn, fn_out, **norm_kwargs)
            else:
                os.symlink(fn, fn_out)
