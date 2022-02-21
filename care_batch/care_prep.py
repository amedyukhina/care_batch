import os
import numpy as np
from skimage import io
from tqdm import tqdm
from am_utils.utils import walk_dir


def __copy_and_normalize(fn_in, fn_out, maxval=255):
    img = io.imread(fn_in)
    img = img - img.min()
    img = img.astype(np.float) / np.max(img) * maxval
    if maxval > 255:
        dtype = np.uint16
    else:
        dtype = np.uint8
    io.imsave(fn_out, img.astype(dtype))


def care_prep(input_pair, output_dir, name_high='high', name_low='low', normalize=False, maxval=255):
    input_high, input_low = input_pair
    for dir_in, dir_out in zip([input_high, input_low],
                               [name_high, name_low]):
        os.makedirs(os.path.join(output_dir, dir_out), exist_ok=True)
        for fn in tqdm(walk_dir(dir_in)):
            fn_out = os.path.join(output_dir, dir_out, fn[len(dir_in.rstrip('/'))+1:].replace('/', '_'))
            if os.path.exists(fn_out):
                os.remove(fn_out)
            if normalize:
                __copy_and_normalize(fn_in, fn_out, maxval)
            else:
                os.symlink(fn, fn_out)
