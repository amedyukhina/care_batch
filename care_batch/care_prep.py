import os

from am_utils.utils import walk_dir
from skimage import io
from tqdm import tqdm
import numpy as np

from .utils import int_type, normalize


def __copy_and_normalize(fn_in, fn_out, **kwargs):
    img = io.imread(fn_in)
    img = normalize(img, **kwargs)
    io.imsave(fn_out, img.astype(int_type(img)))


def __copy_file_list(files, output_dir, dir_in, dir_out, normalize_image, **norm_kwargs):
    if len(files) > 0:
        os.makedirs(os.path.join(output_dir, dir_out), exist_ok=True)
        for fn in tqdm(files):
            fn_out = os.path.join(output_dir, dir_out, fn.replace('/', '_'))
            if os.path.exists(fn_out):
                os.remove(fn_out)
            if normalize_image:
                __copy_and_normalize(os.path.join(dir_in, fn), fn_out, **norm_kwargs)
            else:
                os.symlink(os.path.join(dir_in, fn), fn_out)


def care_prep(input_pair, output_dir, name_high='high', name_low='low',
              test_fraction=0, validation_fraction=0,
              name_train='train', name_validation='validation', name_test='test',
              normalize_image=True, **norm_kwargs):
    input_high, input_low = input_pair
    files = [fn[len(input_high.rstrip('/'))+1:] for fn in walk_dir(input_high)]
    files = [fn for fn in files if os.path.exists(os.path.join(input_low, fn))]
    if test_fraction == 0 and validation_fraction == 0:
        file_list = [files]
        train_test_dirs = ['']
    else:
        train_test_dirs = [name_train, name_validation, name_test]
        files = np.random.choice(files, len(files), replace=False)
        ntest = int(round(test_fraction * len(files)))
        nval = int(round(validation_fraction * len(files)))
        file_list = [files[ntest+nval:],
                     files[ntest:ntest+nval],
                     files[:ntest]]

    for i in range(len(file_list)):
        for dir_in, dir_out in zip([input_high, input_low],
                                   [name_high, name_low]):
            __copy_file_list(file_list[i], os.path.join(output_dir, train_test_dirs[i]),
                             dir_in, dir_out, normalize_image, **norm_kwargs)
