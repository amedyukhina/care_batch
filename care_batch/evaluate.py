import os
import pandas as pd
from skimage import io

import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def rmse(gt: np.ndarray, img: np.ndarray) -> float:
    volume = np.prod(gt.shape)
    return np.sqrt(np.sum((gt - img) ** 2) / volume)


def nrmse(gt: np.ndarray, img: np.ndarray) -> float:
    err = rmse(gt, img)
    return err / (np.max(gt) - np.min(gt))


def ssim(gt: np.ndarray, img: np.ndarray) -> float:
    return structural_similarity(gt * 1., img * 1., full=False)


def psnr(gt: np.ndarray, img: np.ndarray) -> float:
    return peak_signal_noise_ratio(gt * 1., img * 1., data_range=np.max(gt) - np.min(gt))


def evaluate(input_dir, gt_dir, output_fn, model_name=None, pair_name=None):
    files = os.listdir(gt_dir)

    cols = ['RMSE', 'NRMSE', 'pSNR', 'SSIM']
    functions = [rmse, nrmse, psnr, ssim]
    fns = []
    data = []
    for fn in files:
        gt = io.imread(os.path.join(gt_dir, fn))
        if os.path.exists(os.path.join(input_dir, fn)):
            img = io.imread(os.path.join(input_dir, fn))
            cur_data = [func(gt, img) for func in functions]
            data.append(cur_data)
            fns.append(fn)  # new filename list in case some images are missing
    df = pd.DataFrame(data, columns=cols)
    df['image name'] = fns
    if model_name is not None:
        df['model'] = model_name
    if pair_name is not None:
        df['pair'] = pair_name
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)
    df.to_csv(output_fn, index=False)


def summarize_stats(input_fns, output_fn):
    df = [pd.read_csv(fn) for fn in input_fns]
    df = pd.concat(df, ignore_index=True)
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)
    df.to_csv(output_fn)
