from __future__ import print_function, unicode_literals, absolute_import, division

import argparse
import os

import matplotlib.pyplot as plt
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
from csbdeep.utils import axes_dict, plot_history
from csbdeep.utils.tf import limit_gpu_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str,
                        help='File name for training data in ``.npz`` format')
    parser.add_argument('--model_basedir', type=str,
                        help='Path to model folder (which stores configuration, weights, etc.)')
    parser.add_argument('--model_name', type=str,
                        help='Model name (relative to ``model_basedir``')
    parser.add_argument('--validation_split', type=float, default=0.05,
                        help='Fraction of images to use as validation set during training')
    parser.add_argument('--limit_gpu', type=float, default=0.5,
                        help='Fraction of the GPU memory to use. Default: ``0.5``')
    parser.add_argument('--probabilistic', type=bool, default=False,
                        help='Probabilistic prediction of per-pixel Laplace distributions or '
                             'typical regression of per-pixel scalar values')
    parser.add_argument('--n_dim', type=int, default=3,
                        help='Dimensionality of input images (2 or 3)')
    parser.add_argument('--unet_residual', type=bool, default=True,
                        help='Parameter `residual` of :func:`csbdeep.nets.common_unet`. '
                             'Default: ``n_channel_in == n_channel_out``')
    parser.add_argument('--unet_n_depth', type=int, default=2,
                        help='Parameter `n_depth` of :func:`csbdeep.nets.common_unet`. Default: ``2``')
    parser.add_argument('--unet_kern_size', type=int, default=3,
                        help='Parameter `kern_size` of :func:`csbdeep.nets.common_unet`. '
                             'Default: ``5 if n_dim==2 else 3``')
    parser.add_argument('--unet_n_first', type=int, default=32,
                        help='Parameter `n_first` of :func:`csbdeep.nets.common_unet`. Default: ``32``')
    parser.add_argument('--unet_last_activation', type=str, default='linear',
                        help='Parameter `last_activation` of :func:`csbdeep.nets.common_unet`. Default: ``linear``')
    parser.add_argument('--train_loss', type=str, default='mae',
                        help='Name of training loss. Default: ``laplace`` if ``probabilistic`` else ``mae``')
    parser.add_argument('--train_epochs', type=int, default=100,
                        help='Number of training epochs. Default: ``100``')
    parser.add_argument('--train_steps_per_epoch', type=int, default=400,
                        help='Number of parameter update steps per epoch. Default: ``400``')
    parser.add_argument('--train_learning_rate', type=float, default=0.0004,
                        help='Learning rate for training. Default: ``0.0004``')
    parser.add_argument('--train_batch_size', type=int, default=16,
                        help='Batch size for training. Default: ``16``')
    parser.add_argument('--train_tensorboard', type=bool, default=True,
                        help='Enable TensorBoard for monitoring training progress. Default: ``True``')
    parser.add_argument('--train_checkpoint', type=str, default='weights_best.h5',
                        help='Name of checkpoint file for model weights (only best are saved); '
                             'set to ``None`` to disable. Default: ``weights_best.h5``')

    kwargs = vars(parser.parse_args())
    train(**kwargs)


def train(data_file, validation_split, model_name, model_basedir, limit_gpu, **kwargs):
    """

    Parameters
    ----------
    data_file : str
        File name for training data in ``.npz`` format
    validation_split : float
        Fraction of images to use as validation set during training.
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
    probabilistic : bool
        Probabilistic prediction of per-pixel Laplace distributions or
        typical regression of per-pixel scalar values.
    n_dim : int
        Dimensionality of input images (2 or 3).
    unet_residual : bool
        Parameter `residual` of :func:`csbdeep.nets.common_unet`. Default: ``n_channel_in == n_channel_out``
    unet_n_depth : int
        Parameter `n_depth` of :func:`csbdeep.nets.common_unet`. Default: ``2``
    unet_kern_size : int
        Parameter `kern_size` of :func:`csbdeep.nets.common_unet`. Default: ``5 if n_dim==2 else 3``
    unet_n_first : int
        Parameter `n_first` of :func:`csbdeep.nets.common_unet`. Default: ``32``
    unet_last_activation : str
        Parameter `last_activation` of :func:`csbdeep.nets.common_unet`. Default: ``linear``
    train_loss : str
        Name of training loss. Default: ``'laplace' if probabilistic else 'mae'``
    train_epochs : int
        Number of training epochs. Default: ``100``
    train_steps_per_epoch : int
        Number of parameter update steps per epoch. Default: ``400``
    train_learning_rate : float
        Learning rate for training. Default: ``0.0004``
    train_batch_size : int
        Batch size for training. Default: ``16``
    train_tensorboard : bool
        Enable TensorBoard for monitoring training progress. Default: ``True``
    train_checkpoint : str
        Name of checkpoint file for model weights (only best are saved); set to ``None`` to disable.
        Default: ``weights_best.h5``
    train_reduce_lr : dict
        Parameter :class:`dict` of ReduceLROnPlateau_ callback; set to ``None`` to disable.
        Default: ``{'factor': 0.5, 'patience': 10, 'min_delta': 0}``

        .. _ReduceLROnPlateau: https://keras.io/callbacks/#reducelronplateau
    """
    (X, Y), (X_val, Y_val), axes = load_training_data(data_file, validation_split=validation_split, verbose=True)
    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

    limit_gpu_memory(fraction=limit_gpu)
    config = Config(axes, n_channel_in, n_channel_out, **kwargs)
    model = CARE(config, model_name, basedir=model_basedir)
    history = model.train(X, Y, validation_data=(X_val, Y_val))
    plt.figure(figsize=(16, 5))
    plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mae', 'val_mae'])
    plt.savefig(os.path.join(model_basedir, model_name, 'history.png'))


if __name__ == '__main__':
    main()
