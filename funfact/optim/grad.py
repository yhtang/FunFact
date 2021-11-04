#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
import tqdm
import numpy as np
import torch
import torch.nn.functional
import torch.optim
from funfact.util.iterable import as_namedtuple


def gradient_descent(
    f, target, lr, loss='mse_loss', algorithm='Adam', nsteps=10000,
    history_freq=None, progressbar='default'
):
    '''
    Optimize a factorization using gradient descent. A singleton dimension,
    i.e. a dimension with size 1, in ``target`` indicates that the
    factorization is in fact a parallel batch of trials aggregated along
    that dimension. For example, a ``target`` of shape (1, 4, 4) with a
    ``f.forward()`` result of shape (8, 4, 4) corresponds to 8 indenepdent
    trials for factorizing a 4-by-4 matrix.

    Parameters
    ----------
    f: callable
        A factorization object, which upon evaluation returns a reconstructed
        tensor, as created with :py:mod:`funfact.geneprog`.
    target: tensor
        The target tensor to be reconstructed.
    lr: float
        Learning rate of the algorithm.
    loss: str or callable
        The loss function for comparing the difference between the target and
        the reconstructed tensor. Could be either a callable that accepts 2
        arguments, or a string that names a loss function in
        :py:mod:`torch.nn.functional`.
    algorithm: str or class
        The algorithm for carrying out the actual gradient descent. Could be
        either a PyTorch optimizer class or a string that names an optimizer in
        :py:mod:`torch.optim`.
    nsteps: int
        Number of steps to run.
    history_frequency: int > 0
        The frequency to store the loss of the optimiation process.
    '''
    if isinstance(loss, str):
        try:
            loss = getattr(torch.nn.functional, loss)
        except AttributeError:
            raise AttributeError(
                f'The loss function \'{loss}\' does not exist in'
                'torch.nn.functional.'
            )
    try:
        loss(target, target)
    except Exception as e:
        raise AssertionError(
            f'The given loss function does not accept two arguments:\n{e}'
        )

    if isinstance(algorithm, str):
        try:
            algorithm = getattr(torch.optim, algorithm)
        except AttributeError:
            raise AttributeError(
                f'The algorithm \'{algorithm}\' does not exist in torch.optim.'
            )
    try:
        opt = algorithm(f.parameters, lr=lr)
    except Exception:
        raise AssertionError(
            'Invalid optimization algorithm:\n{e}'
        )

    if progressbar == 'default':
        def progressbar(n):
            return tqdm.trange(n, miniters=None, mininterval=0.25, leave=False)

    best = None
    data_dim = None
    loss_history = []
    for step in progressbar(nsteps):
        opt.zero_grad()
        output = f.forward()
        if data_dim is None:
            data_dim = torch.nonzero(
                torch.tensor(output.shape) == torch.tensor(target.shape)
            ).flatten().tolist()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            batch_loss = loss(output, target, reduction='none').mean(data_dim)
        total_loss = batch_loss.sum()

        with torch.no_grad():
            batch_loss_cpu = batch_loss.cpu().numpy()
            i = np.argmin(batch_loss_cpu)
            if best is None or batch_loss_cpu[i] < best.loss:
                best = as_namedtuple(
                    'best', x=f.flat_parameters, i=i, loss=batch_loss_cpu[i]
                )

            if history_freq is not None and (step + 1) % history_freq == 0:
                loss_history.append(batch_loss_cpu)

        total_loss.backward()
        opt.step()

    return as_namedtuple(
        'optimization_result',
        best_loss=best.loss,
        best_flat_parameters=best.x,
        best_i=best.i,
        loss_history=np.array(loss_history, dtype=np.float)
    )
