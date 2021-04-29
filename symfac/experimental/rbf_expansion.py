#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
import numpy as np
import torch
import tqdm


def _get_device(desc):
    if desc == 'auto':
        try:
            return torch.device('cuda')
        except:
            return torch.device('cpu')
    else:
        return torch.device(desc)


def rbfexp(
    target, k, x0=None, a0=None, b0=None, f='gauss', batch=64, max_steps=10000, loss='mse_loss',
    algorithm='Adam', lr=0.1, device='auto', progressbar='default'
):
    n, m = target.shape
    d = _get_device(device)

    if x0 is None:
        x0 = (
            torch.randn((batch, n, k), requires_grad=True, device=d),
            torch.randn((batch, m, k), requires_grad=True, device=d),
        )
    elif not isinstance(x0, torch.Tensor):
        x0 = torch.Tensor()
    
    if a0 is None:
        a0 = torch.randn((batch, n, k), requires_grad=True, device=d),

    x, history = _rbfexp_core(
        target=target[None, :, :].to(d),
        f=f,
        x=x0,
        a=a0,
        b=b0,
        dist=lambda uv: uv[0][:, :, None] - uv[1][:, None, :],
        batch=batch,
        max_steps=max_steps,
        loss=loss,
        algorithm=algorithm,
        lr=lr,
        progressbar=progressbar
    )

    return [
        RBFExpansion(f, x[0][i, :, :], x[1][i, :, :]) for i in range(batch)
    ], history


def rbfexph(
    target, k, f='gauss', batch=64, max_steps=10000, tol=1e-4, device='auto'
):
    _rbfexp_core(
        dist=lambda u: u[:, :, None] - u[:, None, :]
    )


def _rbfexp_core(
    target,
    f,
    x,
    dist,
    batch,
    max_steps,
    loss,
    algorithm,
    lr,
    progressbar
):
    if callable(f):
        fx = f
    elif f == 'gauss':
        def fx(d):
            return torch.exp(-torch.square(d))
    else:
        raise f'Unrecoginized argument f = {f}.'

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
        opt = algorithm(x, lr=lr)
    except Exception:
        raise AssertionError(
            'Invalid optimization algorithm:\n{e}'
        )

    if progressbar == 'default':
        def progressbar(n):
            return tqdm.trange(n, miniters=None, mininterval=0.25, leave=False)

    data_dim = None
    loss_history = []
    for step in progressbar(max_steps):
        opt.zero_grad()
        output = fx(dist(x))
        if data_dim is None:
            data_dim = torch.nonzero(
                torch.tensor(output.shape) == torch.tensor(target.shape)
            ).flatten().tolist()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            batch_loss = loss(output, target, reduction='none').mean(data_dim)
        total_loss = batch_loss.sum()

        with torch.no_grad():
            loss_history.append(batch_loss.cpu().numpy())

        total_loss.backward()
        opt.step()

    return x, np.array(loss_history, dtype=np.float)


class RBFExpansion:
    '''An approximation of a dense matrix as a sum of RBF over distance
    matrices.
    '''
    def __init__(self, f, components):
        pass

    def fit(self, target):
        pass

    @property
    def U(self):
        pass

    @property
    def V(self):
        pass
    
    @property
    def a(self):
        pass

    @property
    def b(self):
        pass
