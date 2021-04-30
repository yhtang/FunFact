#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numbers
import warnings
import numpy as np
import torch
import tqdm
from symfac.util.iterable import flatten


class RBFExpansion:

    @staticmethod
    def _get_device(desc):
        if desc == 'auto':
            try:
                return torch.device('cuda')
            except:
                return torch.device('cpu')
        else:
            return torch.device(desc)

    def __init__(
        self, k, f='gauss', batch=64, max_steps=10000, loss='mse_loss',
        algorithm='Adam', lr=0.1, device='auto', progressbar='default'
    ):
        self.k = k

        if callable(f):
            self.f = f
        elif f == 'gauss':
            self.f = lambda d: torch.exp(-torch.square(d))
        else:
            raise f'Unrecoginized argument f = {f}.'

        self.batch = batch
        self.max_steps = max_steps

        if isinstance(loss, str):
            try:
                self.loss = getattr(torch.nn.functional, loss)
            except AttributeError:
                raise AttributeError(
                    f'The loss function \'{loss}\' does not exist in'
                    'torch.nn.functional.'
                )
        try:
            self.loss(torch.zeros(1), torch.zeros(1))
        except Exception as e:
            raise AssertionError(
                f'The given loss function does not accept two arguments:\n{e}'
            )

        if isinstance(algorithm, str):
            try:
                self.algorithm = getattr(torch.optim, algorithm)
            except AttributeError:
                raise AttributeError(
                    f'The algorithm \'{algorithm}\' does not exist in torch.optim.'
                )

        self.lr = lr

        self.device = self._get_device(device)

        if progressbar == 'default':
            self.progressbar = lambda n: tqdm.trange(
                n, miniters=None, mininterval=0.25, leave=False
            )

    @staticmethod
    def dist_uv(uv):
        return uv[0][:, :, None, :] - uv[1][:, None, :, :]

    def fit(self, target, x0=None, a0=None, b0=None):
        n, m = target.shape

        target = target[None, :, :]
        if not isinstance(target, torch.Tensor) or target.device != self.device:
            target = target.to(self.device)

        if x0 is None:
            x0 = (
                torch.randn((self.batch, n, self.k), requires_grad=True, device=self.device),
                torch.randn((self.batch, m, self.k), requires_grad=True, device=self.device),
            )
        elif not isinstance(x0, torch.Tensor) or x0.device != self.device:
            x0 = torch.Tensor(x0, requires_grad=True, device=self.device)

        if a0 is None:
            a0 = torch.randn((self.batch, self.k), requires_grad=True, device=self.device)
        elif not isinstance(a0, torch.Tensor) or a0.device != self.device:
            a0 = torch.Tensor(a0, requires_grad=True, device=self.device)

        if b0 is None:
            b0 = torch.randn(self.batch, requires_grad=True, device=self.device)
        elif not isinstance(b0, torch.Tensor) or b0.device != self.device:
            b0 = torch.Tensor(b0, requires_grad=True, device=self.device)

        (x, a, b), history = self._rbfexp_core(
            target=target,
            f=self.f,
            x=x0,
            a=a0,
            b=b0,
            dist=self.dist_uv,
            batch=self.batch,
            max_steps=self.max_steps,
            loss=self.loss,
            algorithm=self.algorithm,
            lr=self.lr,
            progressbar=self.progressbar,
            k=self.k,
            device=self.device
        )

        self.x = x
        self.a = a
        self.b = b

        return history


# def rbfexph(
#     target, k, f='gauss', batch=64, max_steps=10000, tol=1e-4, device='auto'
# ):
#     _rbfexp_core(
#         dist=lambda u: u[:, :, None] - u[:, None, :]
#     )

    @staticmethod
    def _rbfexp_core(
        target, f, x, a, b, dist, batch, max_steps, loss, algorithm, lr, progressbar,
        k, device
    ):
        try:
            # opt = algorithm(flatten([x, a, b]), lr=lr)
            opt = algorithm((*x, a, b), lr=lr)
            # opt = algorithm((a, b), lr=lr)
        except Exception:
            raise AssertionError(
                'Cannot instance optimizer of type {algorithm}:\n{e}'
            )

        print(target.shape)
        print(x[0].shape)
        print(x[1].shape)
        print(a.shape)
        print(b.shape)

        data_dim = list(range(1, len(target.shape)))
        loss_history = []
        for step in progressbar(max_steps):
            opt.zero_grad()            
            output = torch.sum(
                f(dist(x)) * a[:, None, None, :],
                dim=-1,
            ) + b[:, None, None]
            # output = torch.zeros(batch, *target.squeeze().shape, device=device)
            # for i in range(batch):
            #     for j in range(k):
            #         output[i, :, :] += f(x[0][i, :, None, j] - x[1][i, None, :, j]) * a[i, j]
            # print(f'output.shape {output.shape}')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                batch_loss = loss(output, target, reduction='none').mean(data_dim)
                # print(f'{loss(output, target, reduction="none").shape} batch_loss.shape {batch_loss.shape}')
            total_loss = batch_loss.sum()

            with torch.no_grad():
                loss_history.append(batch_loss.cpu().numpy())

            total_loss.backward()
            opt.step()

        return (x, a, b), np.array(loss_history, dtype=np.float)


# class RBFExpansion:
#     '''An approximation of a dense matrix as a sum of RBF over distance
#     matrices.
#     '''
#     def __init__(self, f, components):
#         pass

#     def fit(self, target):
#         pass

#     @property
#     def U(self):
#         pass

#     @property
#     def V(self):
#         pass
    
#     @property
#     def a(self):
#         pass

#     @property
#     def b(self):
#         pass
