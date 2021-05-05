#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import warnings
import numpy as np
import torch
import tqdm


def as_namedtuple(name, **kwargs):
    return namedtuple(name, list(kwargs.keys()))(*kwargs.values())


class RBFExpansion:

    @staticmethod
    def _get_device(desc):
        if desc == 'auto':
            try:
                return torch.device('cuda')
            except Exception:
                return torch.device('cpu')
        else:
            return torch.device(desc)

    def __init__(
        self, k, rbf='gauss', batch_size=64, max_steps=10000, loss='mse_loss',
        algorithm='Adam', lr=0.1, device='auto', progressbar='default'
    ):
        self.k = k

        if callable(rbf):
            self.rbf = rbf
        elif rbf == 'gauss':
            self.rbf = lambda d: torch.exp(-torch.square(d))
        else:
            raise f'Unrecoginized RBF: {rbf}.'

        self.batch_size = batch_size
        self.max_steps = max_steps

        if isinstance(loss, str):
            try:
                self.loss = getattr(torch.nn.functional, loss)
            except AttributeError:
                raise AttributeError(
                    f'The loss function \'{loss}\' does not exist in'
                    'torch.nn.functional.'
                )
        else:
            self.loss = loss
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
                    f'Cannot find \'{algorithm}\' in torch.optim.'
                )
        else:
            self.algorithm = algorithm

        self.lr = lr

        self.device = self._get_device(device)

        if progressbar == 'default':
            self.progressbar = lambda n: tqdm.trange(
                n, miniters=None, mininterval=0.25, leave=True
            )
        else:
            self.progressbar = progressbar

    @property
    def config(self):
        return {
            key: self.__dict__[key] for key in self.__dict__
            if not key.startswith('_')
        }

    def randn(self, *shape, requires_grad=True):
        return torch.randn(
            shape, requires_grad=requires_grad, device=self.device
        )

    def as_tensor(self, tsr):
        return torch.as_tensor(tsr, device=self.device).requires_grad_(True)

    def fit(self, target, u0=None, v0=None, a0=None, b0=None, seed=None):

        with torch.random.fork_rng(devices=[self.device]):
            if seed:
                torch.random.manual_seed(seed)

            target = torch.as_tensor(target, device=self.device).unsqueeze(0)
            _, n, m = target.shape

            u0 = self.as_tensor(u0) if u0 is not None else \
                self.randn(self.batch_size, n, self.k)
            v0 = self.as_tensor(v0) if v0 is not None else \
                self.randn(self.batch_size, m, self.k)
            a0 = self.as_tensor(a0) if a0 is not None else \
                self.randn(self.batch_size, self.k)
            b0 = self.as_tensor(b0) if b0 is not None else \
                self.randn(self.batch_size)

            def f(u, v, a, b):
                return torch.sum(
                    self.rbf(u[..., :, None, :] - v[..., None, :, :]) *
                    a[..., None, None, :],
                    dim=-1
                ) + b[..., None, None]

            self.optimum = self._grad_opt(target, f, u0, v0, a0, b0)

            self.result = (f, self.optimum.x, ['u', 'v', 'a', 'b'])

            return self

    def fith(self, target, u0=None, a0=None, b0=None, seed=None):

        with torch.random.fork_rng(devices=[self.device]):
            if seed:
                torch.random.manual_seed(seed)

            target = torch.as_tensor(target, device=self.device).unsqueeze(0)
            _, n, m = target.shape
            assert n == m

            u0 = self.as_tensor(u0) if u0 is not None else \
                self.randn(self.batch_size, n, self.k)
            a0 = self.as_tensor(a0) if a0 is not None else \
                self.randn(self.batch_size, self.k)
            b0 = self.as_tensor(b0) if b0 is not None else \
                self.randn(self.batch_size)

            def f(u, a, b):
                return torch.sum(
                    self.rbf(u[..., :, None, :] - u[..., None, :, :]) *
                    a[..., None, None, :],
                    dim=-1
                ) + b[..., None, None]

            self.optimum = self._grad_opt(target, f, u0, a0, b0)

            self.result = (f, self.optimum.x, ['u', 'a', 'b'])

            return self

    def fit_custom(self, target, f, seed=None, **x0):

        with torch.random.fork_rng(devices=[self.device]):
            if seed:
                torch.random.manual_seed(seed)

            target = torch.as_tensor(target, device=self.device).unsqueeze(0)
            _, n, m = target.shape

            x = [self.as_tensor(w) for w in x0.values()]

            self.optimum = self._grad_opt(target, f, *x)

            self.result = (f, self.optimum.x, list(x0.keys()))

            return self

    @property
    def optimum(self):
        return self._optimum

    @optimum.setter
    def optimum(self, opt_dict):
        self._optimum = as_namedtuple('optimum', **opt_dict)

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, fxv):
        self._result = self.Batch(*fxv)

    def _grad_opt(self, target, f, *x):

        try:
            opt = self.algorithm(x, lr=self.lr)
        except Exception:
            raise AssertionError(
                'Cannot instance optimizer of type {self.algorithm}:\n{e}'
            )

        data_dim = list(range(1, len(target.shape)))
        optimum = {}
        optimum['x'] = [w.clone().detach() for w in x]
        optimum['t'] = torch.zeros(self.batch_size, dtype=torch.int)
        optimum['history'] = []
        for step in self.progressbar(self.max_steps):
            opt.zero_grad()
            output = f(*x)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                loss_batch = self.loss(
                    output, target, reduction='none'
                ).mean(
                    data_dim
                )

            with torch.no_grad():
                loss_cpu = loss_batch.detach().cpu()
                optimum['history'].append(loss_cpu.numpy())
                if 'best_loss' in optimum:
                    optimum['best_loss'] = torch.minimum(
                        optimum['best_loss'], loss_cpu
                    )
                else:
                    optimum['best_loss'] = loss_cpu
                better = optimum['best_loss'] == loss_cpu
                optimum['t'][better] = step
                for current, new in zip(optimum['x'], x):
                    current[better, ...] = new[better, ...]

            loss_batch.sum().backward()
            opt.step()

        optimum['history'] = np.array(optimum['history'], dtype=np.float)
        return optimum

    class Batch:
        '''An approximation of a dense matrix as a sum of RBF over distance
        matrices.
        '''

        def __init__(self, f, x, vars):
            for w in x:
                assert w.shape[0] == x[0].shape[0],\
                    "Inconsisent component size."
            self.f = f
            self.x = x
            self.vars = vars

        def __repr__(self):
            vars_str = ', '.join(self.vars)
            return f'<batch of {len(self)} RBF expansions [vars = {vars_str}]>'

        def __len__(self):
            return len(self.x[0])

        def __call__(self, runs=None, components=None, device='cpu'):
            with torch.no_grad():
                x = self.x
                if runs is not None:
                    x = [w[runs, ...] for w in x]
                if components is not None:
                    components = torch.as_tensor(components)
                    if components.dim() == 0:
                        components = components.unsqueeze(0)
                    x = [w[..., components]
                         if len(w.shape) > 0 else torch.zeros_like(w)
                         for w in x]
                return self.f(*x).to(device)

        @property
        def funrank(self):
            return len(self.x[-1])

        def __getattr__(self, a):
            return self.x[self.vars.index(a)]
