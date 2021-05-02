#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numbers
import warnings
import numpy as np
import torch
import tqdm


class RBFExpansion:

    @property
    def result(self):
        return self._result

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
        self, k, rbf='gauss', batch=64, max_steps=10000, loss='mse_loss',
        algorithm='Adam', lr=0.1, device='auto', progressbar='default'
    ):
        self.k = k

        if callable(rbf):
            self.rbf = rbf
        elif rbf == 'gauss':
            self.rbf = lambda d: torch.exp(-torch.square(d))
        else:
            raise f'Unrecoginized RBF: {rbf}.'

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

    def fit(self, target, u0=None, v0=None, a0=None, b0=None):

        def f(u, v, a, b):
            return torch.sum(
                self.rbf(u[..., :, None, :] - v[..., None, :, :]) *
                a[..., None, None, :],
                dim=-1
            ) + b[..., None, None]

        n, m = target.shape

        target = torch.as_tensor(target[None, :, :], device=self.device)

        def _create(w, *shape):
            if w is None:
                return torch.randn(shape, requires_grad=True, device=self.device)
            else:
                return torch.as_tensor(w, device=self.device)

        u0 = _create(u0, self.batch, n, self.k)
        v0 = _create(v0, self.batch, m, self.k)
        a0 = _create(a0, self.batch, self.k)
        b0 = _create(b0, self.batch)

        self._optimum = self._rbfexp_core(
            target, u0, v0, a0, b0,
            f=f,
            batch=self.batch,
            max_steps=self.max_steps,
            loss=self.loss,
            algorithm=self.algorithm,
            lr=self.lr,
            progressbar=self.progressbar
        )

        self._batch = self.Batch(f, self._optimum['x'], ['u', 'v', 'a', 'b'])
        # return self.Batch(f, optimum)

        return self

    @staticmethod
    def _rbfexp_core(target, *x, **options):

        f = options.pop('f')
        batch = options.pop('batch')
        max_steps = options.pop('max_steps')
        loss = options.pop('loss')
        algorithm = options.pop('algorithm')
        lr = options.pop('lr')
        progressbar = options.pop('progressbar')

        try:
            opt = algorithm(x, lr=lr)
        except Exception:
            raise AssertionError(
                'Cannot instance optimizer of type {algorithm}:\n{e}'
            )

        data_dim = list(range(1, len(target.shape)))
        optimum = {}
        optimum['x'] = [w.clone().detach() for w in x]
        optimum['t'] = torch.zeros(batch, dtype=torch.int)
        optimum['history'] = []
        for step in progressbar(max_steps):
            opt.zero_grad()
            output = f(*x)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                loss_batch = loss(output, target, reduction='none').mean(data_dim)

            with torch.no_grad():
                loss_cpu = loss_batch.detach().cpu()
                optimum['history'].append(loss_cpu.numpy())
                if 'best' in optimum:
                    optimum['best'] = torch.minimum(optimum['best'], loss_cpu)
                else:
                    optimum['best'] = loss_cpu
                better = optimum['best'] == loss_cpu
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
        class RBFSeries:
            def __init__(self, parent, i):
                self.batch = parent
                self.i = i

            def __repr__(self):
                return f'<RBF expansion of {self.batch.funrank} components of {", ".join(self.batch.vars)}>'

            def __getattr__(self, a):
                return self.batch.x[self.batch.vars.index(a)][self.i, ...]

            def __call__(self, device='cpu'):
                with torch.no_grad():
                    return self.batch.f(*[
                        w[self.i, ...] for w in self.batch.x
                    ]).to(device)

        def __init__(self, f, x, vars):
            for w in x:
                assert(
                    w.shape[0] == x[0].shape[0] and
                    w.shape[-1] == x[0].shape[-1],
                    "Inconsisent component size."
                )
            self.f = f
            self.x = x
            self.vars = vars

        def __repr__(self):
            return f'<batch of {len(self)} RBF expansions with {", ".join(self.vars)}>'

        def __len__(self):
            return len(self.x[0])

        def __call__(self, device='cpu'):
            with torch.no_grad():
                return self.f(*self.x).to(device)

        @property
        def funrank(self):
            return len(self.x[-1])

        def __getitem__(self, i):
            return self.RBFSeries(self, i)

        def __getattr__(self, a):
            return self.x[self.vars.index(a)]
