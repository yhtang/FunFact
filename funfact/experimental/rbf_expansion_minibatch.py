#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import numpy as np
import torch
import tqdm
import warnings
from .rbf_expansion_base import RBFExpansionBase


class RBFExpansionMiniBatch(RBFExpansionBase):

    def __init__(
        self, k=1, mini_batch_size=1, rbf='gauss', batch_size=64,
        max_steps=10000, history_freq=10, mini_batch_by='elements',
        loss='mse_loss', algorithm='Adam', lr=0.05, device='auto',
        progressbar='default'
    ):
        self.k = k
        self.mini_batch_size = mini_batch_size

        if callable(rbf):
            self.rbf = rbf
        elif rbf == 'gauss':
            self.rbf = lambda d: torch.exp(-torch.square(d))
        else:
            raise f'Unrecoginized RBF: {rbf}.'

        self.batch_size = batch_size
        self.max_steps = max_steps
        self.history_freq = history_freq
        self.mini_batch_by = mini_batch_by

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

    def fit(self, target, u0=None, v0=None, a0=None, b0=None, seed=None):

        with torch.random.fork_rng(devices=[self.device]):
            if seed:
                torch.random.manual_seed(seed)

            target = torch.as_tensor(target, device=self.device).unsqueeze(0)
            _, n, m = target.shape

            u0 = self.as_tensor(u0) if u0 is not None else \
                self.randn(self.batch_size, n, self.k, requires_grad=True)
            v0 = self.as_tensor(v0) if v0 is not None else \
                self.randn(self.batch_size, m, self.k, requires_grad=True)
            a0 = self.as_tensor(a0) if a0 is not None else \
                self.randn(self.batch_size, self.k, requires_grad=True)
            b0 = self.as_tensor(b0) if b0 is not None else \
                self.randn(self.batch_size, requires_grad=True)

            def f(u, v, a, b):
                return torch.sum(
                    self.rbf(u[..., :, None, :] - v[..., None, :, :]) *
                    a[..., None, None, :],
                    dim=-1
                ) + b[..., None, None]

            def loss_minibatch(target, i, j, u, v, a, b):
                return self.loss(
                    target[..., i, j],
                    torch.sum(
                        self.rbf(u[..., i, :] - v[..., j, :]) *
                        a[..., None, :],
                        dim=-1
                    ) + b[..., None],
                    reduction='mean'
                )

            self.report = self._grad_opt(
                target,
                loss_minibatch,
                self.Model(f, (u0, v0, a0, b0), default_grad_on=True)
            )

            self._optimum = self.Model(
                f, self.report.x_best, x_names=['u', 'v', 'a', 'b'],
                default_device='cpu'
            )

            return self

    # def fith(self, target, u0=None, a0=None, b0=None, seed=None):

    #     with torch.random.fork_rng(devices=[self.device]):
    #         if seed:
    #             torch.random.manual_seed(seed)

    #         target = torch.as_tensor(target, device=self.device).unsqueeze(0)
    #         _, n, m = target.shape
    #         assert n == m

    #         u0 = self.as_tensor(u0) if u0 is not None else \
    #             self.randn(self.batch_size, n, self.k)
    #         a0 = self.as_tensor(a0) if a0 is not None else \
    #             self.randn(self.batch_size, self.k)
    #         b0 = self.as_tensor(b0) if b0 is not None else \
    #             self.randn(self.batch_size)

    #         def f(u, a, b):
    #             return torch.sum(
    #                 self.rbf(u[..., :, None, :] - u[..., None, :, :]) *
    #                 a[..., None, None, :],
    #                 dim=-1
    #             ) + b[..., None, None]

    #         self.report = self._grad_opt(
    #             target,
    #             self.Model(f, (u0, a0, b0), default_grad_on=True)
    #         )

    #         self._optimum = self.Model(
    #             f, self.report.x_best, x_names=['u', 'a', 'b'],
    #             default_device='cpu'
    #         )

    #         return self

    def fit_custom(self, target, f, f_minibatch, seed=None, **x0):

        with torch.random.fork_rng(devices=[self.device]):
            if seed:
                torch.random.manual_seed(seed)

            target = torch.as_tensor(target, device=self.device).unsqueeze(0)
            _, n, m = target.shape

            x = [self.as_tensor(w) for w in x0.values()]

            def loss_minibatch(target, i, j, *x):
                return self.loss(
                    target[..., i, j],
                    f_minibatch(i, j, *x),
                    reduction='mean'
                )

            self.report = self._grad_opt(
                target,
                loss_minibatch,
                self.Model(f, x, default_grad_on=True)
            )

            self._optimum = self.Model(
                f, self.report.x_best, x_names=list(x0.keys()),
                default_device='cpu'
            )

            return self

    def _random_matrix_elements(self, n, m, b):
        ind, _ = torch.sort(torch.randperm(n * m, device=self.device)[:b])
        return ind % n, ind // n

    def _random_submatrix(self, n, m, b):
        if isinstance(b, int):
            b = [b, b]
        else:
            assert len(b) == 2
        i, _ = torch.sort(torch.randperm(n, device=self.device)[:b[0]])
        j, _ = torch.sort(torch.randperm(m, device=self.device)[:b[1]])
        return list(zip(*list(itertools.product(i, j))))

    def _grad_opt(self, target, loss_minibatch, model):

        try:
            opt = self.algorithm(model.x, lr=self.lr)
        except Exception:
            raise AssertionError(
                'Cannot instance optimizer of type {self.algorithm}:\n{e}'
            )

        _, n, m = target.shape
        if self.mini_batch_by in ['element', 'elements']:
            def indexer():
                return self._random_matrix_elements(n, m, self.mini_batch_size)
        elif self.mini_batch_by in ['submatrix', 'submatrices']:
            def indexer():
                return self._random_submatrix(n, m, self.mini_batch_size)
        else:
            raise RuntimeError(
                f'Unknown mini batch style: {self.mini_batch_by}'
            )

        data_dim = list(range(1, len(target.shape)))
        report = {}
        report['x_best'] = [w.clone().detach() for w in model.x]
        report['t_best'] = torch.zeros(self.batch_size, dtype=torch.int)
        report['loss_history'] = []
        report['loss_history_ticks'] = []
        for step in self.progressbar(self.max_steps):
            if step % self.history_freq == 0:
                with torch.no_grad():
                    output = model()
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        loss_cpu = self.loss(
                            output, target, reduction='none'
                        ).mean(
                            data_dim
                        ).cpu()
                    report['loss_history'].append(loss_cpu.numpy())
                    if 'loss_best' in report:
                        report['loss_best'] = torch.minimum(
                            report['loss_best'], loss_cpu
                        )
                    else:
                        report['loss_best'] = loss_cpu
                    better = report['loss_best'] == loss_cpu
                    report['loss_history_ticks'].append(step)
                    report['t_best'][better] = step
                    for current, new in zip(report['x_best'], model.x):
                        current[better, ...] = new[better, ...]

            i, j = indexer()
            opt.zero_grad()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                loss_minibatch(target, i, j, *model.x).backward()
            opt.step()

        report['loss_history'] = np.array(
            report['loss_history'], dtype=np.float
        )

        return report
