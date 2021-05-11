#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import warnings
import numpy as np
import torch
import tqdm


def as_namedtuple(name, **kwargs):
    return namedtuple(name, list(kwargs.keys()))(*kwargs.values())


class RBFExpansion2:

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

            self.report = self._grad_opt(
                target,
                self.Model(f, (u0, v0, a0, b0), default_grad_on=True)
            )

            self._optimum = self.Model(
                f, self.report.x_best, x_names=['u', 'v', 'a', 'b'],
                default_device='cpu'
            )

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

            self.report = self._grad_opt(
                target,
                self.Model(f, (u0, a0, b0), default_grad_on=True)
            )

            self._optimum = self.Model(
                f, self.report.x_best, x_names=['u', 'a', 'b'],
                default_device='cpu'
            )

            return self

    def fit_custom(self, target, f, seed=None, **x0):

        with torch.random.fork_rng(devices=[self.device]):
            if seed:
                torch.random.manual_seed(seed)

            target = torch.as_tensor(target, device=self.device).unsqueeze(0)
            _, n, m = target.shape

            x = [self.as_tensor(w) for w in x0.values()]

            self.report = self._grad_opt(
                target,
                self.Model(f, x, default_grad_on=True)
            )

            self._optimum = self.Model(
                f, self.report.x_best, x_names=list(x0.keys()),
                default_device='cpu'
            )

            return self

    @property
    def report(self):
        return self._report

    @report.setter
    def report(self, opt_dict):
        self._report = as_namedtuple('report', **opt_dict)

    @property
    def optimum(self):
        return self._optimum

    def _grad_opt(self, target, model):

        try:
            opt = self.algorithm(model.x, lr=self.lr)
        except Exception:
            raise AssertionError(
                'Cannot instance optimizer of type {self.algorithm}:\n{e}'
            )

        data_dim = list(range(1, len(target.shape)))
        report = {}
        report['x_best'] = [w.clone().detach() for w in model.x]
        report['t_best'] = torch.zeros(self.batch_size, dtype=torch.int)
        report['loss_history'] = []
        for step in self.progressbar(self.max_steps):
            opt.zero_grad()
            output = model()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                loss_batch = self.loss(
                    output, target, reduction='none'
                ).mean(
                    data_dim
                )

            with torch.no_grad():
                loss_cpu = loss_batch.detach().cpu()
                report['loss_history'].append(loss_cpu.numpy())
                if 'loss_best' in report:
                    report['loss_best'] = torch.minimum(
                        report['loss_best'], loss_cpu
                    )
                else:
                    report['loss_best'] = loss_cpu
                better = report['loss_best'] == loss_cpu
                report['t_best'][better] = step
                for current, new in zip(report['x_best'], model.x):
                    current[better, ...] = new[better, ...]

            loss_batch.sum().backward()
            opt.step()

        report['loss_history'] = np.array(
            report['loss_history'], dtype=np.float
        )

        return report

    class Model:
        '''An approximation of a dense matrix as a sum of RBF over distance
        matrices.
        '''

        def __init__(
            self, f, x, x_names=None, default_device=None,
            default_grad_on=False
        ):
            for w in x:
                assert w.shape[0] == x[0].shape[0],\
                    "Inconsisent component size."
            self.f = f
            self.x = x
            self.x_names = x_names
            self.default_device = default_device
            self.default_grad_on = default_grad_on

        def __repr__(self):
            xns = ', '.join(self.x_names)
            return f'<batch of {len(self)} RBF expansions [x_names = {xns}]>'

        def __len__(self):
            return len(self.x[0])

        def __call__(
            self, runs=None, components=None, device=None, grad_on=None
        ):
            if grad_on is None:
                grad_on = self.default_grad_on
            with torch.set_grad_enabled(grad_on):
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
                device = device or self.default_device
                return self.f(*x).to(device)

        @property
        def funrank(self):
            return len(self.x[-1])

        def __getattr__(self, a):
            return self.x[self.x_names.index(a)]
