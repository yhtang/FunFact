#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import tqdm
import warnings
from .rbf_expansion_base import RBFExpansionBase


class RBFExpansionPlus(RBFExpansionBase):

    def __init__(
        self, k=1, rbf='gauss', batch_size=64, max_steps=10000,
        loss='mse_loss', algorithm='Adam', lr=0.1, device='auto',
        amp=False, progressbar='default'
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
        self.amp = amp

        if progressbar == 'default':
            self.progressbar = lambda n: tqdm.trange(
                n, miniters=None, mininterval=0.25, leave=True
            )
        else:
            self.progressbar = progressbar

    def fit(
        self, target, seed=None, plugins=[], u0=None, v0=None, a0=None, b0=None
    ):

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

            def f(rbf, u, v, a, b):
                return torch.sum(
                    rbf(u[..., :, None, :] - v[..., None, :, :]) *
                    a[..., None, None, :],
                    dim=-1
                ) + b[..., None, None]

            self.report = self._grad_opt(
                target,
                self.ModelPlus(
                    self.rbf, f, (u0, v0, a0, b0),
                    x_names=['u', 'v', 'a', 'b'],
                    default_grad_on=True
                ),
                plugins
            )

            self._optimum = self.ModelPlus(
                self.rbf, f, self.report.x_best, x_names=['u', 'v', 'a', 'b'],
                default_device='cpu'
            )

            return self

    def fith(self, target, seed=None, plugins=[], u0=None, a0=None, b0=None):

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

            def f(rbf, u, a, b):
                return torch.sum(
                    rbf(u[..., :, None, :] - u[..., None, :, :]) *
                    a[..., None, None, :],
                    dim=-1
                ) + b[..., None, None]

            self.report = self._grad_opt(
                target,
                self.ModelPlus(
                    self.rbf, f, (u0, a0, b0), x_names=['u', 'a', 'b'],
                    default_grad_on=True
                ),
                plugins
            )

            self._optimum = self.ModelPlus(
                self.rbf, f, self.report.x_best, x_names=['u', 'a', 'b'],
                default_device='cpu'
            )

            return self

    def fit_custom(self, target, f, seed=None, plugins=[], **x0):

        with torch.random.fork_rng(devices=[self.device]):
            if seed:
                torch.random.manual_seed(seed)

            target = torch.as_tensor(target, device=self.device).unsqueeze(0)
            _, n, m = target.shape

            x = [self.as_tensor(w) for w in x0.values()]

            self.report = self._grad_opt(
                target,
                self.ModelPlus(
                    self.rbf, f, x, x_names=list(x0.keys()),
                    default_grad_on=True
                ),
                plugins
            )

            self._optimum = self.ModelPlus(
                self.rbf, f, self.report.x_best, x_names=list(x0.keys()),
                default_device='cpu'
            )

            return self

    def _grad_opt(self, target, model, plugins=[]):

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
        report['loss_history_ticks'] = []
        for step in self.progressbar(self.max_steps):
            opt.zero_grad()

            with torch.cuda.amp.autocast(
                enabled=self.amp and target.device.type == 'cuda'
            ):
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
                report['loss_history_ticks'].append(step)
                report['t_best'][better] = step
                for current, new in zip(report['x_best'], model.x):
                    current[better, ...] = new[better, ...]

                for plugin in plugins:
                    if step % plugin['every'] == 0:
                        local_vars = locals()

                        try:
                            requires = plugin['requires']
                        except KeyError:
                            requires = plugin['callback'].__code__.co_varnames

                        args = {k: local_vars[k] for k in requires}

                        plugin['callback'](**args)

            loss_batch.sum().backward()
            opt.step()

        report['loss_history'] = np.array(
            report['loss_history'], dtype=np.float
        )

        return report
