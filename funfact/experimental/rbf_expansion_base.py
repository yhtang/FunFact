#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import dill
import torch


def as_namedtuple(name, **kwargs):
    return namedtuple(name, list(kwargs.keys()))(*kwargs.values())


class RBFExpansionBase:

    @staticmethod
    def _get_device(desc):
        if desc == 'auto':
            try:
                return torch.device('cuda')
            except Exception:
                return torch.device('cpu')
        else:
            return torch.device(desc)

    def to_pickle(self, file):
        open(file, 'wb').write(dill.dumps(self.__dict__))

    @classmethod
    def from_pickle(cls, file):
        fac = cls(k=0)
        for key, val in dill.loads(open(file, 'rb').read()).items():
            setattr(fac, key, val)
        return fac

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

    def rand(self, *shape, requires_grad=True):
        return torch.rand(
            shape, requires_grad=requires_grad, device=self.device
        )

    def as_tensor(self, tsr, requires_grad=True):
        return torch.as_tensor(tsr, device=self.device)\
            .detach()\
            .requires_grad_(requires_grad)

    @property
    def report(self):
        return self._report

    @report.setter
    def report(self, report_dict):
        self._report = as_namedtuple('report', **report_dict)

    @property
    def optimum(self):
        return self._optimum

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

        def __getstate__(self):
            return vars(self)

        def __setstate__(self, state):
            vars(self).update(state)

    class ModelPlus:
        '''An approximation of a dense matrix as a sum of RBF over distance
        matrices.
        '''

        def __init__(
            self, rbf, f, x, x_names=None, default_device=None,
            default_grad_on=False
        ):
            for w in x:
                assert w.shape[0] == x[0].shape[0],\
                    "Inconsisent component size."
            self.rbf = rbf
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
                return self.f(self.rbf, *x).to(device)

        @property
        def funrank(self):
            return len(self.x[-1])

        def __getattr__(self, a):
            return self.x[self.x_names.index(a)]

        def __getstate__(self):
            return vars(self)

        def __setstate__(self, state):
            vars(self).update(state)
