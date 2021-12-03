#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import dill
import numpy as np
import pycuda.driver as cuda
from funfact.cuda import context_manager, ManagedArray


class RBFExpansionBasePyCUDA:

    def __init__(self):
        context_manager.autoinit()

    @staticmethod
    def as_namedtuple(name, **kwargs):
        return namedtuple(name, list(kwargs.keys()))(*kwargs.values())

    @staticmethod
    def _as_cuda_array(arr, dtype=None, order=None):
        if (isinstance(arr, np.ndarray) and
                isinstance(arr.base, cuda.ManagedAllocation) and
                arr.dtype == dtype and
                ((order is None) or
                 (order == 'C' and arr.flags.c_contiguous) or
                 (order == 'F' and arr.flags.f_contiguous))):
            return arr
        else:
            return ManagedArray.copy(arr, dtype, order)

    @staticmethod
    def _zero_cuda_array(arr):
        assert isinstance(arr.base, cuda.ManagedAllocation)
        cuda.memset_d32(
            arr.base.get_device_pointer(),
            0,
            arr.dtype.itemsize // 4 * np.prod(arr.shape).item()
        )

    def to_pickle(self, file):
        state = self.__dict__.copy()
        open(file, 'wb').write(dill.dumps(state))

    @classmethod
    def from_pickle(cls, file):
        fac = cls()
        for key, val in dill.loads(open(file, 'rb').read()).items():
            setattr(fac, key, val)
        return fac

    @property
    def config(self):
        return {
            key: self.__dict__[key] for key in self.__dict__
            if not key.startswith('_')
        }

    @property
    def report(self):
        return self._report

    @report.setter
    def report(self, report_dict):
        self._report = self.as_namedtuple('report', **report_dict)

    @property
    def optimum(self):
        return self._optimum

    class Model:
        '''An approximation of a dense matrix as a sum of RBF over distance
        matrices.
        '''

        def __init__(
            self, f, x, x_names=None
        ):
            for w in x:
                assert w.shape[-1] == x[0].shape[-1],\
                    "Inconsisent component size."
            self.f = f
            self.x = x
            self.x_names = x_names

        def __repr__(self):
            xns = ', '.join(self.x_names)
            return f'<ensemble of {len(self)} RBF expansions [x_names = {xns}]>'

        def __len__(self):
            return len(self.x[-1])

        def __call__(
            self, runs=None, components=None, device=None
        ):
            x = self.x
            if components is not None:
                components = np.array(components)
                if components.ndim == 0:
                    components = np.expand_dims(components, 0)
                x = [w[..., components, :] if w.ndim >= 2 else w for w in x]
            if runs is not None:
                x = [w[..., runs] for w in x]
            return self.f(*x)

        @property
        def funrank(self):
            return len(self.x[-2])

        def __getattr__(self, a):
            return self.x[self.x_names.index(a)]

        def __getstate__(self):
            return vars(self)

        def __setstate__(self, state):
            vars(self).update(state)
