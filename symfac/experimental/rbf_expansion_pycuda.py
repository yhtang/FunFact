#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import warnings

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import tqdm

import symfac.cpp
from symfac.cpp import Template
from symfac.cuda import ManagedArray
import symfac.optim as optim

from .rbf_expansion_base_pycuda import RBFExpansionBasePyCUDA


class RBFExpansion(RBFExpansionBasePyCUDA):

    def __init__(
        self, r=1,
        # rbf='gauss',  # TODO: support custom exprs using sympy
        ensemble_size=64, max_steps=10000,
        # loss='mse_loss',  # TODO: support custom exprs using sympy
        algorithm='Adam', lr=0.1,
        progressbar='default',
        cuda_device='auto',
        cuda_thread_per_block=64,
        cuda_block_per_inst=2,
        cuda_tile_size=(8, 8, 8),
    ):
        self.r = r

        # if callable(rbf):
        #     self.rbf = rbf
        # elif rbf == 'gauss':
        #     self.rbf = lambda d: torch.exp(-torch.square(d))
        # else:
        #     raise f'Unrecoginized RBF: {rbf}.'

        self.ensemble_size = ensemble_size
        self.max_steps = max_steps

        # if isinstance(loss, str):
        #     try:
        #         self.loss = getattr(torch.nn.functional, loss)
        #     except AttributeError:
        #         raise AttributeError(
        #             f'The loss function \'{loss}\' does not exist in'
        #             'torch.nn.functional.'
        #         )
        # else:
        #     self.loss = loss
        # try:
        #     self.loss(torch.zeros(1), torch.zeros(1))
        # except Exception as e:
        #     raise AssertionError(
        #         f'The given loss function does not accept two arguments:\n{e}'
        #     )

        if isinstance(algorithm, str):
            try:
                self.algorithm = getattr(optim, algorithm)
            except AttributeError:
                raise AttributeError(
                    f'Cannot find \'{algorithm}\' in torch.optim.'
                )
        else:
            self.algorithm = algorithm

        self.lr = lr
        self.cuda_context = self._get_cuda_context(cuda_device)
        self.cuda_block_per_inst = cuda_block_per_inst
        self.cuda_thread_per_block = cuda_thread_per_block
        self.cuda_tile_size = cuda_tile_size

        if progressbar == 'default':
            self.progressbar = lambda n: tqdm.trange(
                n, miniters=None, mininterval=0.25, leave=True
            )
        else:
            self.progressbar = progressbar

    def __del__(self):
        self.cuda_context.synchronize()
        self.cuda_context.pop()

    @property
    def src(self):
        try:
            return self._src
        except AttributeError:
            self._src = Template(
                open(
                    symfac.cpp.__path__[0] + '/rbf_expansion_ensemble.cu'
                ).read()
            )
        return self._src

    def fit(
        self, target, seed=None, plugins=[], u0=None, v0=None, a0=None, b0=None
    ):
        rng = np.random.default_rng(seed)

        A = self._as_cuda_array(target, dtype=np.float32, order='F')
        E = self.ensemble_size
        R = self.r
        N, M = A.shape

        u0 = rng.normal(0.0, 0.1, (N, R, E)) if u0 is None else u0
        v0 = rng.normal(0.0, 0.1, (M, R, E)) if v0 is None else v0
        a0 = rng.normal(0.0, np.std(target) / np.sqrt(R), (R, E)) if a0 is None else a0
        b0 = rng.normal(0.0, 1.0, (E,)) if b0 is None else b0

        u = self._as_cuda_array(u0, dtype=np.float32, order='F')
        v = self._as_cuda_array(v0, dtype=np.float32, order='F')
        a = self._as_cuda_array(a0, dtype=np.float32, order='F')
        b = self._as_cuda_array(b0, dtype=np.float32, order='F')

        L = ManagedArray.zeros((E,), dtype=np.float32, order='F')
        du = ManagedArray.zeros((N, R, E), dtype=np.float32, order='F')
        dv = ManagedArray.zeros((M, R, E), dtype=np.float32, order='F')
        da = ManagedArray.zeros((R, E), dtype=np.float32, order='F')
        db = ManagedArray.zeros((E,), dtype=np.float32, order='F')

        kernel = SourceModule(
            self.src.render(
                E=E, N=N, M=M, R=R,
                n=self.cuda_tile_size[0],
                m=self.cuda_tile_size[1],
                r=self.cuda_tile_size[2],
                thread_per_block=self.cuda_thread_per_block,
                block_per_inst=self.cuda_block_per_inst
            ),
            options=['-std=c++14',
                     '-O4',
                     '--use_fast_math',
                     '--expt-relaxed-constexpr',
                     '--maxrregcount=64',
                     '-Xptxas', '-v',
                     '-lineinfo',
                     ],
            no_extern_c=True,
            keep=True
        ).get_function('rbf_expansion_ensemble')

        def f_cuda(x):
            u, v, a, b = x
            u = self._as_cuda_array(u, dtype=np.float32, order='F')
            v = self._as_cuda_array(v, dtype=np.float32, order='F')
            a = self._as_cuda_array(a, dtype=np.float32, order='F')
            b = self._as_cuda_array(b, dtype=np.float32, order='F')
            self._zero_cuda_array(L)
            self._zero_cuda_array(du)
            self._zero_cuda_array(dv)
            self._zero_cuda_array(da)
            self._zero_cuda_array(db)

            kernel(
                A, u, v, a, b, L, du, dv, da, db,
                block=(self.cuda_thread_per_block, 1, 1),
                grid=(self.cuda_block_per_inst, self.ensemble_size)
            )

            self.cuda_context.synchronize()

            return np.copy(L), (du, dv, da, db)

        def f_cpu(
            # rbf,
            u, v, a, b
        ):
            return np.sum(
                np.exp(
                    -np.square(
                        u[:, None, ...] - v[None, :, ...]
                    )
                ) * a[None, None, ...],
                axis=2
            ) + b[None, None, ...]

        self.report = self._grad_opt(
            f_cuda, (u, v, a, b), plugins
        )

        self._optimum = self.Model(
            # self.rbf,
            f_cpu, self.report.x_best, x_names=['u', 'v', 'a', 'b']
        )

        return self

    def fith(
        self, target, seed=None, plugins=[], u0=None, a0=None, b0=None
    ):
        rng = np.random.default_rng(seed)

        A = self._as_cuda_array(target, dtype=np.float32, order='F')
        E = self.ensemble_size
        R = self.r
        N, M = A.shape
        assert(
            N == M and np.allclose(A, A.T),
            'fith() only works for symmetric matrices'
        )

        u0 = rng.normal(0.0, 0.1, (N, R, E)) if u0 is None else u0
        a0 = rng.normal(0.0, np.std(target) / np.sqrt(R), (R, E)) if a0 is None else a0
        b0 = rng.normal(0.0, 1.0, (E,)) if b0 is None else b0

        u = self._as_cuda_array(u0, dtype=np.float32, order='F')
        a = self._as_cuda_array(a0, dtype=np.float32, order='F')
        b = self._as_cuda_array(b0, dtype=np.float32, order='F')

        L = ManagedArray.zeros((E,), dtype=np.float32, order='F')
        du = ManagedArray.zeros((N, R, E), dtype=np.float32, order='F')
        da = ManagedArray.zeros((R, E), dtype=np.float32, order='F')
        db = ManagedArray.zeros((E,), dtype=np.float32, order='F')

        kernel = SourceModule(
            self.src.render(
                E=E, N=N, M=M, R=R,
                n=self.cuda_tile_size[0],
                m=self.cuda_tile_size[1],
                r=self.cuda_tile_size[2],
                thread_per_block=self.cuda_thread_per_block,
                block_per_inst=self.cuda_block_per_inst
            ),
            options=['-std=c++14',
                     '-O4',
                     '--use_fast_math',
                     '--expt-relaxed-constexpr',
                     '--maxrregcount=64',
                     '-Xptxas', '-v',
                     '-lineinfo',
                     ],
            no_extern_c=True,
            keep=True
        ).get_function('rbf_expansion_ensemble')

        def f_cuda(x):
            u, a, b = x
            u = self._as_cuda_array(u, dtype=np.float32, order='F')
            a = self._as_cuda_array(a, dtype=np.float32, order='F')
            b = self._as_cuda_array(b, dtype=np.float32, order='F')
            self._zero_cuda_array(L)
            self._zero_cuda_array(du)
            self._zero_cuda_array(da)
            self._zero_cuda_array(db)

            kernel(
                A, u, u, a, b, L, du, du, da, db,
                block=(self.cuda_thread_per_block, 1, 1),
                grid=(self.cuda_block_per_inst, self.ensemble_size)
            )

            self.cuda_context.synchronize()

            return np.copy(L), (du, da, db)

        def f_cpu(
            # rbf,
            u, a, b
        ):
            return np.sum(
                np.exp(
                    -np.square(
                        u[:, None, ...] - u[None, :, ...]
                    )
                ) * a[None, None, ...],
                axis=2
            ) + b[None, None, ...]

        self.report = self._grad_opt(
            f_cuda, (u, a, b), plugins
        )

        self._optimum = self.Model(
            # self.rbf,
            f_cpu, self.report.x_best, x_names=['u', 'a', 'b']
        )

        return self

    def _grad_opt(self, f, x, plugins=[]):

        try:
            opt = self.algorithm(x, self.lr)
        except Exception:
            raise AssertionError(
                'Cannot instance optimizer of type {self.algorithm}:\n{e}'
            )

        report = {}
        report['x_best'] = [np.copy(w) for w in x]
        report['t_best'] = np.zeros(self.ensemble_size, dtype=np.int64)
        report['loss_history'] = []
        report['loss_history_ticks'] = []

        for step in self.progressbar(self.max_steps):
            loss, grad = f(x)

            report['loss_history_ticks'].append(step)
            report['loss_history'].append(loss)

            if 'loss_best' in report:
                report['loss_best'] = np.minimum(
                    report['loss_best'], loss
                )
            else:
                report['loss_best'] = loss

            better = np.flatnonzero(report['loss_best'] == loss)
            report['t_best'][better] = step
            for current, new in zip(report['x_best'], x):
                current[..., better] = new[..., better]

            for plugin in plugins:
                if step % plugin['every'] == 0:
                    local_vars = locals()

                    try:
                        requires = plugin['requires']
                    except KeyError:
                        requires = plugin['callback'].__code__.co_varnames

                    args = {k: local_vars[k] for k in requires}

                    plugin['callback'](**args)

            opt.step(grad)

        report['loss_history'] = np.array(
            report['loss_history'], dtype=np.float
        )

        return report