#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import importlib
from ._context import set_context


available_backends = ['jax', 'torch', 'numpy']
'''The names of the available backends.

Examples:
    >>> from funfact import available_backends
    >>> available_backends
    ['jax', 'torch', 'numpy']
'''


_active_backend = None


def _get_active_backend():
    if _active_backend is None:
        _use_default_backend()
    return _active_backend


def use(backend: str, **context):
    '''Specify the numerical tensor algebra library to use as the computational
    backend.

    Args:
        backend (str):
            Currently implemented backends are:

            - `'numpy'`: the [NumPy](https://numpy.org/) backend only supports
            forward calculations but no automatic differentiation.
            - `'jax'`: [JAX](../JAXBackend).
            - `'torch'`: [PyTorch](https://pytorch.org/).

            Dynamic switching betwewen backends is allowed. However, tensors
            created by the previous backend will not be automatically ported to
            the new backend.

        context (kwargs): Backend-specific additional arguments.
            For details, refer to the individual backends, e.g. [JAX]

        context (kwargs):
            Additional context to be passed on to the specified backend.

    Examples:
        >>> from funfact import use, active_backend as ab
        >>> use('numpy')
        >>> ab
        <backend 'NumpyBackend'>

        >>> use('jax')
        >>> ab
        <backend 'JAXBackend'>

        >>> use('torch')
        >>> ab
        <backend 'PyTorchBackend'>

        >>> use('torch', device='cuda:0')
        >>> ab
        <backend 'PyTorchBackend'>
    '''
    global _active_backend
    try:
        if backend == 'jax':
            enable_x64 = context.pop('enable_x64', False)
            os.environ['JAX_ENABLE_X64'] = 'True' if enable_x64 else 'False'
        with set_context(**context):
            _active_backend = importlib.import_module(
                f'funfact.backend._{backend}'
            )
    except KeyError:
        raise RuntimeError(f'Backend {backend} cannot be imported.')


def _use_default_backend():
    for backend in available_backends:
        try:
            use(backend)
            sys.stderr.write(
                f'Using backend "{_get_active_backend().__name__}".'
            )
            sys.stderr.flush()
            return
        except Exception:
            continue
    raise RuntimeError(
        'None of the backends {abs} appears usable.'.format(
            abs=tuple(available_backends.keys())
        )
    )


class ActiveBackendProxy:

    def __repr__(self):
        return f"<backend '{_get_active_backend().__name__}'>"

    def __getattr__(self, attr):
        ab = _get_active_backend()
        try:
            return getattr(ab, attr)
        except AttributeError:
            try:
                return getattr(ab.nla, attr)
            except AttributeError:
                raise AttributeError(
                    f'Backend {ab.__name__} does not implement {attr}.'
                )

    def is_native(self, array):
        '''Determine if the argument is of type native_t.'''
        return isinstance(array, self.native_t)

    def is_tensor(self, array):
        '''Determine if the argument is one of tensor_t.'''
        return isinstance(array, self.tensor_t)

    def log_add_exp(self, lhs, rhs):
        return self.log(self.add(self.exp(lhs), self.exp(rhs)))

    def log_sum_exp(self, data, axis=None):
        return self.log(self.sum(self.exp(data), axis=axis))


active_backend = ActiveBackendProxy()
'''
`active_backend` is a proxy object that can be used as if it is the
 underlying numerical backend.

Examples:
    >>> from funfact import use, active_backend as ab
    >>> use('jax')
    >>> ab.tensor([[1.0, 2.0], [2.0, 1.0]])
    DeviceArray([[1., 2.],
                [2., 1.]], dtype=float32)

    >>> ab.sin(3.1415926)
    DeviceArray(1.509958e-07, dtype=float32, weak_type=True)

    >>> ab.linspace(0, 1, 3)
    DeviceArray([0. , 0.5, 1. ], dtype=float32)

    >>> use('torch')
    >>> ab.eye(3, dtype=ab.float16)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]], dtype=torch.float16)
'''
