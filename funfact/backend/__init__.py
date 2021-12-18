#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
import warnings


__all__ = ['active_backend', 'available_backends', 'use']

_active_backend = None

available_backends = {
    'jax': 'JAXBackend',
    'torch': 'PyTorchBackend',
    'numpy': 'NumpyBackend',
}
'''A dictionary whose keys are the names of the available backends.

Examples:
    >>> from funfact import available_backends
    >>> available_backends.keys()
    dict_keys(['numpy', 'jax', 'torch'])
'''


def use(backend: str):
    '''Specify the numerical tensor algebra library to use as the computational
    backend.

    Args:
        backend (str):
            Currently implemented backends are:

            - `'numpy'`: the [NumPy](https://numpy.org/) backend only supports
            forward calculations but no automatic differentiation.
            - `'jax'`: [JAX](https://jax.readthedocs.io/en/latest/index.html).
            - `'torch'`: [PyTorch](https://pytorch.org/).

            Dynamic switching betwewen backends is allowed. However, tensors
            created by the previous backend will not be automatically ported to
            the new backend.

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
    '''
    global _active_backend
    try:
        clsname = available_backends[backend]
        _active_backend = getattr(
            importlib.import_module(f'funfact.backend._{backend}'), clsname
        )
    except KeyError:
        raise RuntimeError(f'Unknown backend {backend}.')


class Backend:

    def __init__(self, backend=None):
        if backend is None:
            for backend in available_backends.keys():
                try:
                    use(backend)
                    warnings.warn(f'Default backend: {self}')
                    return
                except Exception:
                    continue
            raise RuntimeError(
                'None of the built-in backends {abs} appears usable.'.format(
                    abs=tuple(available_backends.keys())
                )
            )
        else:
            use(backend)

    def __repr__(self):
        return f"<backend '{_active_backend.__name__}'>"

    def __getattr__(self, attr):
        return getattr(_active_backend, attr)


active_backend = Backend()
'''
This is a proxy object that always points to the actual numerical tensor
backend that is currently active.

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
