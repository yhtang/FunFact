#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib

__all__ = ['active_backend', 'available_backends', 'use']

_active_backend = None
_default_backend = 'jax'
available_backends = {
    'numpy': 'NumpyBackend',
    'jax': 'JAXBackend',
    'torch': 'PyTorchBackend',
}


def use(backend: str):
    global _active_backend
    try:
        clsname = available_backends[backend]
        _active_backend = getattr(
            importlib.import_module(f'funfact.backend._{backend}'), clsname
        )
    except KeyError:
        raise RuntimeError(f'Unknown backend {backend}.')


class Backend:

    def __init__(self, backend=_default_backend):
        use(backend)

    def __repr__(self):
        return f"<backend '{_active_backend.__name__}'>"

    def __getattr__(self, attr):
        return getattr(_active_backend, attr)


active_backend = Backend()
