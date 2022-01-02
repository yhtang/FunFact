#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.backend import active_backend as ab
from contextlib import contextmanager


_grad_on = True


def is_grad_on():
    return _grad_on


@contextmanager
def enable_grad(mode: bool) -> None:
    global _grad_on
    prev_mode = _grad_on
    _grad_on = mode
    if 'torch' in repr(ab).lower():
        ab.set_grad_enabled(mode)
    try:
        yield
    finally:
        _grad_on = prev_mode
        if 'torch' in repr(ab).lower():
            ab.set_grad_enabled(prev_mode)
