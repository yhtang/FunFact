#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.backend import active_backend as ab
from contextlib import contextmanager


_all_optimizable = True


def is_optimizable():
    global _all_optimizable
    return _all_optimizable


@contextmanager
def set_optimizable(mode: bool) -> None:
    global _all_optimizable
    _prev_mode = _all_optimizable
    _all_optimizable = mode
    if 'torch' in ab.__repr__().lower():
        ab.set_grad_enabled(mode)
    try:
        yield None
    finally:
        _all_optimizable = _prev_mode
        if 'torch' in ab.__repr__().lower():
            ab.set_grad_enabled(_prev_mode)
