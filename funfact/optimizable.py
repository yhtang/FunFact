#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.backend import active_backend as ab
from typing import Any


class Optimizable:

    def __init__(self, mode: bool) -> None:
        global _all_optimizable
        self._prev_mode = _all_optimizable
        _all_optimizable = mode
        if 'torch' in ab.__repr__().lower():
            ab.set_grad_enabled(mode)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _all_optimizable
        _all_optimizable = self._prev_mode
        if 'torch' in ab.__repr__().lower():
            ab.set_grad_enabled(self._prev_mode)


_all_optimizable = True


def is_optimizable():
    global _all_optimizable
    return _all_optimizable


def optimizable(mode: bool):
    return Optimizable(mode)
