#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Callable
from dataclasses import dataclass
from funfact.backend import active_backend as ab


@dataclass
class GradientDescentState:
    step: int
    loss: ab.tensor_t
    grad: ab.tensor_t


@dataclass
class GradientDescentPlugin:
    name: str
    trigger: Callable
    action: Callable

    @staticmethod
    def on_step(every, at):
        if at is not None:
            if callable(at):
                return at
            elif hasattr(at, '__contains__'):
                return lambda state: state.step in at
            else:
                raise RuntimeError(f'Invalid argument for `at`: {at}.')
        else:
            return lambda state: state.step % every == 0
