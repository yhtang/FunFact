#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from ._base import GradientDescentPlugin


def gradient_descent_plugin(every=None, at=None, trigger=None):
    '''A decorator that turns a method into a GD plugin'''
    if sum(map(lambda a: a is not None, [every, at, trigger])) != 1:
        raise RuntimeError(
            'One and only one of `every`, `at`, and `trigger` must be '
            'specified.'
        )

    if trigger is None:
        trigger = GradientDescentPlugin.on_step(every, at)

    def wrapper(f):

        return GradientDescentPlugin(
            name=f.__name__,
            trigger=trigger,
            action=f
        )

    return wrapper


def walltime(to, every=1, at=None):

    return GradientDescentPlugin(
        name='walltime',
        trigger=GradientDescentPlugin.on_step(every, at),
        action=lambda state: to.append(
            dict(
                step=state.step,
                time=time.perf_counter()
            )
        )
    )
