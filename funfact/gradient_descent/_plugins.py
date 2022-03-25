#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import dill
from ._base import GradientDescentState, GradientDescentPlugin


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


def snapshot(fac, to, every=1, at=None):

    return GradientDescentPlugin(
        name='snapshot',
        trigger=GradientDescentPlugin.on_step(every, at),
        action=lambda state: to.append(
            dict(
                step=state.step,
                model=dill.loads(dill.dumps(fac))
            )
        )
    )


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
