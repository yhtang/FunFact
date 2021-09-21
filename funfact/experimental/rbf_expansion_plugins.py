#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dill
import time


def _make_trigger(every, at):
    if at is not None:
        if callable(at):
            return at
        elif hasattr(at, '__contains__'):
            return lambda step: step in at
        else:
            raise RuntimeError(f'Invalid argument for `at`: {at}.')
    else:
        return lambda step: step % every == 0


def snapshot(to, every=1, at=None):

    trigger = _make_trigger(every, at)

    def callback(step, model):
        if trigger(step):
            to.append(
                dict(
                    step=step,
                    model=dill.loads(dill.dumps(model))
                )
            )

    return dict(
        every=every,
        callback=callback
    )


def walltime(to, every=1, at=None):

    trigger = _make_trigger(every, at)

    def callback(step):
        if trigger(step):
            to.append(
                dict(
                    step=step,
                    time=time.perf_counter()
                )
            )

    return dict(
        every=every,
        callback=callback
    )
