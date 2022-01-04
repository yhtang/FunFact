#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact import active_backend as ab


def zeros(shape, dtype=None, optimizable=True):
    return ab.set_optimizable(
        ab.zeros(shape, dtype=dtype or ab.float32),
        optimizable
    )


def ones(shape, dtype=None, optimizable=True):
    return ab.set_optimizable(
        ab.ones(shape, dtype=dtype or ab.float32),
        optimizable
    )


def normal(shape, dtype=None, optimizable=True):
    return ab.set_optimizable(
        ab.normal(0.0, 1.0, shape, dtype=dtype or ab.float32),
        optimizable
    )


def decanormal(shape, dtype=None, optimizable=True):
    return ab.set_optimizable(
        ab.normal(0.0, 0.1, shape, dtype=dtype or ab.float32),
        optimizable
    )
