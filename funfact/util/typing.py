#!/usr/bin/env python
# -*- coding: utf-8 -*-


def _is_tensor(x):
    return hasattr(x, "__len__")
