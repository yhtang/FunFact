#!/usr/bin/env python
# -*- coding: utf-8 -*-
import contextlib


context = {}


@contextlib.contextmanager
def set_context(**kwargs):
    global context
    try:
        context = dict(**kwargs)
        yield
    finally:
        context.clear()
