#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._expr import Expr


def exp(input):
    return Expr('call', input, f='exp')


def sin(input):
    return Expr('call', input, f='sin')


def cos(input):
    return Expr('call', input, f='cos')


def square(input):
    return Expr('square', input)
