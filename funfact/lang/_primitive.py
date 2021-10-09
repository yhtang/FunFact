#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import NamedTuple


class Primitive(NamedTuple):
    name: str
    precedence: int


class _PrimitivesMeta(type):
    def primitive(precedence):
        def wrapper(f):
            p = Primitive(f.__name__, precedence)
            return property(lambda self: p)

        return wrapper

    @primitive(precedence=0)
    def lit(self):
        '''literal value'''

    @primitive(precedence=1)
    def _idn(self):
        '''indexed notation for a single tensor'''

    @primitive(precedence=2)
    def _call(self):
        '''nonlinear function call'''

    @primitive(precedence=3)
    def _pow(self):
        '''raise to power'''

    @primitive(precedence=4)
    def _neg(self):
        '''elementwise negation'''

    @primitive(precedence=5)
    def _mul(self):
        '''elementwise multiplication, Hadamard product'''

    @primitive(precedence=5)
    def _div(self):
        '''elementwise division'''

    @primitive(precedence=6)
    def _add(self):
        '''elementwise addition'''

    @primitive(precedence=6)
    def _sub(self):
        '''elementwise subtraction'''


class primitives(metaclass=_PrimitivesMeta):
    pass
