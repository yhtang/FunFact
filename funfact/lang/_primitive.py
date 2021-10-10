#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import NamedTuple


class Primitive(NamedTuple):
    name: str
    precedence: int


class Primitives:
    def primitive(precedence):
        def as_property(f):
            p = Primitive(f.__name__, precedence)
            return property(lambda self: p)
        return as_property

    @primitive(precedence=0)
    def scalar(self):
        '''a scalar number'''

    @primitive(precedence=0)
    def tensor(self):
        '''a tensor'''

    @primitive(precedence=0)
    def index(self):
        '''an index/subscript'''

    @primitive(precedence=1)
    def index_notation(self):
        '''indexed notation for a single tensor'''

    @primitive(precedence=2)
    def call(self):
        '''nonlinear function call'''

    @primitive(precedence=3)
    def pow(self):
        '''raise to power'''

    @primitive(precedence=4)
    def neg(self):
        '''elementwise negation'''

    @primitive(precedence=5)
    def mul(self):
        '''elementwise multiplication, Hadamard product'''

    @primitive(precedence=5)
    def div(self):
        '''elementwise division'''

    @primitive(precedence=6)
    def add(self):
        '''elementwise addition'''

    @primitive(precedence=6)
    def sub(self):
        '''elementwise subtraction'''


primitives = Primitives()
