#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import NamedTuple


class Primitive(NamedTuple):
    name: str
    precedence: int
    terminal: bool


class Primitives:
    def primitive(precedence, terminal):
        def as_property(f):
            p = Primitive(f.__name__, precedence, terminal)
            return property(lambda self: p)
        return as_property

    @primitive(precedence=0, terminal=True)
    def scalar(self):
        '''a scalar number'''

    @primitive(precedence=0, terminal=True)
    def tensor(self):
        '''a tensor'''

    @primitive(precedence=0, terminal=True)
    def index(self):
        '''an index/subscript'''

    @primitive(precedence=1, terminal=False)
    def index_notation(self):
        '''indexed notation for a single tensor'''

    @primitive(precedence=2, terminal=False)
    def call(self):
        '''nonlinear function call'''

    @primitive(precedence=3, terminal=False)
    def pow(self):
        '''raise to power'''

    @primitive(precedence=4, terminal=False)
    def neg(self):
        '''elementwise negation'''

    @primitive(precedence=5, terminal=False)
    def mul(self):
        '''elementwise multiplication, Hadamard product'''

    @primitive(precedence=5, terminal=False)
    def div(self):
        '''elementwise division'''

    @primitive(precedence=6, terminal=False)
    def add(self):
        '''elementwise addition'''

    @primitive(precedence=6, terminal=False)
    def sub(self):
        '''elementwise subtraction'''


primitives = Primitives()
