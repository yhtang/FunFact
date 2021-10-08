#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Operators:
    def undefined(f):
        '''An undefined method is similar to an abstract method except for that
        it only produces an error when being called, rather than prohibiting
        the instantiation of objects.'''
        def undefined_handler(*args, **kwargs):
            raise RuntimeError(f'Method {f.__name__} is undefined.')

        return undefined_handler

    @undefined
    def _lit(self):
        '''literal value'''

    @undefined
    def _idn(self):
        '''indexed notation for a single tensor'''

    @undefined
    def _call(self):
        '''nonlinear function call'''

    @undefined
    def _pow(self):
        '''raise to power'''

    @undefined
    def _neg(self):
        '''elementwise negation'''

    @undefined
    def _einsum(self):
        '''Einstein summation'''

    @undefined
    def _mul(self):
        '''elementwise multiplication, Hadamard product'''

    @undefined
    def _div(self):
        '''elementwise division'''

    @undefined
    def _add(self):
        '''elementwise addition'''

    @undefined
    def _sub(self):
        '''elementwise subtraction'''


class OperatorPrecedence(Operators):

    def __getitem__(self, symbol):
        return getattr(self, f'_{symbol}')()

    def _lit(self):
        return 0

    def _idn(self):
        return 1

    def _call(self):
        return 2

    def _pow(self):
        return 3

    def _neg(self):
        return 4

    def _einsum(self):
        return 5

    def _mul(self):
        return 5

    def _div(self):
        return 5

    def _add(self):
        return 6

    def _sub(self):
        return 6
