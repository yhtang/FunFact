#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod


class SymbolSet(ABC):
    @abstractmethod
    def _lit(self):
        '''literal value'''

    @abstractmethod
    def _idn(self):
        '''indexed notation for a single tensor'''

    @abstractmethod
    def _call(self):
        '''nonlinear function call'''

    @abstractmethod
    def _pow(self):
        '''raise to power'''

    @abstractmethod
    def _neg(self):
        '''elementwise negation'''

    @abstractmethod
    def _einsum(self):
        '''Einstein summation'''

    @abstractmethod
    def _mul(self):
        '''elementwise multiplication, Hadamard product'''

    @abstractmethod
    def _div(self):
        '''elementwise division'''

    @abstractmethod
    def _add(self):
        '''elementwise addition'''

    @abstractmethod
    def _sub(self):
        '''elementwise subtraction'''


class SymbolPrecedence(SymbolSet):

    def __call__(self, symbol):
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
