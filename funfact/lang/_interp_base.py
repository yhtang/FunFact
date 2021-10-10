#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
# from typing import NamedTuple, Any


# class Evaluated(NamedTuple):
#     expr: Any
#     value: Any


def no_op(f):
    '''A no-op rule does nothing.'''
    def do_nothing(*args, **kwargs):
        pass

    return do_nothing


class Interpreter(ABC):
    '''See definitions of the primitives in the `Primitives` class.'''

    @abstractmethod
    def scalar(self):
        pass

    @abstractmethod
    def tensor(self):
        pass

    @abstractmethod
    def index(self):
        pass

    @abstractmethod
    def index_notation(self):
        pass

    @abstractmethod
    def call(self):
        pass

    @abstractmethod
    def pow(self):
        pass

    @abstractmethod
    def neg(self):
        pass

    @abstractmethod
    def mul(self):
        pass

    @abstractmethod
    def div(self):
        pass

    @abstractmethod
    def add(self):
        pass

    @abstractmethod
    def sub(self):
        pass
