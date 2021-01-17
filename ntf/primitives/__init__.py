#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import torch
from deap import gp


def instantiate(prefix_expr, pset, random_state=None):
    s = prefix_expr[0]
    node_cls = pset.context[s.name]
    args = []
    prefix_expr = prefix_expr[1:]
    for i in range(s.arity):
        t, prefix_expr = instantiate(prefix_expr, pset)
        args.append(t)
    node = node_cls(*args)
    return node, prefix_expr


class Types:

    class Matrix:
        pass

    class RowVector:
        pass

    class ColVector:
        pass


class Primitive(ABC):
    pass


class Terminal(Primitive):
    pass


class MatrixAdd(Primitive):

    in_types = [Types.Matrix, Types.Matrix]
    ret_type = Types.Matrix

    @classmethod
    def add_to_pset(cls, pset: gp.PrimitiveSetTyped):
        pset.addPrimitive(cls, cls.in_types, cls.ret_type)

    # def __init__(self, operand1, operand2):
    #     self.op

    # @classmethod
    # def instantiate(cls, random_state=None):
    #     return cls()


class RandomRowVector(Terminal):

    ret_type = Types.RowVector

    @classmethod
    def add_to_pset(cls, pset: gp.PrimitiveSetTyped):
        pset.addTerminal(cls, cls.ret_type)


class RandomColVector(Terminal):

    ret_type = Types.ColVector

    @classmethod
    def add_to_pset(cls, pset: gp.PrimitiveSetTyped):
        pset.addTerminal(cls, cls.ret_type)

    


class OuterProduct(Primitive):

    in_types = [Types.ColVector, Types.RowVector]
    ret_type = Types.Matrix

    @classmethod
    def add_to_pset(cls, pset: gp.PrimitiveSetTyped):
        pset.addPrimitive(cls, cls.in_types, cls.ret_type)

    def __init__(self, op1, op2):
        self.op1 = op1
        self.op2 = op2
