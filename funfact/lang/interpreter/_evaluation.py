#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ._base import ROOFInterpreter
from ._einop import _einop


def _map_index(index_map, idx):
    try:
        return index_map[idx]
    except KeyError:
        index_map[idx] = chr(97 + len(index_map))
        return index_map[idx]


class Evaluator(ROOFInterpreter):
    '''The evaluation interpreter evaluates an initialized tensor expression.
    '''

    def scalar(self, value, **kwargs):
        return value

    def tensor(self, value, data=None, **kwargs):
        return data

    def index(self, value, payload):
        return value.symbol

    def index_notation(self, tensor, indices, payload):
        return (tensor, indices)

    def call(self, f, x, payload):
        return (getattr(np, f)(x[0]), payload[1])

    def pow(self, base, exponent, payload):
        return (np.power(base[0], exponent), payload[1])

    def neg(self, x, live_indices=None, **kwargs):
        return (-x[0], live_indices)

    @staticmethod
    def _binary_operator(op, lhs, rhs, specs):
        survive_indices, lhs_indices, rhs_indices = specs
        index_map = {}
        lhs_spec = ''.join([_map_index(index_map, idx) for idx in lhs_indices])
        rhs_spec = ''.join([_map_index(index_map, idx) for idx in rhs_indices])
        return (
            _einop(f'{lhs_spec},{rhs_spec}', lhs, rhs, op),
            survive_indices
        )

    def div(self, lhs, rhs, payload):
        return self._binary_operator(
            np.divide, lhs[0], rhs[0], (payload[1], lhs[1], rhs[1])
        )

    def mul(self, lhs, rhs, payload):
        return self._binary_operator(
            np.multiply, lhs[0], rhs[0], (payload[1], lhs[1], rhs[1])
        )

    def add(self, lhs, rhs, payload):
        return self._binary_operator(
            np.add, lhs[0], rhs[0], (payload[1], lhs[1], rhs[1])
        )

    def sub(self, lhs, rhs, payload):
        return self._binary_operator(
            np.subtract, lhs[0], rhs[0], (payload[1], lhs[1], rhs[1])
        )
