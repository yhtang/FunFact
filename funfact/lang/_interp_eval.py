#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ._interp_base import ROOFInterpreter
from ._einop import _einop

class EvaluationInterpreter(ROOFInterpreter):
    '''The evaluation interpreter evaluates an initialized tensor expression. '''    
    def scalar(self, value, payload):
        return value

    def tensor(self, value, payload):
        return payload[0]
    
    def index(self, value, payload):
        return value.symbol
    
    def index_notation(self, tensor, indices, payload):
        return (tensor, indices)
    
    def call(self, f, x, payload):
        return (getattr(np, f)(x[0]),payload[1])

    def pow(self, base, exponent, payload):
        return (np.power(base[0], exponent),payload[1])

    def neg(self, x, payload):
        return (-x[0],payload[1])

    def div(self, lhs, rhs, payload):
        index_map = {}
        def get_mapped_index(idx):
            try:
                return index_map[idx]
            except KeyError:
                index_map[idx] = chr(97 + len(index_map))
                return index_map[idx]
        lhs_mapped = list(map(get_mapped_index, lhs[1]))
        rhs_mapped = list(map(get_mapped_index, rhs[1]))
        lhs_idx =''.join(lhs_mapped)
        rhs_idx =''.join(rhs_mapped)
        return (_einop(f'{lhs_idx},{rhs_idx}', lhs[0], rhs[0],np.divide), payload[1])

    def mul(self, lhs, rhs, payload):
        index_map = {}
        def get_mapped_index(idx):
            try:
                return index_map[idx]
            except KeyError:
                index_map[idx] = chr(97 + len(index_map))
                return index_map[idx]
        lhs_mapped = list(map(get_mapped_index, lhs[1]))
        rhs_mapped = list(map(get_mapped_index, rhs[1]))
        lhs_idx =''.join(lhs_mapped)
        rhs_idx =''.join(rhs_mapped)
        #return (jnp.einsum(f'{lhs_idx},{rhs_idx}', lhs[0], rhs[0]), payload[1])
        return (_einop(f'{lhs_idx},{rhs_idx}', lhs[0], rhs[0],np.multiply), payload[1])

    def add(self, lhs, rhs, payload):
        index_map = {}
        def get_mapped_index(idx):
            try:
                return index_map[idx]
            except KeyError:
                index_map[idx] = chr(97 + len(index_map))
                return index_map[idx]
        lhs_mapped = list(map(get_mapped_index, lhs[1]))
        rhs_mapped = list(map(get_mapped_index, rhs[1]))
        lhs_idx =''.join(lhs_mapped)
        rhs_idx =''.join(rhs_mapped)
        return (_einop(f'{lhs_idx},{rhs_idx}', lhs[0], rhs[0],np.add), payload[1])

    def sub(self, lhs, rhs, payload):
        index_map = {}
        def get_mapped_index(idx):
            try:
                return index_map[idx]
            except KeyError:
                index_map[idx] = chr(97 + len(index_map))
                return index_map[idx]
        lhs_mapped = list(map(get_mapped_index, lhs[1]))
        rhs_mapped = list(map(get_mapped_index, rhs[1]))
        lhs_idx =''.join(lhs_mapped)
        rhs_idx =''.join(rhs_mapped)
        return (_einop(f'{lhs_idx},{rhs_idx}', lhs[0], rhs[0],np.subtract), payload[1])