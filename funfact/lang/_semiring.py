#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._ast import Primitives as P
from ._tsrex import _as_node, TsrEx


def minplus(lhs: TsrEx, rhs: TsrEx):
    return TsrEx(
        P.ein(_as_node(lhs), _as_node(rhs), 6, 'min', 'add', None)
    )


def logsumexp(lhs: TsrEx, rhs: TsrEx):
    return TsrEx(
        P.ein(_as_node(lhs), _as_node(rhs), 6, 'log_sum_exp', 'add', None)
    )


def viterbi(lhs: TsrEx, rhs: TsrEx):
    return TsrEx(
        P.ein(_as_node(lhs), _as_node(rhs), 6, 'max', 'log_add_exp', None)
    )
